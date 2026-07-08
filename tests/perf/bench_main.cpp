// tests/perf/bench_main.cpp
//
// C++ benchmark harness for `test_qwen35moe` — measures the impact of CLI
// arguments on inference speed (t/s) by calling ChatEngine directly. Unlike
// the HTTP server, this:
//
//   * Re-uses the same CParam → ChatEngine::init path, so every flag that
//     test_qwen35moe accepts (--dev-mode, --flash-attn, --paged-kv,
//     --parallel, --n-batch, --n-ubatch, --gpu-layer, ...) is honored 1:1.
//   * Skips HTTP/JSON serialization — what we measure is purely
//     prefill + decode + sampling.
//   * Reads precise per-request timings out of TimingStats (populated by
//     ChatEngine::log_sequence_timing), then aggregates min/median/p95
//     across N runs and emits one Markdown row.
//
// Typical use is to invoke this binary many times from a shell wrapper
// (see tests/perf/run_bench_cpp.sh), each invocation pinned to one
// parameter combination. That keeps the C++ side trivial (one process =
// one config) while leaving matrix orchestration to shell.
//
// Example:
//
//   ./qwen35moe_bench --model /path/Q5_K_M.gguf \
//       --dev-mode gpu --parallel 4 --flash-attn --paged-kv \
//       --bench-runs 8 --bench-warmup 2 --bench-max-tokens 128 \
//       --bench-concurrency 4 --bench-label gpu_pkv_fa_p4 \
//       --bench-output /tmp/results.md
//

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <vector>

#include "model/common.h"
#include "model/dev_defaults.h"
#include "model/model.h"
#include "pipeline/chat.h"

namespace {

struct BenchOpts {
    int          runs        = 8;
    int          warmup      = 2;
    int          concurrency = 1;
    int          max_tokens  = 128;
    int          prompt_repeat = 1;
    std::string  prompt      = "Explain in detail how a transformer language model works.";
    std::string  label;
    std::string  output_path; // empty => stdout
    std::string  dump_response_path; // optional: write last measured response
};

void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s --model <path.gguf> [test_qwen35moe options] [bench options]\n"
        "\n"
        "test_qwen35moe options (subset, all behave exactly as in the main CLI):\n"
        "  --dev-mode {cpu|gpu|auto}   Device scheduling mode (default: cpu)\n"
        "  --gpu-layer, --n-gpu-layers <N>  Transformer layers offloaded to GPU\n"
        "                              (llama.cpp --n-gpu-layers semantics; 0=VRAM-only)\n"
        "  --override-tensor, -ot <pat=DEV> Override tensor buffer (regex=CPU|GPU); repeatable\n"
        "  --parallel     <N>          Max concurrent decode slots (default: 4)\n"
        "  --n-batch      <N>          Logical prompt batch tokens\n"
        "  --n-ubatch     <N>          Physical micro-batch tokens\n"
        "  --ctx-size     <N>          KV cache context length (default: 4096)\n"
        "  --threads      <N>          CPU threads for decode (default: 4)\n"
        "  --threads-batch <N>         CPU threads for prefill/batch (default: same as --threads)\n"
        "  --flash-attn                Use ggml_flash_attn_ext\n"
        "  --paged-kv                  Enable phase-1 paged KV cache\n"
        "  --paged-kv-block <N>        Paged KV block size (default: 16)\n"
        "  --temp <f> --top-p <f> --top-k <N>   Sampling params\n"
        "  --no-mmap                   Copy weights, then unmap\n"
        "  --seed <N>                  RNG seed\n"
        "\n"
        "Bench options:\n"
        "  --bench-runs     <N>        Measured request count (default: 8)\n"
        "  --bench-warmup   <N>        Warmup request count, not counted (default: 2)\n"
        "  --bench-concurrency <N>     # of in-flight requests (default: 1)\n"
        "                              Must be <= --parallel\n"
        "  --bench-max-tokens  <N>     Max new tokens per request (default: 128)\n"
        "  --bench-prompt   \"text\"     Prompt to send (default: built-in)\n"
        "  --bench-prompt-repeat <N>   Repeat the prompt body N times to scale\n"
        "                              prefill length (default: 1)\n"
        "  --bench-label    <name>     Free-form label printed in the result row\n"
        "  --bench-output   <path>     Append a Markdown row to this file.\n"
        "                              If the file does not exist, a header is\n"
        "                              written first. Default: stdout.\n"
        "  --bench-dump-response <path> Write the last measured response text\n"
        "                              to this file (for pybind parity checks).\n"
        "\n"
        "Example matrix run via the wrapper:\n"
        "  MODEL=/path.gguf tests/perf/run_bench_cpp.sh\n",
        prog);
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    if (v.size() == 1) return v[0];
    const double idx = p * static_cast<double>(v.size() - 1);
    const size_t lo  = static_cast<size_t>(std::floor(idx));
    const size_t hi  = static_cast<size_t>(std::ceil(idx));
    const double frac = idx - static_cast<double>(lo);
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

double median(std::vector<double> v) { return percentile(std::move(v), 0.5); }

bool file_exists_nonempty(const std::string& path) {
    if (path.empty()) return true; // stdout
    struct stat st {};
    return ::stat(path.c_str(), &st) == 0 && st.st_size > 0;
}

const char* dev_mode_str(DevMode m) {
    switch (m) {
        case DevMode::CPU_MODE:  return "cpu";
        case DevMode::GPU_MODE:  return "gpu";
        case DevMode::AUTO_MODE: return "auto";
    }
    return "?";
}

// One Markdown column ordering, used for both the header and rows. Keep them
// in lockstep — if you reorder one, reorder the other.
constexpr const char* kHeader =
    "| label | dev | parallel | flash | paged_kv | n_batch | n_ubatch | conc"
    " | prompt_tok | gen_tok | runs"
    " | prefill tok/s (med) | prefill tok/s (p95)"
    " | decode tok/s (med) | decode tok/s (p95)"
    " | e2e ms (med) | wall tok/s |";
constexpr const char* kHeaderSep =
    "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|";

} // namespace

int main(int argc, char* argv[]) {
    CParam    param;
    BenchOpts bench;

    // Bench harness owns its sequence — the HTTP `repl_mode` flag isn't
    // meaningful here.
    param.repl_mode = false;
    param.use_chat  = true;
    // The bench is unattended; suppress the verbose per-step logs unless
    // the user explicitly opts in via --verbose.
    param.verbose = false;

    for (int i = 1; i < argc; ++i) {
        auto arg = [&](const char* flag) { return std::strcmp(argv[i], flag) == 0; };
        auto next = [&](const char* flag) -> const char* {
            if (i + 1 < argc) return argv[++i];
            std::fprintf(stderr, "ERROR: %s requires an argument\n", flag);
            std::exit(1);
        };

        // ----- shared flags with main.cpp -----
        if      (arg("--model"))          param.model_path = next("--model");
        else if (arg("--temp")) {
            param.temperature = static_cast<float>(std::atof(next("--temp")));
            param.user_flags.temperature = true;
        } else if (arg("--top-p")) {
            param.top_p = static_cast<float>(std::atof(next("--top-p")));
            param.user_flags.top_p = true;
        } else if (arg("--top-k")) {
            param.top_k = std::atoi(next("--top-k"));
            param.user_flags.top_k = true;
        } else if (arg("--threads")) {
            param.n_threads = std::max(1, std::atoi(next("--threads")));
            param.user_flags.n_threads = true;
        } else if (arg("--threads-batch")) {
            param.n_threads_batch = std::max(1, std::atoi(next("--threads-batch")));
            param.user_flags.n_threads_batch = true;
        }
        else if (arg("--ctx-size"))       param.ctx_size = std::atoi(next("--ctx-size"));
        else if (arg("--n-batch"))        param.n_batch = std::atoi(next("--n-batch"));
        else if (arg("--n-ubatch"))       param.n_ubatch = std::atoi(next("--n-ubatch"));
        else if (arg("--parallel") || arg("--max-seqs")) {
            param.max_sequences = std::max(1, std::atoi(next(argv[i])));
            param.user_flags.max_sequences = true;
        }
        else if (arg("--seed"))           param.rng_seed = static_cast<uint64_t>(std::atoll(next("--seed")));
        else if (arg("--no-chat"))        param.use_chat = false;
        else if (arg("--verbose"))        param.verbose = true;
        else if (arg("--flash-attn")) {
            param.flash_attention = true;
            param.user_flags.flash_attention = true;
        } else if (arg("--paged-kv")) {
            param.enable_paged_kv = true;
            param.user_flags.enable_paged_kv = true;
        }
        else if (arg("--no-mmap"))        param.no_mmap = true;
        else if (arg("--paged-kv-block")) param.paged_kv_block_size =
            static_cast<uint32_t>(std::max(1, std::atoi(next("--paged-kv-block"))));
        else if (arg("--gpu-layer") || arg("--n-gpu-layers"))
            param.gpu_layer = static_cast<size_t>(std::atoi(next(argv[i])));
        else if (arg("--override-tensor") || arg("-ot"))
            param.tensor_overrides.push_back(next(argv[i]));
        else if (arg("--gpu-id"))         param.gpu_id = std::atoi(next("--gpu-id"));
        else if (arg("--dev-mode")) {
            const char* mode = next("--dev-mode");
            param.user_flags.dev_mode = true;
            if      (std::strcmp(mode, "cpu") == 0)  param.dev_mode = DevMode::CPU_MODE;
            else if (std::strcmp(mode, "gpu") == 0)  param.dev_mode = DevMode::GPU_MODE;
            else if (std::strcmp(mode, "auto") == 0) param.dev_mode = DevMode::AUTO_MODE;
            else {
                std::fprintf(stderr, "ERROR: unknown --dev-mode '%s'\n", mode);
                return 1;
            }
        }
        // ----- bench-only flags -----
        else if (arg("--bench-runs"))         bench.runs        = std::max(1, std::atoi(next("--bench-runs")));
        else if (arg("--bench-warmup"))       bench.warmup      = std::max(0, std::atoi(next("--bench-warmup")));
        else if (arg("--bench-concurrency"))  bench.concurrency = std::max(1, std::atoi(next("--bench-concurrency")));
        else if (arg("--bench-max-tokens"))   bench.max_tokens  = std::max(1, std::atoi(next("--bench-max-tokens")));
        else if (arg("--bench-prompt"))       bench.prompt      = next("--bench-prompt");
        else if (arg("--bench-prompt-repeat")) bench.prompt_repeat =
            std::max(1, std::atoi(next("--bench-prompt-repeat")));
        else if (arg("--bench-label"))        bench.label       = next("--bench-label");
        else if (arg("--bench-output"))       bench.output_path = next("--bench-output");
        else if (arg("--bench-dump-response")) bench.dump_response_path = next("--bench-dump-response");
        else if (arg("-h") || arg("--help")) { print_usage(argv[0]); return 0; }
        else {
            std::fprintf(stderr, "WARNING: unknown argument: %s\n", argv[i]);
        }
    }

    if (param.model_path.empty()) {
        std::fprintf(stderr, "ERROR: --model is required\n\n");
        print_usage(argv[0]);
        return 1;
    }
    if (bench.concurrency > param.max_sequences) {
        std::fprintf(stderr,
            "ERROR: --bench-concurrency %d exceeds --parallel %d. Raise --parallel or lower --bench-concurrency.\n",
            bench.concurrency, param.max_sequences);
        return 1;
    }
    if (param.n_batch  <= 0) param.n_batch  = param.ctx_size;
    if (param.n_ubatch <= 0) param.n_ubatch = param.n_batch;

    // Build the bench prompt (optional repetition for scaling prefill length).
    std::string prompt;
    prompt.reserve(bench.prompt.size() * static_cast<size_t>(bench.prompt_repeat) + 16);
    for (int r = 0; r < bench.prompt_repeat; ++r) {
        prompt += bench.prompt;
        if (r + 1 < bench.prompt_repeat) prompt += "\n";
    }

    std::fprintf(stderr, "[bench] model=%s dev_mode=%s parallel=%d flash=%d paged_kv=%d "
                         "n_batch=%d n_ubatch=%d concurrency=%d runs=%d warmup=%d max_tokens=%d\n",
        param.model_path.c_str(), dev_mode_str(param.dev_mode), param.max_sequences,
        param.flash_attention ? 1 : 0, param.enable_paged_kv ? 1 : 0,
        param.n_batch, param.n_ubatch,
        bench.concurrency, bench.runs, bench.warmup, bench.max_tokens);

    ChatEngine engine;
    if (!engine.init(param)) {
        std::fprintf(stderr, "ERROR: ChatEngine init failed\n");
        return 1;
    }

    auto run_batch = [&](int total_requests, std::vector<TimingStats>& out_stats,
                         double& wall_ms_out, bool collect,
                         std::string* last_response_out) {
        const auto wall_start = std::chrono::steady_clock::now();

        // Distribute work round-robin across concurrent driver threads.
        std::atomic<int> next_idx{0};
        std::mutex       stats_mu;
        std::vector<TimingStats> stats;
        std::atomic<bool> any_failed{false};
        std::string last_response;
        std::mutex response_mu;

        const int n_threads = std::min(bench.concurrency, total_requests);
        std::vector<std::thread> drivers;
        drivers.reserve(static_cast<size_t>(n_threads));
        for (int t = 0; t < n_threads; ++t) {
            drivers.emplace_back([&] {
                while (true) {
                    const int idx = next_idx.fetch_add(1);
                    if (idx >= total_requests) break;

                    std::string  response;
                    TimingStats  timing;
                    const bool ok = engine.run_complete(prompt, bench.max_tokens, response, timing);
                    if (!ok) any_failed.store(true);
                    if (collect) {
                        std::lock_guard<std::mutex> g(stats_mu);
                        stats.push_back(timing);
                        std::lock_guard<std::mutex> rg(response_mu);
                        last_response = std::move(response);
                    }
                }
            });
        }
        for (auto& th : drivers) th.join();

        wall_ms_out = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wall_start).count();
        if (collect) {
            out_stats = std::move(stats);
            if (last_response_out != nullptr) {
                *last_response_out = std::move(last_response);
            }
        }
        return !any_failed.load();
    };

    // Warmup — discard timings. This both populates the cached decode/prefill
    // graphs and pages the model into the caches that matter (KV slab,
    // CUDA kernels, etc.) so the measured runs reflect steady-state cost.
    if (bench.warmup > 0) {
        std::vector<TimingStats> _;
        double wall_warmup_ms = 0.0;
        std::fprintf(stderr, "[bench] running %d warmup request(s)...\n", bench.warmup);
        if (!run_batch(bench.warmup, _, wall_warmup_ms, /*collect=*/false, nullptr)) {
            std::fprintf(stderr, "ERROR: warmup failed\n");
            return 2;
        }
    }

    std::vector<TimingStats> samples;
    double wall_ms = 0.0;
    std::string measured_response;
    std::fprintf(stderr, "[bench] running %d measured request(s)...\n", bench.runs);
    if (!run_batch(bench.runs, samples, wall_ms, /*collect=*/true, &measured_response)) {
        std::fprintf(stderr, "ERROR: one or more measured runs failed\n");
        return 2;
    }
    if (samples.empty()) {
        std::fprintf(stderr, "ERROR: no samples collected\n");
        return 2;
    }

    if (!bench.dump_response_path.empty()) {
        std::ofstream dump_out(bench.dump_response_path, std::ios::trunc);
        if (!dump_out) {
            std::fprintf(stderr, "ERROR: failed to open --bench-dump-response %s\n",
                bench.dump_response_path.c_str());
            return 3;
        }
        dump_out << measured_response;
        std::fprintf(stderr, "[bench] wrote response dump to %s\n",
            bench.dump_response_path.c_str());
    }

    // Aggregate. We report:
    //   * per-request prefill / decode throughput stats (med + p95) — these
    //     reflect "what a user sees on one request".
    //   * wall tok/s — total generated tokens / wall clock — reflects what
    //     the system delivers at the chosen concurrency level.
    std::vector<double> prefill_tps, decode_tps, e2e_ms;
    prefill_tps.reserve(samples.size());
    decode_tps.reserve(samples.size());
    e2e_ms.reserve(samples.size());
    size_t total_prompt = 0, total_gen = 0;
    for (const auto& s : samples) {
        if (!s.timed) continue;
        if (s.prefill_ms > 0 && s.prompt_tokens > 0) {
            prefill_tps.push_back(static_cast<double>(s.prompt_tokens) * 1000.0 / s.prefill_ms);
        }
        if (s.decode_ms > 0 && s.generated_tokens > 0) {
            decode_tps.push_back(static_cast<double>(s.generated_tokens) * 1000.0 / s.decode_ms);
        }
        if (s.inference_ms > 0) e2e_ms.push_back(s.inference_ms);
        total_prompt += s.prompt_tokens;
        total_gen    += s.generated_tokens;
    }
    const double prefill_med = median(prefill_tps);
    const double prefill_p95 = percentile(prefill_tps, 0.95);
    const double decode_med  = median(decode_tps);
    const double decode_p95  = percentile(decode_tps, 0.95);
    const double e2e_med     = median(e2e_ms);
    const double wall_tps    = wall_ms > 0 ? static_cast<double>(total_gen) * 1000.0 / wall_ms : 0.0;

    // Build the result row.
    auto fmt = [](double v, int prec = 2) {
        std::ostringstream o;
        o << std::fixed << std::setprecision(prec) << v;
        return o.str();
    };
    const size_t prompt_tok_med = samples.empty() ? 0 : samples[samples.size() / 2].prompt_tokens;
    const size_t gen_tok_med    = samples.empty() ? 0 : samples[samples.size() / 2].generated_tokens;

    std::ostringstream row;
    row << "| " << (bench.label.empty() ? "(unlabeled)" : bench.label)
        << " | " << dev_mode_str(param.dev_mode)
        << " | " << param.max_sequences
        << " | " << (param.flash_attention ? "on" : "off")
        << " | " << (param.enable_paged_kv ? "on" : "off")
        << " | " << param.n_batch
        << " | " << param.n_ubatch
        << " | " << bench.concurrency
        << " | " << prompt_tok_med
        << " | " << gen_tok_med
        << " | " << bench.runs
        << " | " << fmt(prefill_med)
        << " | " << fmt(prefill_p95)
        << " | " << fmt(decode_med)
        << " | " << fmt(decode_p95)
        << " | " << fmt(e2e_med)
        << " | " << fmt(wall_tps)
        << " |";

    // Write the row. If --bench-output is set and the file is empty/missing,
    // emit the Markdown header first so the file is a self-contained table.
    if (bench.output_path.empty()) {
        std::printf("\n%s\n%s\n%s\n", kHeader, kHeaderSep, row.str().c_str());
    } else {
        const bool needs_header = !file_exists_nonempty(bench.output_path);
        std::ofstream out(bench.output_path, std::ios::app);
        if (!out) {
            std::fprintf(stderr, "ERROR: failed to open --bench-output %s\n", bench.output_path.c_str());
            return 3;
        }
        if (needs_header) {
            out << kHeader     << "\n";
            out << kHeaderSep  << "\n";
        }
        out << row.str() << "\n";
        std::fprintf(stderr, "[bench] appended row to %s\n", bench.output_path.c_str());
        std::fprintf(stderr, "%s\n", row.str().c_str());
    }

    return 0;
}
