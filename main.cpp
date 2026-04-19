// main.cpp — Qwen3.5 MoE CPU text generation
//
// Usage:
//   ./test_qwen35moe --model <path.gguf> [options]
//
// Options:
//   --model          <path>   GGUF model file (required)
//   --prompt         <text>   Input prompt (default: interactive REPL)
//   --temp           <f>      Temperature (default: 0.8; 0 = greedy)
//   --top-p          <f>      Top-p sampling (default: 0.95)
//   --top-k          <N>      Top-k sampling (default: 40; 0 = disabled)
//   --threads        <N>      CPU threads (default: 4)
//   --ctx-size       <N>      KV cache context length (default: auto for single prompt, 2048 for REPL)
//   --no-chat                Disable chat template (pass prompt verbatim)
//   --verbose                Print tokenization and timing info
//   --gpu-mode      <mode>   GPU mode: off|hybrid|full (default: off)
//
// Minimal run example:
//   ./test_qwen35moe \
//     --model Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_M.gguf \
//     --prompt "Hello, who are you?" --ctx-size 128 --temp 0.7
//
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <cctype>
#include <iostream>
#include <unordered_map>
#include "core/gguf_reader.hpp"
#include "model/model.hpp"
#include "model/loader.hpp"
#include "pipeline/recognizer.hpp"
#include "pipeline/tokenizer.hpp"
#include "pipeline/inference.hpp"

// Temperature + top-k + top-p sampling with optional repetition penalty
static int sample(const std::vector<float>& logits, float temperature, float top_p, int top_k, std::mt19937& rng,
    const std::vector<int32_t>& recent_tokens = {}) {
    const int n = (int)logits.size();

    // Apply repetition / presence penalties in logit space.
    std::vector<float> probs(logits.begin(), logits.end());

    // --- 新增：重复惩罚 (Repetition Penalty) ---
    // 设定一个惩罚系数，例如 1.1
    const float penalty = 1.1f;
    for (int32_t tok : recent_tokens) {
        if (probs[tok] > 0) {
            probs[tok] /= penalty; // 如果 logit 是正数，减小它
        } else {
            probs[tok] *= penalty; // 如果 logit 是负数，使其更负
        }
    }
    
    if (temperature <= 0.0f) {
        return (int)(std::max_element(probs.begin(), probs.end()) - probs.begin());
    }

    // Apply temperature
    for (auto& v : probs) v /= temperature;

    // Softmax
    float max_v = *std::max_element(probs.begin(), probs.end());
    double sum = 0.0;
    for (auto& v : probs) { v = std::exp(v - max_v); sum += v; }
    for (auto& v : probs) v /= (float)sum;

    // Build sorted indices
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return probs[a] > probs[b]; });

    // Top-k filter
    int k = (top_k > 0 && top_k < n) ? top_k : n;
    if (k <= 0) k = 1;

    // Top-p filter
    const float effective_top_p = (top_p > 0.0f && top_p < 1.0f) ? top_p : 1.0f;
    if (effective_top_p < 1.0f) {
        float cum = 0.0f;
        int p_cutoff = k;
        for (int i = 0; i < k; i++) {
            cum += probs[idx[i]];
            if (cum >= effective_top_p) { p_cutoff = i + 1; break; }
        }
        k = std::min(k, p_cutoff);
    }
    if (k <= 0) k = 1;

    // Sample from top-k/p
    float total = 0.0f;
    for (int i = 0; i < k; i++) total += probs[idx[i]];
    std::uniform_real_distribution<float> dist(0.0f, total);
    float r = dist(rng);
    float c = 0.0f;
    for (int i = 0; i < k; i++) {
        c += probs[idx[i]];
        if (r <= c) return idx[i];
    }
    return idx[k - 1];
}

// ============================================================
// CLI helpers
// ============================================================

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s --model <path.gguf> [options]\n"
        "\n"
        "Options:\n"
        "  --model          <path>   GGUF model file (required)\n"
        "  --prompt         <text>   Input prompt (omit for REPL mode)\n"
        "  --temp           <f>      Sampling temperature (default: GGUF metadata or 0.8; 0=greedy)\n"
        "  --top-p          <f>      Top-p sampling threshold (default: GGUF metadata or 0.95)\n"
        "  --top-k          <N>      Top-k sampling (default: GGUF metadata or 40; 0=off)\n"
        "  --threads        <N>      CPU threads (default: 4)\n"
        "  --ctx-size       <N>      KV cache context length (default: auto for single prompt, 2048 for REPL)\n"
        "  --seed           <N>      RNG seed (default: random)\n"
        "  --no-chat                Pass prompt verbatim (no chat template)\n"
        "  --verbose                Show tokenization and timing info\n"
        "  --gpu-mode      <mode>   GPU mode: off|hybrid|full (default: off)\n"
        "\n"
        "Example:\n"
        "  %s --model model.gguf --prompt \"Hello!\" --ctx-size 128 --temp 0.7\n",
        prog, prog);
}

// ============================================================
// Generation loop
// ============================================================

static void generate(InferenceEngine& engine, BPETokenizer& tokenizer, const std::string& prompt_text, float temperature, float top_p, 
    int top_k, bool use_chat, bool verbose, int ctx_size, std::mt19937& rng) {
    engine.reset_state();

    std::string formatted = prompt_text;
    if (use_chat) {
        formatted = tokenizer.make_chat_prompt(prompt_text);
    }
    if (verbose) fprintf(stderr, "[main] Formatted prompt:%s\n", formatted.c_str());
    std::vector<int32_t> tokens = tokenizer.encode(formatted);
    if (tokens.empty()) {
        fprintf(stderr, "[main] WARNING: prompt encoded to 0 tokens\n");
        return;
    }

    if (verbose) {
        fprintf(stderr, "[main] Prompt: \"%s\"\n", formatted.c_str());
        fprintf(stderr, "[main] Prompt tokens (%zu) ctx-size=%d: ", tokens.size(), ctx_size);
        for (int t : tokens) fprintf(stderr, "%d ", t);
        fprintf(stderr, "\n");
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    int n_generated = 0;
    std::vector<int32_t> recent_tokens;
    std::vector<int32_t> next_input = tokens;
    auto t_fwd_start = t_start;
    auto t_fwd_end = t_start;
    while (n_generated < ctx_size) {
        t_fwd_start = std::chrono::high_resolution_clock::now();
        std::vector<float> logits = engine.forward(next_input);
        t_fwd_end = std::chrono::high_resolution_clock::now();

        if (logits.empty()) {
            fprintf(stderr, "\n[main] ERROR: forward() returned empty logits\n");
            break;
        }

        int next_token = sample(logits, temperature, top_p, top_k, rng, recent_tokens);
        if (tokenizer.is_stop_token(next_token)) {
            if (verbose) fprintf(stderr, "\n[main] Stop token %d reached\n", next_token);
            break;
        }

        const std::string piece = tokenizer.decode_one(next_token);
        printf("%s", piece.c_str());
        fflush(stdout);

        next_input = {next_token};
        recent_tokens.push_back(next_token);
        n_generated++;
    }

    if (verbose) {
        auto t_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        fprintf(stderr, "\n[main] Generated %d tokens in %.1f ms (%.1f ms/tok)\n", n_generated, total_ms, n_generated > 0 ? total_ms / n_generated : 0.0);
        double ms = std::chrono::duration<double, std::milli>(t_fwd_end - t_fwd_start).count();
        fprintf(stderr, "[main] First token forward: %.1f ms (seq_len=%zu)\n", ms, tokens.size());
    }
}

// ============================================================
// main()\n// ============================================================

int main(int argc, char* argv[]) {
    // ---- Defaults ----
    std::string model_path;
    std::string prompt;
    float       temperature  = -1.0f;
    float       top_p        = -1.0f;
    int         top_k        = -1;
    int         n_threads    = 1;
    int         ctx_size     = 1024;
    bool        use_chat     = true;
    bool        verbose      = true;
    bool        repl_mode    = false;
    GpuMode     gpu_mode     = GpuMode::Off;
    uint64_t    rng_seed     = std::random_device{}(); // random by default

    // ---- Parse args ----
    for (int i = 1; i < argc; i++) {
        auto arg = [&](const char* flag) {
            return strcmp(argv[i], flag) == 0;
        };
        auto next = [&](const char* flag) -> const char* {
            if (i + 1 < argc) return argv[++i];
            fprintf(stderr, "ERROR: %s requires an argument\n", flag);
            exit(1);
        };

        if (arg("--model"))          model_path  = next("--model");
        else if (arg("--prompt"))    prompt      = next("--prompt");
        else if (arg("--temp"))      { temperature = (float)atof(next("--temp")); }
        else if (arg("--top-p"))     { top_p = (float)atof(next("--top-p")); }
        else if (arg("--top-k"))     { top_k = atoi(next("--top-k")); }
        else if (arg("--threads"))        n_threads   = atoi(next("--threads"));
        else if (arg("--ctx-size"))   ctx_size     = atoi(next("--ctx-size"));
        else if (arg("--seed"))      rng_seed    = (uint64_t)atoll(next("--seed"));
        else if (arg("--no-chat"))   use_chat    = false;
        else if (arg("--verbose"))   verbose     = true;
        else if (arg("--gpu-mode")) {
            const char* mode = next("--gpu-mode");
            if (strcmp(mode, "off") == 0) {
                gpu_mode = GpuMode::Off;
            } else if (strcmp(mode, "hybrid") == 0) {
                gpu_mode = GpuMode::Hybrid;
            } else if (strcmp(mode, "full") == 0) {
                gpu_mode = GpuMode::Full;
            } else {
                gpu_mode    = GpuMode::Off;
            }
        }
        else if (arg("-h") || arg("--help")) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "WARNING: unknown argument: %s\n", argv[i]);
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "ERROR: --model is required\n\n");
        print_usage(argv[0]);
        return 1;
    }

    if (prompt.empty()) {
        repl_mode = true;
    }

    fprintf(stderr, "[main] Loading model: %s\n", model_path.c_str());
    Recognizer recognizer;
    if (!recognizer.init(model_path, verbose, gpu_mode)) {
        fprintf(stderr, "[main] ERROR: failed to load model: %s\n", recognizer.last_error().c_str());
        return 1;
    }

    Qwen35moeModel* model = recognizer.model();
    fprintf(stderr, "[main] Model loaded. vocab_size=%lld n_layers=%u\n", model->weights.token_embd->ne[1], model->config.qwen35moe.block_count);

    BPETokenizer tokenizer;
    if (!tokenizer.load(model->config.tokenizer)) {
        fprintf(stderr, "[main] ERROR: failed to load tokenizer\n");
        return 1;
    }

    if (temperature < 0.0f) {
        temperature = model->config.general.sampling_temp > 0.0f ? model->config.general.sampling_temp : 0.8f;
    }
    if (top_p < 0.0f) {
        top_p = model->config.general.sampling_top_p > 0.0f ? model->config.general.sampling_top_p : 0.95f;
    }
    if (top_k < 0) {
        top_k = model->config.general.sampling_top_k > 0 ? model->config.general.sampling_top_k : 40;
    }
    const int model_ctx_size = model->config.qwen35moe.context_length;
    if (model_ctx_size > 0 && ctx_size > model_ctx_size) {
        fprintf(stderr, "[main] WARNING: requested ctx-size %d exceeds model context_length %d; clamping\n", ctx_size, model_ctx_size);
        ctx_size = model_ctx_size;
    }

    InferenceEngine engine(*model, gpu_mode != GpuMode::Off ? recognizer.reader() : nullptr, n_threads, ctx_size, gpu_mode);
    std::mt19937 rng(rng_seed);
    if (verbose) {
        fprintf(stderr, "[main] RNG seed: %llu\n", (unsigned long long)rng_seed);
        fprintf(stderr, "[main] Sampling params: temp=%.3f top_p=%.3f top_k=%d\n", temperature, top_p, top_k);
        fprintf(stderr, "[main] Chat prompt mode: %s\n", use_chat ? "enabled" : "disabled");
        fprintf(stderr, "[main] Chat template: %s\n", tokenizer.has_chat_template() ? "GGUF template + Qwen fallback" : "Qwen fallback only");
        fprintf(stderr, "[main] Effective ctx-size: %d (model max %d)\n", ctx_size, model_ctx_size);
    }

    if (!repl_mode) {
        generate(engine, tokenizer, prompt, temperature, top_p, top_k, use_chat, verbose, ctx_size, rng);
    } else {
        // Interactive REPL
        fprintf(stderr, "[main] Interactive mode. Type your prompt (empty line to quit).\n");
        fprintf(stderr, "       Ctrl+D or empty input to exit.\n\n");

        std::string line;
        while (true) {
            printf("> ");
            fflush(stdout);
            if (!std::getline(std::cin, line) || line.empty()) {
                printf("\n");
                break;
            }
            generate(engine, tokenizer, line, temperature, top_p, top_k, use_chat, verbose, ctx_size, rng);
        }
    }

    return 0;
}