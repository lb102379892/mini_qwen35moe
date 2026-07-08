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
//   --threads        <N>      CPU threads for decode (default: 4)
//   --threads-batch  <N>      CPU threads for prefill/batch (default: same as --threads)
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
#include <unordered_set>
#include "model/model.h"
#include "pipeline/chat.h"
#include "api/http_server.h"
#include "model/dev_defaults.h"

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
        "  --threads        <N>      CPU threads for decode (default: 4)\n"
        "  --threads-batch  <N>      CPU threads for prefill/batch (default: same as --threads)\n"
        "  --ctx-size       <N>      KV cache context length (default: auto for single prompt, 2048 for REPL)\n"
        "  --n-batch        <N>      Logical prompt batch tokens (default: ctx-size)\n"
        "  --n-ubatch       <N>      Physical micro-batch tokens (default: n-batch)\n"
        "  --no-mmap               Copy weights then unmap file (default: mmap during load;\n"
        "                          GPU/auto release file after upload; CPU may zero-copy)\n"
        "  --parallel       <N>      Max concurrent decode sequences / KV slots (default: 4)\n"
        "  --seed           <N>      RNG seed (default: random)\n"
        "  --no-chat                Pass prompt verbatim (no chat template)\n"
        "  --verbose                Show tokenization and timing info\n"
        "  --gpu-mode      <mode>   GPU mode: off|hybrid|full (default: off)\n"
        "  --flash-attn             Use ggml_flash_attn_ext for attention\n"
        "  --paged-kv               Enable phase-1 paged KV cache (auto-disabled\n"
        "                           under --dev-mode auto / mixed GPU+CPU because\n"
        "                           fused flash-attention decode is CUDA-only there;\n"
        "                           set QWEN35MOE_PAGED_KV_FORCE_MIXED=1 to override\n"
        "                           for A/B testing — typically SLOWER under mixed)\n"
        "  --paged-kv-block <N>     Paged KV block size in tokens (default: 16)\n"
        "  --host           <addr>  HTTP bind address (default: 0.0.0.0)\n"
        "  --port           <N>     HTTP listen port (default: 6666)\n"
        "  --gpu-layer, --n-gpu-layers <N>\n"
        "                          Transformer layers on GPU in auto mode\n"
        "                          (llama.cpp --n-gpu-layers; 0=VRAM budget only)\n"
        "  --override-tensor, -ot <pat=DEV>\n"
        "                          Override tensor buffer (regex=CPU|GPU); repeatable\n"
        "  --gpu-id         <N>     CUDA device index (default: 0)\n"
        "  --mtp                    Enable MTP/NextN speculative decoding using the\n"
        "                           model's built-in nextn draft head (requires the\n"
        "                           GGUF to carry an MTP block; cpu/gpu + --parallel 1\n"
        "                           + greedy sampling; otherwise falls back silently)\n"
        "  --spec-draft-n-max, --mtp-n-max <N>   Max draft tokens per MTP step (default: 6)\n"
        "  --dev-mode       <mode>  cpu|gpu|auto (default: cpu; applies mode-specific perf defaults)\n"
        "\n"
        "Example:\n"
        "  %s --model model.gguf --prompt \"Hello!\" --ctx-size 128 --temp 0.7\n",
        prog, prog);
}

// ============================================================
// Generation loop
// ============================================================

// ============================================================
// main()\n// ============================================================

int main(int argc, char* argv[]) {
    // ---- Defaults ----
    CParam param;

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

        if (arg("--model"))          
            param.model_path  = next("--model");
        else if (arg("--temp")) { 
            param.temperature = (float)atof(next("--temp"));
            param.user_flags.temperature = true;
        } else if (arg("--top-p")) { 
            param.top_p = (float)atof(next("--top-p"));
            param.user_flags.top_p = true;
        } else if (arg("--top-k")) { 
            param.top_k = atoi(next("--top-k"));
            param.user_flags.top_k = true;
        } else if (arg("--threads")) {
            param.n_threads = std::max(1, atoi(next("--threads")));
            param.user_flags.n_threads = true;
        } else if (arg("--threads-batch")) {
            param.n_threads_batch = std::max(1, atoi(next("--threads-batch")));
            param.user_flags.n_threads_batch = true;
        } else if (arg("--ctx-size"))
            param.ctx_size = atoi(next("--ctx-size"));
        else if (arg("--n-batch"))
            param.n_batch = atoi(next("--n-batch"));
        else if (arg("--n-ubatch"))
            param.n_ubatch = atoi(next("--n-ubatch"));
        else if (arg("--parallel") || arg("--max-seqs")) {
            param.max_sequences = std::max(1, atoi(next(argv[i])));
            param.user_flags.max_sequences = true;
        }
        else if (arg("--seed"))
            param.rng_seed = (uint64_t)atoll(next("--seed"));
        else if (arg("--no-chat"))
            param.use_chat = false;
        else if (arg("--verbose"))
            param.verbose = true;
        else if (arg("--flash-attn")) {
            param.flash_attention = true;
            param.user_flags.flash_attention = true;
        }
        else if (arg("--paged-kv")) {
            param.enable_paged_kv = true;
            param.user_flags.enable_paged_kv = true;
        }
        else if (arg("--no-mmap"))
            param.no_mmap = true;
        else if (arg("--mtp"))
            param.mtp = true;
        else if (arg("--spec-draft-n-max") || arg("--mtp-n-max"))
            param.spec_draft_n_max = std::max(1, atoi(next(argv[i])));
        else if (arg("--paged-kv-block"))
            param.paged_kv_block_size = static_cast<uint32_t>(std::max(1, atoi(next("--paged-kv-block"))));
        else if (arg("--gpu-layer") || arg("--n-gpu-layers"))
            param.gpu_layer = static_cast<size_t>(atoi(next(argv[i])));
        else if (arg("--override-tensor") || arg("-ot"))
            param.tensor_overrides.push_back(next(argv[i]));
        else if (arg("--host"))
            param.http_host = next("--host");
        else if (arg("--port"))
            param.http_port = atoi(next("--port"));
        else if (arg("--gpu-id"))
            param.gpu_id = atoi(next("--gpu-id"));
        else if (arg("--dev-mode")) {
            const char* mode = next("--dev-mode");
            param.user_flags.dev_mode = true;
            if (strcmp(mode, "cpu") == 0) {
                param.dev_mode = DevMode::CPU_MODE;
            } else if (strcmp(mode, "gpu") == 0) {
                param.dev_mode = DevMode::GPU_MODE;
            } else if (strcmp(mode, "auto") == 0) {
                param.dev_mode = DevMode::AUTO_MODE;
            } else {
                param.dev_mode = DevMode::AUTO_MODE;
            }
        }
        else if (arg("-h") || arg("--help")) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "WARNING: unknown argument: %s\n", argv[i]);
        }
    }

    if (param.model_path.empty()) {
        fprintf(stderr, "ERROR: --model is required\n\n");
        print_usage(argv[0]);
        return 1;
    }

    if (param.prompt.empty()) {
        param.repl_mode = true;
    }
    if (param.n_batch <= 0) {
        param.n_batch = param.ctx_size;
    }
    if (param.n_ubatch <= 0) {
        param.n_ubatch = param.n_batch;
    }
    if (param.max_sequences <= 0) {
        param.max_sequences = 1;
    }

    fprintf(stderr, "[main] Loading model: %s\n", param.model_path.c_str());

    ChatEngine chat;
    if (!chat.init(param)) {
        fprintf(stderr, "Engine initialization failed\n");
        return 1;
    }

    HttpServer server;
    if (!server.init(param.http_host, param.http_port, &chat)) {
        fprintf(stderr, "HTTP server initialization failed: %s\n", server.last_error().c_str());
        return 1;
    }

    fprintf(stderr, "Starting HTTP server on %s:%d\n", param.http_host.c_str(), param.http_port);
    server.run();

    fprintf(stderr, "llama server ready\n");
    return 0;
}
