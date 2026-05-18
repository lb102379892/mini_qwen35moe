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
#include <unordered_set>
#include "model/model.h"
#include "pipeline/chat.h"
#include "api/http_server.h"

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
        "  --n-batch        <N>      Logical prompt batch tokens (default: ctx-size)\n"
        "  --n-ubatch       <N>      Physical micro-batch tokens (default: n-batch)\n"
        "  --seed           <N>      RNG seed (default: random)\n"
        "  --no-chat                Pass prompt verbatim (no chat template)\n"
        "  --verbose                Show tokenization and timing info\n"
        "  --gpu-mode      <mode>   GPU mode: off|hybrid|full (default: off)\n"
        "  --flash-attn             Use ggml_flash_attn_ext for attention\n"
        "  --paged-kv               Enable phase-1 paged KV cache\n"
        "  --paged-kv-block <N>     Paged KV block size in tokens (default: 16)\n"
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
    std::string model_path;
    std::string prompt;
    float       temperature  = 0.7f;
    float       top_p        = 0.9f;
    int         top_k        = 50;
    int         n_threads    = 4;
    int         ctx_size     = 4096;
    int         n_batch      = -1;
    int         n_ubatch     = -1;
    size_t      gpu_layer    = 0;
    bool        use_chat     = true;
    bool        verbose      = true;
    bool        repl_mode    = false;
    bool        flash_attention = false;
    bool        enable_paged_kv = false;
    uint32_t    paged_kv_block_size = 16;
    DevMode     dev_mode     = DevMode::CPU_MODE;
    uint64_t    rng_seed     = 79977733;

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
        else if (arg("--temp"))      { temperature = (float)atof(next("--temp")); }
        else if (arg("--top-p"))     { top_p = (float)atof(next("--top-p")); }
        else if (arg("--top-k"))     { top_k = atoi(next("--top-k")); }
        else if (arg("--threads"))        n_threads   = atoi(next("--threads"));
        else if (arg("--ctx-size"))   ctx_size     = atoi(next("--ctx-size"));
        else if (arg("--n-batch"))   n_batch      = atoi(next("--n-batch"));
        else if (arg("--n-ubatch"))  n_ubatch     = atoi(next("--n-ubatch"));
        else if (arg("--seed"))      rng_seed    = (uint64_t)atoll(next("--seed"));
        else if (arg("--no-chat"))   use_chat    = false;
        else if (arg("--verbose"))   verbose     = true;
        else if (arg("--flash-attn")) flash_attention = true;
        else if (arg("--paged-kv")) enable_paged_kv = true;
        else if (arg("--paged-kv-block")) paged_kv_block_size = static_cast<uint32_t>(std::max(1, atoi(next("--paged-kv-block"))));
        else if (arg("--gpu-layer"))   gpu_layer     = atoi(next("--gpu-layer"));
        else if (arg("--dev-mode")) {
            const char* mode = next("--dev-mode");
            if (strcmp(mode, "cpu") == 0) {
                dev_mode = DevMode::CPU_MODE;
            } else if (strcmp(mode, "gpu") == 0) {
                dev_mode = DevMode::GPU_MODE;
            } else if (strcmp(mode, "auto") == 0) {
                dev_mode = DevMode::AUTO_MODE;
            } else {
                dev_mode = DevMode::AUTO_MODE;
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
    if (n_batch <= 0) {
        n_batch = ctx_size;
    }
    if (n_ubatch <= 0) {
        n_ubatch = n_batch;
    }

    fprintf(stderr, "[main] Loading model: %s\n", model_path.c_str());

    ChatEngine chat;
    if (!chat.init(model_path, dev_mode, n_threads, ctx_size, top_p, top_k, temperature, gpu_layer, flash_attention, n_batch, n_ubatch,
            enable_paged_kv, paged_kv_block_size)) {
        fprintf(stderr, "Engine initialization failed\n");
        return 1;
    }

    HttpServer server;
    if (!server.init("0.0.0.0", 6666, &chat)) {
        fprintf(stderr, "HTTP server initialization failed: %s\n", server.last_error().c_str());
        return 1;
    }

    fprintf(stderr, "Starting HTTP server on 0.0.0.0:6666\n");
    server.run();

    fprintf(stderr, "llama server ready\n");
    return 0;
}
