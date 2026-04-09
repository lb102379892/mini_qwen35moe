// main.cpp — Qwen3.5 MoE CPU text generation
//
// Usage:
//   ./test_qwen35moe --model <path.gguf> [options]
//
// Options:
//   --model    <path>   Path to the GGUF model file (required)
//   --prompt   <text>   Input prompt (default: "Hello, how are you?")
//   --n-predict <n>     Maximum tokens to generate (default: 128)
//   --temp     <f>      Sampling temperature; 0 = greedy (default: 0.7)
//   --top-k    <n>      Top-k sampling (default: 40)
//   --top-p    <f>      Top-p nucleus sampling (default: 0.95)
//   --threads  <n>      CPU thread count (default: 4)
//   --max-ctx  <n>      KV-cache size in tokens (default: 2048)
//
#include "pipeline/generator.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s --model <path.gguf> [options]\n"
        "\n"
        "Options:\n"
        "  --model    <path>   GGUF model file (required)\n"
        "  --prompt   <text>   Input prompt (default: Hello!)\n"
        "  --n-predict <n>     Max new tokens (default: 128)\n"
        "  --temp     <f>      Sampling temperature (0=greedy, default: 0.7)\n"
        "  --top-k    <n>      Top-k filter (default: 40)\n"
        "  --top-p    <f>      Top-p nucleus filter (default: 0.95)\n"
        "  --threads  <n>      CPU threads (default: 4)\n"
        "  --max-ctx  <n>      KV-cache slots (default: 2048)\n"
        "  --seed     <n>      RNG seed for sampling (default: 42)\n",
        prog);
}

int main(int argc, char* argv[]) {
    // Defaults
    std::string       model_path;
    std::string       prompt     = "Hello!";
    Generator::Params p;
    p.n_predict   = 128;
    p.temperature = 0.7f;
    p.top_k       = 40;
    p.top_p       = 0.95f;
    p.n_threads   = 4;
    p.max_ctx     = 2048;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "--model" || arg == "-m") && i + 1 < argc) {
            model_path = argv[++i];
        } else if ((arg == "--prompt" || arg == "-p") && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--n-predict" && i + 1 < argc) {
            p.n_predict = std::atoi(argv[++i]);
        } else if (arg == "--temp" && i + 1 < argc) {
            p.temperature = (float)std::atof(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            p.top_k = std::atoi(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            p.top_p = (float)std::atof(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            p.n_threads = std::atoi(argv[++i]);
        } else if (arg == "--max-ctx" && i + 1 < argc) {
            p.max_ctx = std::atoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            p.seed = (unsigned int)std::atoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty()) {
        fprintf(stderr, "ERROR: --model is required\n\n");
        print_usage(argv[0]);
        return 1;
    }

    // Initialise the generation pipeline
    Generator gen;
    if (!gen.init(model_path, p)) {
        fprintf(stderr, "ERROR: failed to initialise generator\n");
        return 1;
    }

    // Run generation
    gen.generate(prompt);

    return 0;
}
