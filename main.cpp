// main.cpp — Qwen3.5 MoE CPU text generation
//
// Usage:
//   ./test_qwen35moe --model <path.gguf> [options]
//
// Options:
//   --model    <path>   Path to the GGUF model file (required)
//   --prompt   <text>   Input prompt (default: "Hello, how are you?")
#include <cstdio>
#include <cstdlib>
#include <string>

static void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s --model <path.gguf> [options]\n"
        "\n"
        "Options:\n"
        "  --model    <path>   GGUF model file (required)\n"
        "  --prompt   <text>   Input prompt (default: Hello!)\n",
        prog);
}

int main(int argc, char* argv[]) {
    // Defaults
    std::string       model_path;
    std::string       prompt     = "Hello!";
    
    return 0;
}
