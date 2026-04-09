// main.cpp — FunASR-GGML minimal example
//
// Build:
//   cmake .. [-DQWEN35MOE_CUDA=ON] && make funasr_main
// Run:
//   ./funasr_main FunAsr_q8.bin audio.wav [--gpu]
//
#include "pipeline/recognizer.hpp"
#include <cstdio>
#include <string>

int main(int argc, char* argv[]) {
    const std::string model_path = "/home/xc/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_M.gguf";

    // Initialize
    Recognizer recognizer;
    if (!recognizer.init(model_path)) 
        return 1;

    return 0;
}