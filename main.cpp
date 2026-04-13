// main.cpp — Qwen3.5 MoE CPU text generation
//
// Usage:
//   ./test_qwen35moe --model <path.gguf> [options]
//
// Options:
//   --model      <path>   GGUF model file (required)
//   --prompt     <text>   Input prompt (default: interactive REPL)
//   --system     <text>   System message for chat template
//   --n-predict  <N>      Max tokens to generate (default: 256)
//   --temp       <f>      Temperature (default: 0.8; 0 = greedy)
//   --top-p      <f>      Top-p sampling (default: 0.95)
//   --top-k      <N>      Top-k sampling (default: 40; 0 = disabled)
//   --threads    <N>      CPU threads (default: 4)
//   --no-chat            Disable chat template (pass prompt verbatim)
//   --verbose            Print tokenization and timing info
//
// Minimal run example:
//   ./test_qwen35moe \
//     --model Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_M.gguf \
//     --prompt "Hello, who are you?" --n-predict 128 --temp 0.7
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

#include "core/gguf_reader.hpp"
#include "model/model.hpp"
#include "model/loader.hpp"
#include "pipeline/recognizer.hpp"
#include "pipeline/tokenizer.hpp"
#include "pipeline/inference.hpp"
#include <iostream>

// ============================================================
// Sampling helpers
// ============================================================

// Greedy: argmax
static int sample_greedy(const std::vector<float>& logits) {
    return (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

// Temperature + top-k + top-p sampling
static int sample(const std::vector<float>& logits, float temperature,
                  float top_p, int top_k, std::mt19937& rng) {
    if (temperature <= 0.0f) return sample_greedy(logits);

    const int n = (int)logits.size();

    // Apply temperature
    std::vector<float> probs(logits.begin(), logits.end());
    for (auto& v : probs) v /= temperature;

    // Softmax
    float max_v = *std::max_element(probs.begin(), probs.end());
    float sum = 0.0f;
    for (auto& v : probs) { v = std::exp(v - max_v); sum += v; }
    for (auto& v : probs) v /= sum;

    // Build sorted indices
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return probs[a] > probs[b]; });

    // Top-k filter
    int k = (top_k > 0 && top_k < n) ? top_k : n;

    // Top-p filter
    float cum = 0.0f;
    int p_cutoff = k;
    for (int i = 0; i < k; i++) {
        cum += probs[idx[i]];
        if (cum >= top_p) { p_cutoff = i + 1; break; }
    }
    k = std::min(k, p_cutoff);

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
        "  --model      <path>   GGUF model file (required)\n"
        "  --prompt     <text>   Input prompt (omit for REPL mode)\n"
        "  --system     <text>   System prompt for chat template\n"
        "  --n-predict  <N>      Max new tokens to generate (default: 256)\n"
        "  --temp       <f>      Sampling temperature (default: 0.8; 0=greedy)\n"
        "  --top-p      <f>      Top-p sampling threshold (default: 0.95)\n"
        "  --top-k      <N>      Top-k sampling (default: 40; 0=off)\n"
        "  --threads    <N>      CPU threads (default: 4)\n"
        "  --seed       <N>      RNG seed (default: random)\n"
        "  --no-chat            Pass prompt verbatim (no chat template)\n"
        "  --verbose            Show tokenization and timing info\n"
        "\n"
        "Example:\n"
        "  %s --model model.gguf --prompt \"Hello!\" --n-predict 128 --temp 0.7\n",
        prog, prog);
}

// ============================================================
// Generation loop
// ============================================================

static void generate(InferenceEngine& engine, BPETokenizer& tokenizer,
                     const std::string& prompt_text,
                     int n_predict, float temperature, float top_p, int top_k,
                     bool use_chat, bool verbose, std::mt19937& rng) {

    // Tokenize
    std::string formatted = prompt_text;
    if (use_chat) {
        formatted = tokenizer.make_chat_prompt(prompt_text);
    }

    std::vector<int32_t> tokens;
    if (use_chat && tokenizer.im_start_id() >= 0) {
        tokens = tokenizer.encode_special(formatted);
    } else {
        tokens = tokenizer.encode(formatted);
    }

    if (tokens.empty()) {
        fprintf(stderr, "[main] WARNING: prompt encoded to 0 tokens\n");
        return;
    }

    if (verbose) {
        fprintf(stderr, "[main] Prompt: \"%s\"\n", formatted.c_str());
        fprintf(stderr, "[main] Prompt tokens (%zu): ", tokens.size());
        for (int t : tokens) fprintf(stderr, "%d ", t);
        fprintf(stderr, "\n");
    }

    // Generation
    auto t_start = std::chrono::high_resolution_clock::now();
    int n_generated = 0;

    printf("\n");

    // --- Prefill: reset state and process the full prompt ---
    engine.reset_state();
    auto t_prefill_start = std::chrono::high_resolution_clock::now();
    std::vector<float> logits = engine.forward(tokens);
    auto t_prefill_end = std::chrono::high_resolution_clock::now();

    if (logits.empty()) {
        fprintf(stderr, "\n[main] ERROR: forward() returned empty logits\n");
        printf("\n");
        auto t_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        fprintf(stderr, "\n[main] Generated %d tokens in %.1f ms (%.1f ms/tok)\n",
                n_generated, total_ms,
                n_generated > 0 ? total_ms / n_generated : 0.0);
        return;
    }

    if (verbose) {
        double ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();
        fprintf(stderr, "[main] Prefill: %.1f ms (seq_len=%zu)\n", ms, tokens.size());
    }

    int next_token = sample(logits, temperature, top_p, top_k, rng);

    if (tokenizer.is_stop_token(next_token)) {
        if (verbose) fprintf(stderr, "\n[main] Stop token %d reached\n", next_token);
    } else {
        std::string piece = tokenizer.decode_one(next_token);
        printf("%s", piece.c_str());
        fflush(stdout);
        n_generated++;

        // --- Decode loop: pass exactly one new token per step ---
        while (n_generated < n_predict) {
            logits = engine.forward({next_token});

            if (logits.empty()) {
                fprintf(stderr, "\n[main] ERROR: forward() returned empty logits\n");
                break;
            }

            next_token = sample(logits, temperature, top_p, top_k, rng);

            if (tokenizer.is_stop_token(next_token)) {
                if (verbose) fprintf(stderr, "\n[main] Stop token %d reached\n", next_token);
                break;
            }

            piece = tokenizer.decode_one(next_token);
            printf("%s", piece.c_str());
            fflush(stdout);
            n_generated++;
        }
    }

    printf("\n");

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    if (verbose || n_generated > 0) {
        fprintf(stderr, "\n[main] Generated %d tokens in %.1f ms (%.1f ms/tok)\n",
                n_generated, total_ms,
                n_generated > 0 ? total_ms / n_generated : 0.0);
    }
}

// ============================================================
// main()
// ============================================================

int main(int argc, char* argv[]) {
    // ---- Defaults ----
    std::string model_path;
    std::string prompt;
    std::string system_msg   = "You are a helpful assistant.";
    int         n_predict    = 256;
    float       temperature  = 0.8f;
    float       top_p        = 0.95f;
    int         top_k        = 40;
    int         n_threads    = 4;
    bool        use_chat     = true;
    bool        verbose      = false;
    bool        repl_mode    = false;
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
        else if (arg("--system"))    system_msg  = next("--system");
        else if (arg("--n-predict")) n_predict   = atoi(next("--n-predict"));
        else if (arg("--temp"))      temperature = (float)atof(next("--temp"));
        else if (arg("--top-p"))     top_p       = (float)atof(next("--top-p"));
        else if (arg("--top-k"))     top_k       = atoi(next("--top-k"));
        else if (arg("--threads"))   n_threads   = atoi(next("--threads"));
        else if (arg("--seed"))      rng_seed    = (uint64_t)atoll(next("--seed"));
        else if (arg("--no-chat"))   use_chat    = false;
        else if (arg("--verbose"))   verbose     = true;
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

    // ---- Load model ----
    fprintf(stderr, "[main] Loading model: %s\n", model_path.c_str());
    Recognizer recognizer;
    if (!recognizer.init(model_path, verbose)) {
        fprintf(stderr, "[main] ERROR: failed to load model: %s\n",
                recognizer.last_error().c_str());
        return 1;
    }

    const Qwen35moeModel* model = recognizer.model();
    fprintf(stderr, "[main] Model loaded. vocab_size=%ld n_layers=%u\n",
            model->weights.token_embd->ne[1],
            model->config.qwen35moe.block_count);

    // ---- Load tokenizer ----
    BPETokenizer tokenizer;
    if (!tokenizer.load(model->config.tokenizer)) {
        fprintf(stderr, "[main] ERROR: failed to load tokenizer\n");
        return 1;
    }

    // ---- Create inference engine ----
    InferenceEngine engine(*model, n_threads);

    // ---- Sampling RNG ----
    std::mt19937 rng(rng_seed);
    if (verbose) fprintf(stderr, "[main] RNG seed: %llu\n", (unsigned long long)rng_seed);

    // ---- Single prompt or REPL ----
    if (!repl_mode) {
        generate(engine, tokenizer, prompt, n_predict, temperature, top_p, top_k,
                 use_chat, verbose, rng);
    } else {
        // Interactive REPL
        fprintf(stderr, "[main] Interactive mode. Type your prompt (empty line to quit).\n");
        fprintf(stderr, "       Ctrl+D or empty input to exit.\n\n");

        std::string line;
        while (true) {
            printf("You: ");
            fflush(stdout);
            if (!std::getline(std::cin, line) || line.empty()) {
                printf("\n");
                break;
            }
            printf("Assistant:");
            fflush(stdout);
            generate(engine, tokenizer, line, n_predict, temperature, top_p, top_k,
                     use_chat, verbose, rng);
        }
    }

    return 0;
}
