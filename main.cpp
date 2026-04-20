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
#include "core/gguf_reader.hpp"
#include "model/model.hpp"
#include "model/loader.hpp"
#include "pipeline/recognizer.hpp"
#include "pipeline/tokenizer.hpp"
#include "pipeline/inference.hpp"

static int sample(const std::vector<float>& logits, float temperature, float top_p, int top_k, std::mt19937& rng,
    const std::vector<int32_t>& recent_tokens, BPETokenizer& tokenizer) {
    
    const int n = (int)logits.size();
    std::vector<float> modified_logits = logits;

    // ==========================================================
    // 1. 三维复合惩罚引擎 (专治各类复读机)
    // ==========================================================
    const float rep_penalty   = 1.15f;  // 乘法惩罚：基础压制
    const float freq_penalty  = 0.05f;  // 频率惩罚：出现越多，扣分越狠
    const float pres_penalty  = 0.05f;  // 存在惩罚：只要出现过，就扣基础分
    const int penalty_window  = 256;    // 恢复 256 的大窗口，确保它能记住很久之前说过的词
    
    if (!recent_tokens.empty()) {
        std::unordered_map<int32_t, int> counts;
        int start_idx = std::max(0, (int)recent_tokens.size() - penalty_window);
        for (int i = start_idx; i < (int)recent_tokens.size(); i++) {
            counts[recent_tokens[i]]++;
        }
        
        for (const auto& kv : counts) {
            int32_t tok = kv.first;
            int count = kv.second;
            
            if (tok < 0 || tok >= n) continue;
            
            // 【核心护城河】：绝对豁免 结束符、换行符(198)、空格(220)
            if (tokenizer.is_stop_token(tok) || tok == 198 || tok == 220) {
                continue;
            }
            
            // A. 乘法压制 (Repetition Penalty)
            if (rep_penalty > 1.0f) {
                if (modified_logits[tok] > 0) modified_logits[tok] /= rep_penalty;
                else modified_logits[tok] *= rep_penalty;
            }
            
            // B. 减法强扣 (Frequency & Presence Penalty)
            modified_logits[tok] -= (count * freq_penalty + pres_penalty);
        }
    }

    // ==========================================================
    // 2. 贪婪解码分支 (Temp <= 0)
    // ==========================================================
    if (temperature <= 0.0f) {
        return (int)(std::max_element(modified_logits.begin(), modified_logits.end()) - modified_logits.begin());
    }

    // ==========================================================
    // 3. 应用 Temperature
    // ==========================================================
    for (auto& v : modified_logits) v /= temperature;

    // 构建 TokenData 数组用于高级采样
    struct TokenData { int id; float logit; float p; };
    std::vector<TokenData> cur_p;
    cur_p.reserve(n);
    float max_logit = -INFINITY;
    for (int i = 0; i < n; i++) {
        cur_p.push_back({i, modified_logits[i], 0.0f});
        if (modified_logits[i] > max_logit) {
            max_logit = modified_logits[i];
        }
    }

    // ==========================================================
    // 4. Min-P 采样 (动态剔除长尾垃圾词汇)
    // ==========================================================
    const float min_p = 0.05f;
    const float min_logit = max_logit + std::log(min_p);
    
    std::vector<TokenData> filtered_p;
    for (const auto& tok : cur_p) {
        if (tok.logit >= min_logit) {
            filtered_p.push_back(tok);
        }
    }
    if (filtered_p.empty()) filtered_p.push_back(cur_p[0]); // 兜底

    // ==========================================================
    // 5. Top-K 采样
    // ==========================================================
    if (top_k > 0 && top_k < (int)filtered_p.size()) {
        std::partial_sort(filtered_p.begin(), filtered_p.begin() + top_k, filtered_p.end(),
                          [](const TokenData& a, const TokenData& b) { return a.logit > b.logit; });
        filtered_p.resize(top_k);
        max_logit = filtered_p[0].logit; 
    }

    // ==========================================================
    // 6. Top-P 采样
    // ==========================================================
    if (top_p > 0.0f && top_p < 1.0f) {
        std::sort(filtered_p.begin(), filtered_p.end(), [](const TokenData& a, const TokenData& b) { return a.logit > b.logit; });
        double sum_cum = 0.0;
        for (auto& tok : filtered_p) {
            tok.p = std::exp(tok.logit - max_logit);
            sum_cum += tok.p;
        }
        double cum = 0.0;
        int p_cutoff = filtered_p.size();
        for (int i = 0; i < (int)filtered_p.size(); i++) {
            cum += filtered_p[i].p / sum_cum;
            if (cum >= top_p) { 
                p_cutoff = i + 1; 
                break; 
            }
        }
        filtered_p.resize(std::max(1, p_cutoff));
    }

    // ==========================================================
    // 7. 终极 Softmax 与单趟轮盘赌采样
    // ==========================================================
    double final_sum = 0.0;
    for (auto& tok : filtered_p) {
        tok.p = std::exp(tok.logit - max_logit);
        final_sum += tok.p;
    }

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double rnd = dist(rng);
    double sum_tgt = final_sum * rnd;
    double sum_run = 0.0;
    
    for (const auto& tok : filtered_p) {
        sum_run += tok.p;
        if (sum_run >= sum_tgt) {
            return tok.id;
        }
    }

    return filtered_p.back().id;
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

        int next_token = sample(logits, temperature, top_p, top_k, rng, recent_tokens, tokenizer);
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

    InferenceEngine engine(*model, recognizer.reader(), n_threads, ctx_size, gpu_mode);
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