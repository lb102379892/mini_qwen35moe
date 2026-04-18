// main.cpp — Qwen3.5 MoE CPU text generation
//
// Usage:
//   ./test_qwen35moe --model <path.gguf> [options]
//
// Options:
//   --model          <path>   GGUF model file (required)
//   --prompt         <text>   Input prompt (default: interactive REPL)
//   --system         <text>   System message for chat template
//   --n-predict      <N>      Max tokens to generate (default: 256)
//   --temp           <f>      Temperature (default: 0.8; 0 = greedy)
//   --top-p          <f>      Top-p sampling (default: 0.95)
//   --top-k          <N>      Top-k sampling (default: 40; 0 = disabled)
//   --repeat-penalty <f>      Repetition penalty (default: 1.1; 1.0 = disabled)
//   --repeat-last-n  <N>      Look-back window for repetition penalty (default: 64)
//   --threads        <N>      CPU threads (default: 4)
//   --ctx-size       <N>      KV cache context length (default: auto for single prompt, 2048 for REPL)
//   --no-chat                Disable chat template (pass prompt verbatim)
//   --thinking               Enable Qwen thinking mode in the chat template
//   --no-thinking            Disable Qwen thinking mode (default)
//   --verbose                Print tokenization and timing info
//   --gpu-mode      <mode>   GPU mode: off|hybrid|full (default: off)
//   --gpu                    Alias for --gpu-mode hybrid
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
#include <cctype>

#include "core/gguf_reader.hpp"
#include "model/model.hpp"
#include "model/loader.hpp"
#include "pipeline/recognizer.hpp"
#include "pipeline/tokenizer.hpp"
#include "pipeline/inference.hpp"
#include <iostream>

class OutputRenderer {
public:
    std::string push_token(const BPETokenizer& tokenizer, int token_id) {
        if (tokenizer.should_skip_output_token(token_id)) {
            return flush_visible_text();
        }

        pending_ += tokenizer.decode_one(token_id);
        return flush_visible_text();
    }

    std::string finish() {
        return flush_visible_text(/*final_flush=*/true);
    }

private:
    static size_t utf8_safe_prefix_len(const std::string& text) {
        if (text.empty()) return 0;

        size_t end = text.size();
        size_t i = end;
        int continuation_bytes = 0;

        while (i > 0 && continuation_bytes < 3) {
            unsigned char c = (unsigned char)text[i - 1];
            if ((c & 0xC0) == 0x80) {
                ++continuation_bytes;
                --i;
                continue;
            }

            int expected_len = 1;
            if ((c & 0x80) == 0x00) {
                expected_len = 1;
            } else if ((c & 0xE0) == 0xC0) {
                expected_len = 2;
            } else if ((c & 0xF0) == 0xE0) {
                expected_len = 3;
            } else if ((c & 0xF8) == 0xF0) {
                expected_len = 4;
            } else {
                return i - 1;
            }

            const size_t actual_len = end - (i - 1);
            if (actual_len < (size_t)expected_len) {
                return i - 1;
            }
            return end;
        }

        if (continuation_bytes == 0) return end;
        return end - (size_t)continuation_bytes;
    }

    std::string flush_visible_text(bool final_flush = false) {
        static const std::string kThinkOpen = "<think>";
        static const std::string kThinkClose = "</think>";

        std::string visible;
        while (true) {
            if (in_think_) {
                size_t close_pos = pending_.find(kThinkClose);
                if (close_pos == std::string::npos) {
                    if (!final_flush && pending_.size() > kThinkClose.size()) {
                        pending_.erase(0, pending_.size() - (kThinkClose.size() - 1));
                    }
                    break;
                }
                pending_.erase(0, close_pos + kThinkClose.size());
                in_think_ = false;
                continue;
            }

            size_t open_pos = pending_.find(kThinkOpen);
            if (open_pos != std::string::npos) {
                visible += pending_.substr(0, open_pos);
                pending_.erase(0, open_pos + kThinkOpen.size());
                in_think_ = true;
                continue;
            }

            if (final_flush) {
                visible += pending_;
                pending_.clear();
            } else if (pending_.size() >= kThinkOpen.size()) {
                size_t safe_len = pending_.size() - (kThinkOpen.size() - 1);
                safe_len = utf8_safe_prefix_len(pending_.substr(0, safe_len));
                if (safe_len > 0) {
                    visible += pending_.substr(0, safe_len);
                    pending_.erase(0, safe_len);
                }
            }
            break;
        }

        return visible;
    }

    std::string pending_;
    bool in_think_ = false;
};

// ============================================================
// Sampling helpers
// ============================================================

// Greedy: argmax
static int sample_greedy(const std::vector<float>& logits) {
    return (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

static void dump_top_candidates(const std::vector<float>& logits,
                                const BPETokenizer& tokenizer,
                                int top_n) {
    if (top_n <= 0 || logits.empty()) return;

    std::vector<int> idx((int)logits.size());
    std::iota(idx.begin(), idx.end(), 0);
    const int limit = std::min((int)idx.size(), top_n);
    std::partial_sort(idx.begin(), idx.begin() + limit, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });

    fprintf(stderr, "[main] Top %d logits:\n", limit);
    for (int i = 0; i < limit; ++i) {
        const int tok = idx[i];
        std::string piece = tokenizer.decode_one(tok);
        for (char& ch : piece) {
            if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';
        }
        fprintf(stderr, "  #%d tok=%d logit=%.6f piece=\"%s\"\n",
                i + 1, tok, logits[tok], piece.c_str());
    }
}

static bool is_single_repeated_punct_piece(const std::string& piece) {
    if (piece.empty()) return false;
    if (piece == ":" || piece == "：" || piece == "." || piece == "。" ||
        piece == "," || piece == "，" || piece == ";" || piece == "；" ||
        piece == "-" || piece == "_" || piece == "=" || piece == "*" ||
        piece == "#" || piece == "/" || piece == "\\" || piece == "|" ||
        piece == "(" || piece == ")" || piece == "[" || piece == "]" ||
        piece == "{" || piece == "}") {
        return true;
    }
    return false;
}

static bool contains_ascii_nocase(const std::string& haystack, const std::string& needle) {
    if (needle.empty()) return true;
    std::string lower_haystack = haystack;
    std::string lower_needle = needle;
    std::transform(lower_haystack.begin(), lower_haystack.end(), lower_haystack.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    std::transform(lower_needle.begin(), lower_needle.end(), lower_needle.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return lower_haystack.find(lower_needle) != std::string::npos;
}

static bool looks_like_code_request(const std::string& prompt) {
    static const char* keywords[] = {
        "代码", "示例", "服务器", "服务端", "脚本", "函数", "类", "编译", "运行",
        "linux", "server", "http", "tcp", "socket", "python", "c++", "cpp",
        "java", "golang", "go ", "rust", "node", "nginx", "apache"
    };
    for (const char* kw : keywords) {
        if (contains_ascii_nocase(prompt, kw)) return true;
    }
    return false;
}

static std::string resolve_system_message(const std::string& explicit_system_msg,
                                          const std::string& prompt_text) {
    if (!explicit_system_msg.empty()) return explicit_system_msg;
    if (looks_like_code_request(prompt_text)) {
        //return "你是一个严谨的编程助手。若用户要求代码示例，请优先给出完整、可运行、尽量简洁的示例代码，再补充必要的编译、运行或修改说明。不要先泛泛介绍概念，也不要只给半截代码。";
    }
    return "";
}

// Temperature + top-k + top-p sampling with optional repetition penalty
static int sample(const std::vector<float>& logits, float temperature,
                  float top_p, int top_k, std::mt19937& rng,
                  float repeat_penalty = 1.0f,
                  float presence_penalty = 0.0f,
                  const std::vector<int32_t>& recent_tokens = {}) {
    const int n = (int)logits.size();

    // Apply repetition / presence penalties in logit space.
    std::vector<float> probs(logits.begin(), logits.end());
    if (!recent_tokens.empty() && (repeat_penalty != 1.0f || presence_penalty != 0.0f)) {
        std::vector<uint8_t> seen((size_t)n, 0);
        for (int tok : recent_tokens) {
            if (tok >= 0 && tok < n && !seen[(size_t)tok]) {
                seen[(size_t)tok] = 1;
                if (probs[tok] > 0.0f) {
                    probs[tok] /= repeat_penalty;
                } else {
                    probs[tok] *= repeat_penalty;
                }
                probs[tok] -= presence_penalty;
            }
        }
    }

    if (temperature <= 0.0f) {
        return (int)(std::max_element(probs.begin(), probs.end()) - probs.begin());
    }

    // Apply temperature
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
        "  --model          <path>   GGUF model file (required)\n"
        "  --prompt         <text>   Input prompt (omit for REPL mode)\n"
        "  --system         <text>   System prompt for chat template\n"
        "  --n-predict      <N>      Max new tokens to generate (default: 256)\n"
        "  --temp           <f>      Sampling temperature (default: GGUF metadata or 0.8; 0=greedy)\n"
        "  --top-p          <f>      Top-p sampling threshold (default: GGUF metadata or 0.95)\n"
        "  --top-k          <N>      Top-k sampling (default: GGUF metadata or 40; 0=off)\n"
        "  --repeat-penalty <f>      Repetition penalty (default: mode-aware; 1.0=off)\n"
        "  --presence-penalty <f>    Presence penalty in logit space (default: mode-aware)\n"
        "  --repeat-last-n  <N>      Look-back window for repetition penalty (default: 64)\n"
        "  --threads        <N>      CPU threads (default: 4)\n"
        "  --ctx-size       <N>      KV cache context length (default: auto for single prompt, 2048 for REPL)\n"
        "  --seed           <N>      RNG seed (default: random)\n"
        "  --dump-top       <N>      Dump top-N logits for the first generation step\n"
        "  --thinking               Enable Qwen thinking mode in the chat template\n"
        "  --no-thinking            Disable Qwen thinking mode (default)\n"
        "  --no-chat                Pass prompt verbatim (no chat template)\n"
        "  --verbose                Show tokenization and timing info\n"
        "  --gpu-mode      <mode>   GPU mode: off|hybrid|full (default: off)\n"
        "  --gpu                    Alias for --gpu-mode hybrid\n"
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
                     float repeat_penalty, float presence_penalty, int repeat_last_n,
                     const std::string& system_msg, bool enable_thinking,
                     bool use_chat, bool verbose, int dump_top_n, int max_seq_len, std::mt19937& rng) {

    // Reset engine state before each new prompt (required for REPL multi-turn)
    engine.reset_state();

    // Tokenize
    std::string formatted = prompt_text;
    if (use_chat) {
        formatted = tokenizer.make_chat_prompt(prompt_text, system_msg, enable_thinking);
    }
    if (verbose) fprintf(stderr, "[main] Formatted prompt:%s\n", formatted.c_str());

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

    const int max_new_tokens = std::max(0, max_seq_len - (int)tokens.size());
    if (max_new_tokens <= 0) {
        fprintf(stderr, "[main] ERROR: prompt length %zu exceeds/equal ctx-size %d\n",
                tokens.size(), max_seq_len);
        return;
    }

    if (verbose) {
        fprintf(stderr, "[main] Prompt: \"%s\"\n", formatted.c_str());
        fprintf(stderr, "[main] Prompt tokens (%zu): ", tokens.size());
        for (int t : tokens) fprintf(stderr, "%d ", t);
        fprintf(stderr, "\n");
        if (n_predict > max_new_tokens) {
            fprintf(stderr, "[main] Requested n_predict=%d but only %d tokens fit in ctx-size=%d after the prompt; generation will stop at the context limit\n",
                    n_predict, max_new_tokens, max_seq_len);
        }
    }

    // Generation
    auto t_start = std::chrono::high_resolution_clock::now();
    int n_generated = 0;
    bool first_token = true;
    int last_token = -1;
    int repeated_token_run = 0;

    // Sliding window of recently generated tokens for repetition penalty
    std::vector<int32_t> recent_tokens;
    OutputRenderer renderer;

    printf("\n");

    // First forward pass feeds the full prompt (prefill).
    // Subsequent passes feed exactly 1 new token (decode).
    std::vector<int32_t> next_input = tokens;

    while (n_generated < n_predict && n_generated < max_new_tokens) {
        auto t_fwd_start = std::chrono::high_resolution_clock::now();
        std::vector<float> logits = engine.forward(next_input);
        auto t_fwd_end = std::chrono::high_resolution_clock::now();

        if (logits.empty()) {
            fprintf(stderr, "\n[main] ERROR: forward() returned empty logits\n");
            break;
        }

        if (dump_top_n > 0 && n_generated == 0) {
            dump_top_candidates(logits, tokenizer, dump_top_n);
        }

        int next_token = sample(logits, temperature, top_p, top_k, rng,
                                repeat_penalty, presence_penalty, recent_tokens);

        if (verbose && first_token) {
            double ms = std::chrono::duration<double, std::milli>(t_fwd_end - t_fwd_start).count();
            fprintf(stderr, "[main] First token forward: %.1f ms (seq_len=%zu)\n",
                    ms, tokens.size());
            first_token = false;
        }

        if (tokenizer.is_stop_token(next_token)) {
            if (verbose) fprintf(stderr, "\n[main] Stop token %d reached\n", next_token);
            break;
        }

        const std::string decoded_piece = tokenizer.decode_one(next_token);
        if (next_token == last_token) {
            repeated_token_run++;
        } else {
            repeated_token_run = 1;
            last_token = next_token;
        }
        if (repeated_token_run >= 64 ||
            (repeated_token_run >= 24 && is_single_repeated_punct_piece(decoded_piece))) {
            if (verbose) {
                fprintf(stderr, "\n[main] Degenerate repetition detected on token %d piece=\"%s\" run=%d; stopping early\n",
                        next_token, decoded_piece.c_str(), repeated_token_run);
            }
            break;
        }

        std::string piece = renderer.push_token(tokenizer, next_token);
        if (!piece.empty()) {
            printf("%s", piece.c_str());
            fflush(stdout);
        }

        // After the first (full-prompt) pass, feed only 1 token at a time
        next_input = {next_token};

        // Maintain sliding window for repetition penalty
        recent_tokens.push_back(next_token);
        if (repeat_last_n > 0 && (int)recent_tokens.size() > repeat_last_n) {
            recent_tokens.erase(recent_tokens.begin());
        }

        n_generated++;
    }

    std::string tail = renderer.finish();
    if (!tail.empty()) {
        printf("%s", tail.c_str());
    }
    printf("\n");

    if (verbose && n_generated >= max_new_tokens && n_generated < n_predict) {
        fprintf(stderr, "[main] Context limit reached after %d generated tokens (ctx-size=%d)\n",
                n_generated, max_seq_len);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    if (verbose || n_generated > 0) {
        fprintf(stderr, "\n[main] Generated %d tokens in %.1f ms (%.1f ms/tok)\n",
                n_generated, total_ms,
                n_generated > 0 ? total_ms / n_generated : 0.0);
    }
}

// ============================================================
// main()\n// ============================================================

int main(int argc, char* argv[]) {
    // ---- Defaults ----
    std::string model_path;
    std::string prompt;
    std::string system_msg;
    int         n_predict    = 256;
    float       temperature  = -1.0f;
    float       top_p        = -1.0f;
    int         top_k        = -1;
    float       repeat_penalty = -1.0f;
    float       presence_penalty = -1.0f;
    int         repeat_last_n  = 64;
    int         n_threads    = 4;
    int         ctx_size     = 0;
    int         dump_top_n   = 0;
    bool        user_set_temp = false;
    bool        user_set_top_p = false;
    bool        user_set_top_k = false;
    bool        user_set_repeat_penalty = false;
    bool        user_set_presence_penalty = false;
    bool        use_chat     = true;
    bool        verbose      = false;
    bool        repl_mode    = false;
    bool        enable_thinking = false;
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
        else if (arg("--system"))    system_msg  = next("--system");
        else if (arg("--n-predict")) n_predict   = atoi(next("--n-predict"));
        else if (arg("--temp"))      { temperature = (float)atof(next("--temp")); user_set_temp = true; }
        else if (arg("--top-p"))     { top_p = (float)atof(next("--top-p")); user_set_top_p = true; }
        else if (arg("--top-k"))     { top_k = atoi(next("--top-k")); user_set_top_k = true; }
        else if (arg("--repeat-penalty")) { repeat_penalty = (float)atof(next("--repeat-penalty")); user_set_repeat_penalty = true; }
        else if (arg("--presence-penalty")) { presence_penalty = (float)atof(next("--presence-penalty")); user_set_presence_penalty = true; }
        else if (arg("--repeat-last-n"))  repeat_last_n  = atoi(next("--repeat-last-n"));
        else if (arg("--threads"))        n_threads   = atoi(next("--threads"));
        else if (arg("--ctx-size"))   ctx_size     = atoi(next("--ctx-size"));
        else if (arg("--seed"))      rng_seed    = (uint64_t)atoll(next("--seed"));
        else if (arg("--dump-top"))  dump_top_n  = atoi(next("--dump-top"));
        else if (arg("--thinking"))  enable_thinking = true;
        else if (arg("--no-thinking")) enable_thinking = false;
        else if (arg("--no-chat"))   use_chat    = false;
        else if (arg("--verbose"))   verbose     = true;
        else if (arg("--gpu"))       gpu_mode    = GpuMode::Hybrid;
        else if (arg("--gpu-mode")) {
            const char* mode = next("--gpu-mode");
            if (strcmp(mode, "off") == 0) {
                gpu_mode = GpuMode::Off;
            } else if (strcmp(mode, "hybrid") == 0) {
                gpu_mode = GpuMode::Hybrid;
            } else if (strcmp(mode, "full") == 0) {
                gpu_mode = GpuMode::Full;
            } else {
                fprintf(stderr, "ERROR: invalid --gpu-mode '%s' (expected off|hybrid|full)\n", mode);
                return 1;
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

    const std::string resolved_system_msg = repl_mode
        ? system_msg
        : resolve_system_message(system_msg, prompt);
    const bool auto_code_profile = !repl_mode && system_msg.empty() && looks_like_code_request(prompt);

    // ---- Load model ----
    fprintf(stderr, "[main] Loading model: %s\n", model_path.c_str());
    Recognizer recognizer;
    if (!recognizer.init(model_path, verbose, gpu_mode)) {
        fprintf(stderr, "[main] ERROR: failed to load model: %s\n",
                recognizer.last_error().c_str());
        return 1;
    }

    Qwen35moeModel* model = recognizer.model();
    fprintf(stderr, "[main] Model loaded. vocab_size=%lld n_layers=%u\n",
            model->weights.token_embd->ne[1],
            model->config.qwen35moe.block_count);

    // ---- Load tokenizer ----
    BPETokenizer tokenizer;
    if (!tokenizer.load(model->config.tokenizer)) {
        fprintf(stderr, "[main] ERROR: failed to load tokenizer\n");
        return 1;
    }

    const int model_ctx_size = model->config.qwen35moe.context_length > 0
        ? (int)model->config.qwen35moe.context_length
        : 2048;

    int effective_ctx_size = ctx_size;
    if (effective_ctx_size <= 0) {
        if (!repl_mode) {
            std::string formatted = use_chat
                ? tokenizer.make_chat_prompt(prompt, resolved_system_msg, enable_thinking)
                : prompt;
            std::vector<int32_t> prompt_tokens = (use_chat && tokenizer.im_start_id() >= 0)
                ? tokenizer.encode_special(formatted)
                : tokenizer.encode(formatted);
            const int requested_total = (int)prompt_tokens.size() + std::max(0, n_predict);
            effective_ctx_size = std::max(2048, requested_total);
        } else {
            effective_ctx_size = 2048;
        }
    }
    if (effective_ctx_size > model_ctx_size) {
        fprintf(stderr, "[main] WARNING: requested ctx-size %d exceeds model context_length %d; clamping\n",
                effective_ctx_size, model_ctx_size);
        effective_ctx_size = model_ctx_size;
    }
    if (effective_ctx_size <= 0) {
        fprintf(stderr, "[main] ERROR: invalid ctx-size %d\n", effective_ctx_size);
        return 1;
    }

    // ---- Create inference engine ----
    InferenceEngine engine(*model, gpu_mode != GpuMode::Off ? recognizer.reader() : nullptr,
                           n_threads, effective_ctx_size, gpu_mode);

    const bool non_thinking_mode = use_chat && !enable_thinking;
    if (temperature < 0.0f) {
        if (auto_code_profile && !user_set_temp) {
            temperature = 0.2f;
        } else if (non_thinking_mode && !user_set_temp) {
            temperature = 0.7f;
        } else {
            temperature = model->config.general.sampling_temp > 0.0f
                ? model->config.general.sampling_temp : 0.8f;
        }
    }
    if (top_p < 0.0f) {
        if (auto_code_profile && !user_set_top_p) {
            top_p = 0.9f;
        } else if (non_thinking_mode && !user_set_top_p) {
            top_p = 0.8f;
        } else {
            top_p = model->config.general.sampling_top_p > 0.0f
                ? model->config.general.sampling_top_p : 0.95f;
        }
    }
    if (top_k < 0) {
        if (auto_code_profile && !user_set_top_k) {
            top_k = 40;
        } else if (non_thinking_mode && !user_set_top_k) {
            top_k = 20;
        } else {
            top_k = model->config.general.sampling_top_k > 0
                ? model->config.general.sampling_top_k : 40;
        }
    }
    if (repeat_penalty < 0.0f) {
        repeat_penalty = 1.05f;
    }
    if (presence_penalty < 0.0f) {
        presence_penalty = 0.0f;
    }
    if (temperature <= 0.0f) {
        if (!user_set_repeat_penalty) {
            repeat_penalty = 1.0f;
        }
        if (!user_set_presence_penalty) {
            presence_penalty = 0.0f;
        }
    }

    // ---- Sampling RNG ----
    std::mt19937 rng(rng_seed);
    if (verbose) {
        fprintf(stderr, "[main] RNG seed: %llu\n", (unsigned long long)rng_seed);
        fprintf(stderr, "[main] Sampling params: temp=%.3f top_p=%.3f top_k=%d repeat_penalty=%.3f presence_penalty=%.3f repeat_last_n=%d\n",
                temperature, top_p, top_k, repeat_penalty, presence_penalty, repeat_last_n);
        if (temperature <= 0.0f && !user_set_presence_penalty) {
            fprintf(stderr, "[main] Greedy mode: default presence penalty disabled for apples-to-apples comparison\n");
        }
        fprintf(stderr, "[main] Chat template: %s\n",
                tokenizer.has_chat_template() ? "GGUF template + Qwen fallback" : "Qwen fallback only");
        fprintf(stderr, "[main] Thinking mode: %s\n", enable_thinking ? "enabled" : "disabled");
        fprintf(stderr, "[main] Effective ctx-size: %d (model max %d)\n",
                effective_ctx_size, model_ctx_size);
        if (auto_code_profile) {
            fprintf(stderr, "[main] Auto code profile: enabled\n");
        }
        if (!resolved_system_msg.empty()) {
            fprintf(stderr, "[main] Resolved system prompt: %s\n", resolved_system_msg.c_str());
        }
    }

    // ---- Single prompt or REPL ----
    if (!repl_mode) {
        generate(engine, tokenizer, prompt, n_predict, temperature, top_p, top_k,
                 repeat_penalty, presence_penalty, repeat_last_n, resolved_system_msg, enable_thinking,
                 use_chat, verbose, dump_top_n, effective_ctx_size, rng);
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
                     repeat_penalty, presence_penalty, repeat_last_n, system_msg, enable_thinking,
                     use_chat, verbose, dump_top_n, effective_ctx_size, rng);
        }
    }

    return 0;
}
