#include "pipeline/chat.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cstdio>
#include "pipeline/context_limit.h"

ChatEngine::ChatEngine() {
    std::random_device rd;
    gpu_sampling_rng_ = std::mt19937(rd());
}

ChatEngine::~ChatEngine() {
}

bool ChatEngine::init(const CParam& param) {
    dev_mode_ = param.dev_mode;
    top_p_ = param.top_p;
    top_k_ = param.top_k;
    temperature_ = param.temperature;
    n_batch_ = param.n_batch > 0 ? param.n_batch : param.ctx_size;
    n_ubatch_ = param.n_ubatch > 0 ? param.n_ubatch : n_batch_;
    use_gpu_topk_sampling_ = (dev_mode_ != DevMode::CPU_MODE &&
        ((temperature_ > 0.0f && top_k_ > 0) || temperature_ <= 0.0f));

    ggml_backend_load_all();

    model_ = std::make_shared<Qwen35moeModel>();
    if (!model_->init(param.model_path, param.dev_mode, param.n_threads, param.gpu_layer, param.no_mmap)) {
        fprintf(stderr, "ERROR: Failed to init model from %s\n", param.model_path.c_str());
        return false;
    }

    tokenizer_ = std::make_shared<Tokenizer>(model_->meta_->tokenizer);
    if (tokenizer_->init() != 0) {
        fprintf(stderr, "ERROR: Failed to initialize tokenizer from model\n");
        return false;
    }

    if (param.temperature > 0.0f) {
        sampler_ = std::make_shared<TemperatureSampler>(param.temperature, 1.1f, 64, param.top_k, param.top_p);
    } else {
        sampler_ = std::make_shared<GreedySampler>();
    }
    sampler_->set_eos_token_id(model_->meta_->tokenizer.ggml_eos_token_id);
    sched_ = model_->get_scheduler();

    forward_pass_ = std::make_shared<Qwen35moeForwardPass>();
    forward_pass_->init(param.ctx_size, 1, model_, static_cast<uint32_t>(n_batch_), static_cast<uint32_t>(n_ubatch_),
        param.enable_paged_kv, param.paged_kv_block_size);
    if (use_gpu_topk_sampling_) {
        const int device_top_k = temperature_ > 0.0f ? top_k_ : 1;
        forward_pass_->configure_device_sampling(device_top_k, temperature_);
    }
    if (param.flash_attention) {
        forward_pass_->set_flash_attention_enabled(true);
    }

    return true;
}

int ChatEngine::sample_from_topk_candidates(const TopKSampleCandidates& candidates, float top_p) {
    if (!candidates.token_ids.empty() && (temperature_ <= 0.0f || candidates.token_ids.size() == 1)) {
        return candidates.token_ids[0];
    }
    if (candidates.token_ids.empty() || candidates.token_ids.size() != candidates.logits.size()) {
        throw std::runtime_error("Invalid top-k candidates");
    }

    std::vector<std::pair<int32_t, float>> sorted;
    sorted.reserve(candidates.token_ids.size());
    for (size_t i = 0; i < candidates.token_ids.size(); ++i) {
        sorted.emplace_back(candidates.token_ids[i], candidates.logits[i]);
    }

    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
    });

    if (top_p > 0.0f && top_p < 1.0f && sorted.size() > 1) {
        const float max_l = sorted[0].second;
        float sum_exp = 0.0f;
        for (const auto& p : sorted) {
            sum_exp += std::exp(p.second - max_l);
        }
        float cum = 0.0f;
        size_t cutoff = sorted.size();
        for (size_t i = 0; i < sorted.size(); ++i) {
            cum += std::exp(sorted[i].second - max_l) / sum_exp;
            if (cum >= top_p) {
                cutoff = i + 1;
                break;
            }
        }
        sorted.resize(cutoff);
    }

    const float max_l = sorted[0].second;
    float sum_exp = 0.0f;
    std::vector<float> probs;
    probs.reserve(sorted.size());
    for (const auto& p : sorted) {
        const float e = std::exp(p.second - max_l);
        probs.push_back(e);
        sum_exp += e;
    }
    for (float& p : probs) {
        p /= sum_exp;
    }

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    return sorted[dist(gpu_sampling_rng_)].first;
}

bool ChatEngine::run_complete(const std::string& prompt, const int max_tokens, std::string& response, std::string* finish_reason) {
    printf("run_complete: prompt='%s', max_tokens=%d\n", prompt.c_str(), max_tokens);
    auto m = model_->meta_;
    std::string finish_reason_local = "stop";
    if (finish_reason) {
        *finish_reason = finish_reason_local;
    }
    forward_pass_->reset_sequence(0);

    std::vector<int32_t> tokens = tokenizer_->encode(prompt);

    using Clock = std::chrono::steady_clock;
    const size_t n_prompt_tokens = tokens.size();
    auto t_prefill_start = Clock::now();
    std::vector<float> logits;
    TopKSampleCandidates prefill_candidates;
    if (use_gpu_topk_sampling_) {
        prefill_candidates = forward_pass_->run_prefill_topk(tokens, 0, 0, sched_);
    } else {
        logits = forward_pass_->run_prefill(tokens, 0, 0, sched_);
    }
    auto t_prefill_end = Clock::now();

    size_t vocab_size = m->tokenizer.ggml_tokens.size();
    std::vector<float> last_token_logits;
    int next_token_id = 0;
    if (use_gpu_topk_sampling_) {
        next_token_id = sample_from_topk_candidates(prefill_candidates, top_p_);
    } else {
        last_token_logits.assign(logits.end() - vocab_size, logits.end());
        next_token_id = sampler_->sample(last_token_logits, tokens);
    }

    // Decode phase
    const int32_t eos_token_id = m->tokenizer.ggml_eos_token_id;
    const int32_t im_end_token_id = tokenizer_->get_special_token_id("<|im_end|>");
    const int32_t endoftext_token_id = tokenizer_->get_special_token_id("<|endoftext|>");
    const bool has_im_end_token_id = im_end_token_id >= 0;
    const bool has_endoftext_token_id = endoftext_token_id >= 0;
    const bool needs_string_stop_check = !has_im_end_token_id || !has_endoftext_token_id;
    const std::string im_end_str = "<|im_end|>";
    const std::string eos_str = "<|endoftext|>";

    // PLD state for single-prompt mode
    std::vector<int32_t> prompt_tokens_for_pld = tokens;  // Original prompt
    std::vector<int32_t> generated_tokens;
    const uint32_t requested_tokens = max_tokens > 0 ? static_cast<uint32_t>(max_tokens) : 0;
    const int initial_cache_pos = static_cast<int>(forward_pass_->get_cache_pos(0));
    const uint32_t context_budget = qwen35moe_clamp_generation_tokens(
        initial_cache_pos, forward_pass_->get_context_len(), requested_tokens);
    const bool context_budget_limited = context_budget < requested_tokens;
    tokens.reserve(tokens.size() + static_cast<size_t>(context_budget));
    generated_tokens.reserve(static_cast<size_t>(context_budget));

    double decode_tok_decode_ms = 0.0;
    double decode_forward_ms = 0.0;
    double decode_sample_ms = 0.0;
    double decode_logits_copy_ms = 0.0;
    size_t perf_steps = 0;

    auto t_decode_start = Clock::now();
    for (uint32_t i = 0; i < context_budget; ++i) {
        const int current_pos = static_cast<int>(forward_pass_->get_cache_pos(0));
        if (!forward_pass_->can_decode_at_position(current_pos)) {
            finish_reason_local = "length";
            break;
        }

        if (next_token_id == eos_token_id ||
            (has_im_end_token_id && next_token_id == im_end_token_id) ||
            (has_endoftext_token_id && next_token_id == endoftext_token_id)) {
            break;
        }

        auto t0 = Clock::now();
        std::string decoded_token = tokenizer_->decode(next_token_id);
        auto t1 = Clock::now();

        decode_tok_decode_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (needs_string_stop_check && (decoded_token == im_end_str || decoded_token == eos_str)) {
            break;
        }
        response += decoded_token;

        tokens.push_back(next_token_id);
        generated_tokens.push_back(next_token_id);

        // --- Normal decode path ---
        if (use_gpu_topk_sampling_) {
            auto t2 = Clock::now();
            TopKSampleCandidates decode_candidates =
                forward_pass_->run_decode_cached_topk(next_token_id, current_pos, 0, sched_);
            auto t3 = Clock::now();
            if (decode_candidates.token_ids.empty()) {
                finish_reason_local = "length";
                break;
            }

            next_token_id = sample_from_topk_candidates(decode_candidates, top_p_);
            auto t4 = Clock::now();

            decode_forward_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();
            decode_sample_ms += std::chrono::duration<double, std::milli>(t4 - t3).count();

            if (i < 8 || ((i + 1) % 64 == 0)) {
                std::printf("[PERF][decode step=%d pos=%d][gpu-topk] forward=%.3f ms sample_cpu=%.3f ms k=%zu\n",
                    static_cast<int>(i), current_pos,
                    std::chrono::duration<double, std::milli>(t3 - t2).count(),
                    std::chrono::duration<double, std::milli>(t4 - t3).count(),
                    decode_candidates.token_ids.size());
            }
        } else {
            auto t2 = Clock::now();
            std::vector<float> token_logits =
                forward_pass_->run_decode_cached(next_token_id, current_pos, 0, sched_);
            auto t3 = Clock::now();
            if (token_logits.empty()) {
                finish_reason_local = "length";
                break;
            }

            last_token_logits.assign(token_logits.begin(), token_logits.begin() + vocab_size);
            auto t4 = Clock::now();

            next_token_id = sampler_->sample(last_token_logits, tokens);
            auto t5 = Clock::now();

            decode_forward_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();
            decode_logits_copy_ms += std::chrono::duration<double, std::milli>(t4 - t3).count();
            decode_sample_ms += std::chrono::duration<double, std::milli>(t5 - t4).count();

            if (i < 8 || ((i + 1) % 64 == 0)) {
                std::printf("[PERF][decode step=%d pos=%d][cpu-sample] forward=%.3f ms logits_copy=%.3f ms sample=%.3f ms vocab=%zu\n",
                    static_cast<int>(i), current_pos,
                    std::chrono::duration<double, std::milli>(t3 - t2).count(),
                    std::chrono::duration<double, std::milli>(t4 - t3).count(),
                    std::chrono::duration<double, std::milli>(t5 - t4).count(),
                    vocab_size);
            }
        }

        perf_steps++;
    }
    if (finish_reason_local == "stop" && context_budget_limited && generated_tokens.size() >= context_budget) {
        finish_reason_local = "length";
    }
    auto t_decode_end = Clock::now();
    printf("response: [%s].\n", response.c_str());

    auto prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_prefill_end - t_prefill_start).count();
    auto decode_ms  = std::chrono::duration_cast<std::chrono::milliseconds>(t_decode_end - t_decode_start).count();
    const size_t n_decoded = generated_tokens.size();

    double prefill_tps = (prefill_ms > 0) ? (n_prompt_tokens * 1000.0 / prefill_ms) : 0.0;
    double decode_tps  = (decode_ms  > 0 && n_decoded > 0) ? (n_decoded * 1000.0 / decode_ms) : 0.0;

    printf("[Timing] prefill=%lld ms (%ld tokens, %d t/s) decode=%lldms (%ld tokens, %d t/s)\n", 
        prefill_ms, n_prompt_tokens, static_cast<int>(prefill_tps), decode_ms, n_decoded, static_cast<int>(decode_tps));

    if (perf_steps > 0) {
        std::printf(
            "[PERF][decode-summary] steps=%zu tok_decode=%.3f ms/step forward=%.3f ms/step logits_copy=%.3f ms/step sample=%.3f ms/step\n",
            perf_steps,
            decode_tok_decode_ms / perf_steps,
            decode_forward_ms / perf_steps,
            decode_logits_copy_ms / perf_steps,
            decode_sample_ms / perf_steps
        );
    }

    if (finish_reason) {
        *finish_reason = finish_reason_local;
    }

    return true;
}
