#include "pipeline/chat.h"

#include "model/dev_defaults.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <utility>

namespace {
double elapsed_ms(
    std::chrono::steady_clock::time_point begin,
    std::chrono::steady_clock::time_point end
) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

double tokens_per_second(size_t tokens, double ms) {
    return ms > 0.0 ? static_cast<double>(tokens) * 1000.0 / ms : 0.0;
}
} // namespace

ChatEngine::ChatEngine() {
    std::random_device rd;
    gpu_sampling_rng_ = std::mt19937(rd());
}

ChatEngine::~ChatEngine() {
    {
        std::lock_guard<std::mutex> lock(batch_mutex_);
        batch_stop_ = true;
    }
    batch_cv_.notify_all();
    if (batch_worker_.joinable()) {
        batch_worker_.join();
    }
}

bool ChatEngine::init(const CParam& param_in) {
    CParam param = param_in;
    apply_dev_mode_env_defaults(param.dev_mode, param);
    apply_dev_mode_cparam_defaults(param, param.user_flags);
    dev_mode_ = param.dev_mode;
    top_p_ = param.top_p;
    top_k_ = param.top_k;
    temperature_ = param.temperature;
    n_batch_ = param.n_batch > 0 ? param.n_batch : param.ctx_size;
    n_ubatch_ = param.n_ubatch > 0 ? param.n_ubatch : n_batch_;
    ctx_size_ = param.ctx_size;
    max_sequences_ = std::max(1, param.max_sequences);
    // GPU argmax / top-k avoids a full-vocab logits D2H every decode step (~200+
    // tok/s on GPU vs ~150 tok/s with CPU sampling). Mixed-mode correctness no
    // longer depends on this path after per-layer DeltaNet gating.
    use_gpu_topk_sampling_ = (
        dev_mode_ != DevMode::CPU_MODE &&
        ((temperature_ > 0.0f && top_k_ > 0) || temperature_ <= 0.0f)
    );

    ggml_backend_load_all();
    init_ggml_logging();

    model_ = std::make_shared<Qwen35moeModel>();
    if (!model_->init(
            param.model_path,
            param.dev_mode,
            param.n_threads,
            param.gpu_layer,
            param.no_mmap,
            param.gpu_id,
            param.tensor_overrides,
            param.n_threads_batch)) {
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
    forward_pass_->init(param.ctx_size, static_cast<uint32_t>(max_sequences_), model_,
        static_cast<uint32_t>(n_batch_), static_cast<uint32_t>(n_ubatch_),
        param.enable_paged_kv, param.paged_kv_block_size);
    if (use_gpu_topk_sampling_) {
        const int device_top_k = temperature_ > 0.0f ? top_k_ : 1;
        forward_pass_->configure_device_sampling(device_top_k, temperature_);
    }
    if (param.flash_attention) {
        forward_pass_->set_flash_attention_enabled(true);
    }

    // MTP speculative decoding: only for a single sequence, greedy sampling, and
    // non-mixed placement (the verify/rollback path assumes one slot and a
    // homogeneous trunk). Otherwise fall back to standard decode with a notice.
    if (param.mtp) {
        const bool greedy = (temperature_ <= 0.0f) || (top_k_ == 1);
        const bool single = (max_sequences_ == 1);
        const bool mixed = model_->is_mixed_mode();
        if (greedy && single && !mixed) {
            forward_pass_->configure_mtp(true, param.spec_draft_n_max);
            mtp_loop_enabled_ = forward_pass_->mtp_active();
        }
        if (!mtp_loop_enabled_) {
            std::fprintf(stderr,
                "[mtp] --mtp ignored: requires --parallel 1, greedy sampling "
                "(temp<=0 or top_k=1), non-mixed placement, and a model with a "
                "nextn block (greedy=%d single=%d mixed=%d).\n",
                greedy ? 1 : 0, single ? 1 : 0, mixed ? 1 : 0);
        }
    }

    eos_token_id_ = static_cast<int32_t>(model_->meta_->tokenizer.ggml_eos_token_id);
    auto special_id = [&](const char* token) -> int32_t {
        const int32_t id = tokenizer_->get_special_token_id(token);
        const auto& vocab = model_->meta_->tokenizer.ggml_tokens;
        if (id >= 0 && static_cast<size_t>(id) < vocab.size() && vocab[static_cast<size_t>(id)] == token) {
            return id;
        }
        return -1;
    };
    im_end_token_id_ = special_id("<|im_end|>");
    endoftext_token_id_ = special_id("<|endoftext|>");
    needs_string_stop_check_ = (im_end_token_id_ < 0 || endoftext_token_id_ < 0);

    {
        std::lock_guard<std::mutex> lock(batch_mutex_);
        free_slots_.clear();
        free_slots_.reserve(static_cast<size_t>(max_sequences_));
        for (int i = 0; i < max_sequences_; ++i) {
            free_slots_.push_back(static_cast<uint32_t>(i));
        }
        batch_stop_ = false;
    }
    batch_worker_ = std::thread(&ChatEngine::batch_worker_loop, this);
    std::printf("[batch] continuous batching enabled: max_sequences=%d, gpu_topk=%s\n",
        max_sequences_, use_gpu_topk_sampling_ ? "on" : "off");

    std::string bos_token;
    std::string eos_token;
    if (eos_token_id_ >= 0) {
        eos_token = tokenizer_->decode(eos_token_id_);
    }
    chat_template_ = std::make_unique<ChatTemplateApplier>(
        model_->meta_->tokenizer.chat_template, bos_token, eos_token);
    if (chat_template_->ready()) {
        std::printf("[chat] GGUF Jinja chat template loaded (%zu bytes)\n",
            model_->meta_->tokenizer.chat_template.size());
    } else {
        std::fprintf(stderr, "[chat] WARNING: GGUF chat template missing or failed to compile\n");
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

bool ChatEngine::run_complete(const std::string& prompt, const int max_tokens, std::string& response) {
    TimingStats discard;
    return run_complete(prompt, max_tokens, response, discard);
}

bool ChatEngine::run_complete(const std::string& prompt, const int max_tokens, std::string& response,
                              TimingStats& timing) {
    response.clear();
    timing = TimingStats{};
    if (max_tokens <= 0) {
        timing.ok = true;
        return true;
    }
    if (!batch_worker_.joinable()) {
        std::fprintf(stderr, "ERROR: ChatEngine batch worker is not running\n");
        return false;
    }

    auto request = std::make_shared<CompletionRequest>();
    request->prompt = prompt;
    request->max_tokens = max_tokens;
    request->enqueue_time = std::chrono::steady_clock::now();

    {
        std::lock_guard<std::mutex> lock(batch_mutex_);
        if (batch_stop_) {
            std::fprintf(stderr, "ERROR: ChatEngine is stopping\n");
            return false;
        }
        pending_requests_.push_back(request);
    }
    batch_cv_.notify_one();

    std::unique_lock<std::mutex> lock(request->mutex);
    request->cv.wait(lock, [&] { return request->done; });
    response = request->response;
    timing = request->timing;
    if (!request->ok) {
        std::fprintf(stderr, "ERROR: completion failed: %s\n", request->error.c_str());
        return false;
    }
    return true;
}

bool ChatEngine::prefill_sequence(ActiveSequence& seq, std::string& error) {
    try {
        using Clock = std::chrono::steady_clock;
        seq.prefill_start_time = Clock::now();
        seq.prefill_timed = true;

        forward_pass_->reset_sequence(seq.slot);
        seq.tokens = tokenizer_->encode(seq.request->prompt);
        seq.prompt_tokens = seq.tokens.size();
        if (seq.tokens.empty()) {
            seq.prefill_end_time = Clock::now();
            error = "Prompt tokenization produced no tokens";
            return false;
        }
        seq.tokens.reserve(seq.tokens.size() + static_cast<size_t>(seq.request->max_tokens));

        if (use_gpu_topk_sampling_) {
            TopKSampleCandidates candidates = forward_pass_->run_prefill_topk(seq.tokens, 0, seq.slot, sched_);
            seq.next_token_id = sample_from_topk_candidates(candidates, top_p_);
        } else {
            std::vector<float> logits = forward_pass_->run_prefill(seq.tokens, 0, seq.slot, sched_);
            const size_t vocab_size = model_->meta_->tokenizer.ggml_tokens.size();
            if (logits.size() < vocab_size) {
                error = "Prefill logits are smaller than vocabulary";
                return false;
            }
            std::vector<float> last_token_logits(logits.end() - static_cast<std::ptrdiff_t>(vocab_size), logits.end());
            seq.next_token_id = sampler_->sample(last_token_logits, seq.tokens);
        }

        seq.prefill_end_time = Clock::now();
        const double prefill_ms = elapsed_ms(seq.prefill_start_time, seq.prefill_end_time);
        std::printf("[batch][slot=%u] admitted prompt_tokens=%zu prefill=%.2f ms (%.2f tok/s)\n",
            seq.slot,
            seq.prompt_tokens,
            prefill_ms,
            tokens_per_second(seq.prompt_tokens, prefill_ms));
        return true;
    } catch (const std::exception& ex) {
        if (seq.prefill_timed) {
            seq.prefill_end_time = std::chrono::steady_clock::now();
        }
        error = ex.what();
        return false;
    }
}

bool ChatEngine::mtp_emit_token(ActiveSequence& seq, int32_t token_id) {
    if (is_stop_token(token_id)) {
        return true;
    }
    const std::string decoded = tokenizer_->decode(token_id);
    if (needs_string_stop_check_ &&
        (decoded == "<|im_end|>" || decoded == "<|endoftext|>")) {
        return true;
    }
    seq.response += decoded;
    seq.tokens.push_back(token_id);
    seq.generated++;
    if (seq.generated >= seq.request->max_tokens) {
        return true;
    }
    return false;
}

bool ChatEngine::generate_mtp(ActiveSequence& seq, std::string& error) {
    try {
        using Clock = std::chrono::steady_clock;
        seq.prefill_start_time = Clock::now();
        seq.prefill_timed = true;

        forward_pass_->reset_sequence(seq.slot);
        seq.tokens = tokenizer_->encode(seq.request->prompt);
        seq.prompt_tokens = seq.tokens.size();
        if (seq.tokens.empty()) {
            seq.prefill_end_time = Clock::now();
            error = "Prompt tokenization produced no tokens";
            return false;
        }

        // Prompt tokens vector is consumed by mtp_prefill; keep a copy so the
        // record of emitted tokens stays clean.
        std::vector<int32_t> prompt_tokens = seq.tokens;

        std::vector<float> seed_h;
        int pos = 0;
        int32_t seed = forward_pass_->mtp_prefill(prompt_tokens, seq.slot, seed_h, pos, sched_);

        seq.prefill_end_time = Clock::now();
        const double prefill_ms = elapsed_ms(seq.prefill_start_time, seq.prefill_end_time);
        std::printf("[mtp][slot=%u] admitted prompt_tokens=%zu prefill=%.2f ms (%.2f tok/s)\n",
            seq.slot, seq.prompt_tokens, prefill_ms,
            tokens_per_second(seq.prompt_tokens, prefill_ms));

        bool stop = mtp_emit_token(seq, seed);
        while (!stop) {
            // Leave room for the bonus token inside the context window.
            if (ctx_size_ > 0 && pos >= ctx_size_ - 2) {
                break;
            }
            std::vector<int32_t> confirmed;
            forward_pass_->mtp_step(seed, pos, seed_h, seq.slot, confirmed, sched_);
            for (int32_t tok : confirmed) {
                if (mtp_emit_token(seq, tok)) {
                    stop = true;
                    break;
                }
            }
        }

        forward_pass_->mtp_log_stats();
        return true;
    } catch (const std::exception& ex) {
        if (seq.prefill_timed) {
            seq.prefill_end_time = std::chrono::steady_clock::now();
        }
        error = ex.what();
        return false;
    }
}

void ChatEngine::finish_sequence(ActiveSequence& seq, bool ok, const std::string& error) {
    log_sequence_timing(seq, ok, error);
    finish_request(seq.request, seq.response, ok, error);
}

void ChatEngine::log_sequence_timing(const ActiveSequence& seq, bool ok, const std::string& error) {
    if (!seq.request || !seq.prefill_timed) {
        return;
    }

    const auto finish_time = std::chrono::steady_clock::now();
    const double queue_ms = elapsed_ms(seq.request->enqueue_time, seq.prefill_start_time);
    const double prefill_ms = elapsed_ms(seq.prefill_start_time, seq.prefill_end_time);
    const double decode_ms = std::max(0.0, elapsed_ms(seq.prefill_end_time, finish_time));
    const double inference_ms = elapsed_ms(seq.prefill_start_time, finish_time);
    const double total_ms = elapsed_ms(seq.request->enqueue_time, finish_time);
    const size_t generated = static_cast<size_t>(std::max(0, seq.generated));

    // Make the same numbers we print available to programmatic callers
    // (bench harness etc.). Done before finish_request flips `done`, so the
    // waiter sees a fully populated TimingStats on wake-up.
    seq.request->timing = TimingStats{
        seq.prompt_tokens,
        generated,
        queue_ms,
        prefill_ms,
        decode_ms,
        inference_ms,
        total_ms,
        ok,
        true,
    };

    std::printf(
        "[Timing][slot=%u][%s] prompt_tokens=%zu generated_tokens=%zu "
        "queue=%.2f ms prefill=%.2f ms (%.2f tok/s) "
        "decode=%.2f ms (%.2f tok/s) total=%.2f ms e2e=%.2f ms output=%.2f tok/s all=%.2f tok/s\n",
        seq.slot,
        ok ? "ok" : "error",
        seq.prompt_tokens,
        generated,
        queue_ms,
        prefill_ms,
        tokens_per_second(seq.prompt_tokens, prefill_ms),
        decode_ms,
        tokens_per_second(generated, decode_ms),
        inference_ms,
        total_ms,
        tokens_per_second(generated, inference_ms),
        tokens_per_second(seq.prompt_tokens + generated, inference_ms)
    );
    if (!ok && !error.empty()) {
        std::printf("[Timing][slot=%u][error] %s\n", seq.slot, error.c_str());
    }
}

void ChatEngine::finish_request(const std::shared_ptr<CompletionRequest>& request,
    const std::string& response, bool ok, const std::string& error) {
    {
        std::lock_guard<std::mutex> lock(request->mutex);
        request->response = response;
        request->ok = ok;
        request->error = error;
        request->done = true;
    }
    request->cv.notify_one();
}

void ChatEngine::release_slot(uint32_t slot) {
    {
        std::lock_guard<std::mutex> lock(batch_mutex_);
        free_slots_.push_back(slot);
    }
    batch_cv_.notify_one();
}

bool ChatEngine::is_stop_token(int token_id) const {
    return token_id == eos_token_id_ ||
        (im_end_token_id_ >= 0 && token_id == im_end_token_id_) ||
        (endoftext_token_id_ >= 0 && token_id == endoftext_token_id_);
}

void ChatEngine::batch_worker_loop() {
    std::vector<ActiveSequence> active;
    active.reserve(static_cast<size_t>(max_sequences_));

    while (true) {
        std::vector<std::pair<uint32_t, std::shared_ptr<CompletionRequest>>> admissions;
        std::deque<std::shared_ptr<CompletionRequest>> canceled_pending;
        bool stopping = false;

        {
            std::unique_lock<std::mutex> lock(batch_mutex_);
            batch_cv_.wait(lock, [&] {
                return batch_stop_ || !pending_requests_.empty() || !active.empty();
            });

            if (batch_stop_) {
                stopping = true;
                canceled_pending.swap(pending_requests_);
            } else {
                const size_t admission_budget = active.empty() ? free_slots_.size() : (free_slots_.empty() ? 0 : 1);
                while (!pending_requests_.empty() && !free_slots_.empty() && admissions.size() < admission_budget) {
                    const uint32_t slot = free_slots_.back();
                    free_slots_.pop_back();
                    admissions.emplace_back(slot, pending_requests_.front());
                    pending_requests_.pop_front();
                }
            }
        }

        for (auto& request : canceled_pending) {
            finish_request(request, "", false, "ChatEngine is stopping");
        }
        if (stopping) {
            for (auto& seq : active) {
                finish_sequence(seq, false, "ChatEngine is stopping");
            }
            active.clear();
            break;
        }

        for (auto& admission : admissions) {
            ActiveSequence seq;
            seq.slot = admission.first;
            seq.request = std::move(admission.second);
            if (seq.request->max_tokens <= 0) {
                finish_request(seq.request, "", true);
                release_slot(seq.slot);
                continue;
            }

            std::string error;
            if (mtp_loop_enabled_) {
                // MTP runs the whole sequence (prefill + speculative decode)
                // synchronously on this worker thread; nothing joins `active`.
                const bool ok = generate_mtp(seq, error);
                finish_sequence(seq, ok, error);
                release_slot(seq.slot);
                continue;
            }
            if (!prefill_sequence(seq, error)) {
                finish_sequence(seq, false, error);
                release_slot(seq.slot);
                continue;
            }
            active.push_back(std::move(seq));
        }

        if (active.empty()) {
            continue;
        }

        std::vector<int32_t> decode_tokens;
        std::vector<uint32_t> decode_slots;
        std::vector<int32_t> decode_positions;
        std::vector<size_t> decode_active_indices;
        decode_tokens.reserve(active.size());
        decode_slots.reserve(active.size());
        decode_positions.reserve(active.size());
        decode_active_indices.reserve(active.size());

        std::vector<ActiveSequence> survivors;
        survivors.reserve(active.size());
        for (auto& seq : active) {
            if (is_stop_token(seq.next_token_id)) {
                finish_sequence(seq, true);
                release_slot(seq.slot);
                continue;
            }

            std::string decoded_token = tokenizer_->decode(seq.next_token_id);
            if (needs_string_stop_check_ &&
                (decoded_token == "<|im_end|>" || decoded_token == "<|endoftext|>")) {
                finish_sequence(seq, true);
                release_slot(seq.slot);
                continue;
            }

            seq.response += decoded_token;
            seq.tokens.push_back(seq.next_token_id);
            seq.generated++;

            if (seq.generated >= seq.request->max_tokens) {
                finish_sequence(seq, true);
                release_slot(seq.slot);
                continue;
            }

            const uint32_t current_pos = forward_pass_->get_cache_pos(seq.slot);
            if (ctx_size_ > 0 && current_pos >= static_cast<uint32_t>(ctx_size_)) {
                finish_sequence(seq, true);
                release_slot(seq.slot);
                continue;
            }

            decode_tokens.push_back(seq.next_token_id);
            decode_slots.push_back(seq.slot);
            decode_positions.push_back(static_cast<int32_t>(current_pos));
            decode_active_indices.push_back(survivors.size());
            survivors.push_back(std::move(seq));
        }
        active = std::move(survivors);

        if (decode_tokens.empty()) {
            continue;
        }

        try {
            using Clock = std::chrono::steady_clock;
            auto start = Clock::now();
            if (use_gpu_topk_sampling_) {
                std::vector<TopKSampleCandidates> candidates =
                    forward_pass_->run_decode_batch_topk(decode_tokens, decode_slots, decode_positions, sched_);
                for (size_t i = 0; i < candidates.size(); ++i) {
                    active[decode_active_indices[i]].next_token_id =
                        sample_from_topk_candidates(candidates[i], top_p_);
                }
            } else {
                std::vector<std::vector<float>> logits_batch =
                    forward_pass_->run_decode_batch(decode_tokens, decode_slots, decode_positions, sched_);
                for (size_t i = 0; i < logits_batch.size(); ++i) {
                    ActiveSequence& seq = active[decode_active_indices[i]];
                    seq.next_token_id = sampler_->sample(logits_batch[i], seq.tokens);
                }
            }
            auto end = Clock::now();
            const double decode_ms = std::chrono::duration<double, std::milli>(end - start).count();
            static uint64_t step = 0;
            step++;
            if (step <= 8 || step % 64 == 0 || decode_tokens.size() > 1) {
                std::printf("[batch][decode step=%llu] batch=%zu forward+sample=%.3f ms\n",
                    static_cast<unsigned long long>(step), decode_tokens.size(), decode_ms);
            }
        } catch (const std::exception& ex) {
            std::vector<bool> failed(active.size(), false);
            for (size_t idx : decode_active_indices) {
                if (idx < failed.size()) {
                    failed[idx] = true;
                }
            }

            std::vector<ActiveSequence> retry_survivors;
            retry_survivors.reserve(active.size());
            for (size_t i = 0; i < active.size(); ++i) {
                if (failed[i]) {
                    finish_sequence(active[i], false, ex.what());
                    release_slot(active[i].slot);
                } else {
                    retry_survivors.push_back(std::move(active[i]));
                }
            }
            active = std::move(retry_survivors);
        }
    }
}

bool ChatEngine::chat_template_ready() const {
    return chat_template_ && chat_template_->ready();
}

ChatTemplateApplyResult ChatEngine::build_chat_prompt(const std::string& request_json) const {
    if (!chat_template_ || !chat_template_->ready()) {
        ChatTemplateApplyResult out{};
        out.error = "chat template unavailable";
        return out;
    }
    return chat_template_->apply_request_json(request_json);
}

ChatTemplateParseResult ChatEngine::parse_chat_response(const std::string& raw,
                                                        bool enable_thinking) const {
    if (!chat_template_ || !chat_template_->ready()) {
        ChatTemplateParseResult out{};
        out.content = raw;
        return out;
    }
    return chat_template_->parse_assistant_output(raw, enable_thinking);
}
