#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include "model/model.h"
#include "pipeline/chat_template.h"
#include "pipeline/sampling.h"
#include "pipeline/tokenizer.h"
#include "graph/graph.h"

// Per-request timing snapshot, populated by the engine when the request
// completes. Numbers come straight from steady_clock measurements taken
// inside the batch worker, so they're directly comparable across runs and
// across parameter combinations — no client-side wall-clock guesswork.
struct TimingStats {
    size_t prompt_tokens     = 0;
    size_t generated_tokens  = 0;
    double queue_ms          = 0.0;  // enqueue → prefill_start
    double prefill_ms        = 0.0;  // prefill_start → prefill_end
    double decode_ms         = 0.0;  // prefill_end → finish
    double inference_ms      = 0.0;  // prefill_start → finish
    double total_ms          = 0.0;  // enqueue → finish (queue + inference)
    bool   ok                = false;
    bool   timed             = false; // false if the request never reached prefill
};

class ChatEngine {
public:
    ChatEngine();
    ~ChatEngine();

    bool init(const CParam& param);
    bool run_complete(const std::string& prompt, const int max_tokens, std::string& response);
    // Same as above, but also returns precise timing. Used by the benchmark
    // harness (tests/perf/bench_main.cpp). The HTTP server still uses the
    // 3-arg overload.
    bool run_complete(const std::string& prompt, const int max_tokens, std::string& response,
                      TimingStats& timing);

    ChatTemplateApplyResult build_chat_prompt(const std::string& request_json) const;
    ChatTemplateParseResult parse_chat_response(const std::string& raw, bool enable_thinking) const;
    bool chat_template_ready() const;

private:
    struct CompletionRequest {
        std::string prompt;
        int max_tokens = 0;
        std::string response;
        bool done = false;
        bool ok = false;
        std::string error;
        std::chrono::steady_clock::time_point enqueue_time;
        std::mutex mutex;
        std::condition_variable cv;
        // Filled by ChatEngine::log_sequence_timing immediately before
        // finish_request flips `done`, so the timing is visible to the
        // waiting caller as soon as it wakes up.
        TimingStats timing{};
    };

    struct ActiveSequence {
        std::shared_ptr<CompletionRequest> request;
        uint32_t slot = 0;
        std::vector<int32_t> tokens;
        std::string response;
        int next_token_id = 0;
        int generated = 0;
        size_t prompt_tokens = 0;
        bool prefill_timed = false;
        std::chrono::steady_clock::time_point prefill_start_time;
        std::chrono::steady_clock::time_point prefill_end_time;
    };

    int sample_from_topk_candidates(const TopKSampleCandidates& candidates, float top_p);
    void batch_worker_loop();
    bool prefill_sequence(ActiveSequence& seq, std::string& error);
    // MTP speculative-decode path for a single sequence (used when mtp_loop_enabled_).
    // Runs prefill + the draft/verify/accept loop to completion, filling seq.
    bool generate_mtp(ActiveSequence& seq, std::string& error);
    // Append one confirmed token to the sequence; returns true if generation
    // should stop (stop token or max_tokens reached).
    bool mtp_emit_token(ActiveSequence& seq, int32_t token_id);
    void finish_sequence(ActiveSequence& seq, bool ok, const std::string& error = "");
    // Logs the [Timing] line and also publishes a TimingStats snapshot to
    // seq.request->timing so programmatic callers (bench harness) can read
    // it once run_complete returns. Not const because of that publication.
    void log_sequence_timing(const ActiveSequence& seq, bool ok, const std::string& error);
    void finish_request(const std::shared_ptr<CompletionRequest>& request,
        const std::string& response, bool ok, const std::string& error = "");
    void release_slot(uint32_t slot);
    bool is_stop_token(int token_id) const;

private:
    DevMode dev_mode_ = DevMode::CPU_MODE;
    float top_p_ = 0.95f;
    int top_k_ = 40;
    float temperature_ = 0.7f;
    bool use_gpu_topk_sampling_ = false;
    int n_batch_ = 0;
    int n_ubatch_ = 0;
    int ctx_size_ = 0;
    int max_sequences_ = 1;
    int32_t eos_token_id_ = -1;
    int32_t im_end_token_id_ = -1;
    int32_t endoftext_token_id_ = -1;
    bool needs_string_stop_check_ = true;
    // When true, single-sequence generation uses the MTP (nextn) speculative
    // decode path instead of standard token-by-token decode.
    bool mtp_loop_enabled_ = false;
    std::mt19937 gpu_sampling_rng_;
    std::shared_ptr<Qwen35moeModel> model_ = nullptr;
    std::shared_ptr<Tokenizer> tokenizer_ = nullptr;
    std::shared_ptr<Sampler> sampler_ = nullptr;  
    std::shared_ptr<Qwen35moeForwardPass> forward_pass_ = nullptr;
    std::unique_ptr<ChatTemplateApplier> chat_template_;
    ggml_backend_sched_t sched_ = nullptr;

    std::thread batch_worker_;
    std::mutex batch_mutex_;
    std::condition_variable batch_cv_;
    std::deque<std::shared_ptr<CompletionRequest>> pending_requests_;
    std::vector<uint32_t> free_slots_;
    bool batch_stop_ = false;
};
