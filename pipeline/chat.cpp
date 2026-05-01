#include "pipeline/chat.h"


ChatEngine::ChatEngine() {
}

ChatEngine::~ChatEngine() {
}

bool ChatEngine::init(const std::string& model_path_, DevMode dev_mode, int n_threads, int max_seq_len, float top_p, int top_k, float temperature, int gpu_layer) {
    dev_mode_ = dev_mode;
    max_seq_len_= max_seq_len;

    ggml_backend_load_all();

    model_ = std::make_shared<Qwen35moeModel>();
    if (!model_->init(model_path_, dev_mode, n_threads, gpu_layer)) {
        fprintf(stderr, "ERROR: Failed to init model from %s\n", model_path_.c_str());
        return false;
    }

    tokenizer_ = std::make_shared<Tokenizer>(model_->meta_->tokenizer);
    if (tokenizer_->init() != 0) {
        fprintf(stderr, "ERROR: Failed to initialize tokenizer from model\n");
        return false;
    }

    if (temperature > 0.0f) {
        sampler_ = std::make_shared<TemperatureSampler>(temperature, 1.1f, 64, top_k, top_p);
    } else {
        sampler_ = std::make_shared<GreedySampler>();
    }
    sampler_->set_eos_token_id(model_->meta_->tokenizer.ggml_eos_token_id);
    sched_ = model_->get_scheduler();

    return true;
}

bool ChatEngine::run_complete(const std::string& prompt, const int max_tokens, std::string& response) {
    std::shared_ptr<Qwen35moeForwardPass> forward_pass = std::make_shared<Qwen35moeForwardPass>();
    auto m = model_->meta_;
    forward_pass->init(m->qwen35moe.context_length, 1, model_);

    std::vector<int32_t> tokens = tokenizer_->encode(prompt);

    const size_t n_prompt_tokens = tokens.size();
    std::vector<float> logits = forward_pass->run_prefill(tokens, 0, 0, sched_);

    size_t vocab_size = m->tokenizer.ggml_tokens.size();
    std::vector<float> last_token_logits(logits.end() - vocab_size, logits.end());
    int next_token_id = sampler_->sample(last_token_logits, tokens);

    // Decode phase
    const int32_t eos_token_id = m->tokenizer.ggml_eos_token_id;
    const std::string im_end_str = "<|im_end|>";
    const std::string eos_str = "<|endoftext|>";

    // PLD state for single-prompt mode
    std::vector<int32_t> prompt_tokens_for_pld = tokens;  // Original prompt
    std::vector<int32_t> generated_tokens;

    for (int i = 0; i < max_tokens; ++i) {
        std::string decoded_token = tokenizer_->decode(next_token_id);
        if (next_token_id == eos_token_id || decoded_token == im_end_str || decoded_token == eos_str) {
            break;
        }
        response += decoded_token;
        printf("%s", decoded_token.c_str());

        tokens.push_back(next_token_id);

        // --- Normal decode path ---
        std::vector<int32_t> current_token_vec = { next_token_id };
        int current_pos = forward_pass->get_cache_pos(0); // Slot 0

        std::vector<float> token_logits = forward_pass->run_prefill(current_token_vec, current_pos, 0, sched_);
        last_token_logits.assign(token_logits.begin(), token_logits.begin() + vocab_size);
        
        next_token_id = sampler_->sample(last_token_logits, tokens);
    }
    printf("\n");
    return true;
}
    