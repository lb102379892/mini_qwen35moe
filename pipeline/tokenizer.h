#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include "model/metadata.h"

// GPT-2 style byte-level BPE tokenizer.
// Initialized from GGUF tokenizer.ggml.* metadata (vocab + merges).
// DC-06: From-scratch implementation — no external tokenizer libraries.
class Tokenizer {
public:
    Tokenizer() = default;

    // Initialize from TokenizerInfo tokenizer data.
    // Returns false if vocab or merges are empty.
    bool init(const TokenizerInfo& tokenizer_info);

    // Encode text into token IDs using BPE.
    std::vector<int> encode(const std::string& text) const;

    // Decode token IDs back to text.
    std::string decode(const std::vector<int>& token_ids) const;
    std::string decode(int token_id) const;

    int eos_token_id() const { return eos_token_id_; }
    int bos_token_id() const { return bos_token_id_; }
    int im_start_token_id() const { return im_start_token_id_; }
    int im_end_token_id() const { return im_end_token_id_; }
    int vocab_size() const { return static_cast<int>(id_to_token_.size()); }
    bool initialized() const { return initialized_; }

    // Look up a token string → ID (returns -1 if not found)
    int token_to_id(const std::string& token) const {
        auto it = token_to_id_.find(token);
        return (it != token_to_id_.end()) ? it->second : -1;
    }

private:
    // BPE encode a single pre-tokenized word (byte sequence)
    std::vector<int> bpe_encode_word(const std::string& word) const;

    // Vocabulary mappings
    std::vector<std::string> id_to_token_;        // token_id → token string
    std::unordered_map<std::string, int> token_to_id_;  // token string → token_id

    // BPE merge rules: (token_a, token_b) → merge rank (lower = higher priority)
    std::unordered_map<std::string, int> merge_ranks_;  // "tokenA tokenB" → rank

    int eos_token_id_ = -1;
    int bos_token_id_ = -1;
    int im_start_token_id_ = -1;
    int im_end_token_id_ = -1;
    bool initialized_ = false;
};
