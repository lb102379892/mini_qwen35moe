#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <utility>
#include <regex>
#include <mutex>
#include "model/metadata.h"

// Tokenizer-specific structures
enum class TokenType : uint8_t {
    NORMAL = 1,
    UNKNOWN = 2,
    CONTROL = 3,
    USER_DEFINED = 4,
    UNUSED = 5,
    BYTE = 6
};

// Custom hash for std::pair, enabling its use as a key in std::unordered_map.
struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

class Tokenizer {
public:
    Tokenizer(const TokenizerInfo& tok_info);

    int init();
    std::vector<int32_t> encode(const std::string& text) const;
    std::string decode(int32_t token_id) const;
    std::string decode(const std::vector<int32_t>& token_ids) const;
    int32_t get_special_token_id(const std::string& token_str) const;
    int32_t get_eos_token_id() const;

    // Vocabulary access
    const std::vector<std::string>& get_vocabulary() const;
private:
    // Special token handling
    void initialize_special_tokens();
    void initialize_byte_mapping();

    // Pre-tokenization
    std::vector<std::string> pretokenize(const std::string& text) const;

    // BPE core algorithm
    std::vector<int32_t> apply_bpe(const std::vector<int32_t>& byte_tokens) const;
    
    std::vector<int32_t> encode_with_special_tokens(const std::string& text) const;
    std::vector<int32_t> encode_single_token(const std::string& text) const;
    bool is_special_token(int32_t token_id) const;

private:
    const TokenizerInfo& tok_info_;

    // Core BPE components
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<std::pair<int32_t, int32_t>, std::pair<int, int32_t>, pair_hash> merges_;
    
    // BPE cache (thread-safe)
    mutable std::unordered_map<std::string, std::vector<int32_t>> bpe_cache_;
    mutable std::mutex bpe_cache_mutex_;

    // Special tokens management
    int32_t unk_token_id_;
    std::unordered_set<int32_t> special_token_ids_;
    std::unordered_map<std::string, int32_t> special_tokens_;
    
    // Pre-tokenization regex
    std::regex pretokenization_regex_;
    
    // UTF-8 byte mapping
    std::vector<std::string> byte_encoder_;
    std::unordered_map<std::string, int32_t> byte_decoder_;
};