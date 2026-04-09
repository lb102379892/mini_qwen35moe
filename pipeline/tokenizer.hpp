// pipeline/tokenizer.hpp
// GPT-2 style BPE tokenizer for Qwen3.5 MoE
//
// Loads vocabulary and merge rules from the TokenizerConfig that was read
// from GGUF metadata.  Supports encode (text → token-ids) and decode
// (token-ids → text) using the standard byte-level encoding.
//
#ifndef FUNASR_PIPELINE_TOKENIZER_HPP
#define FUNASR_PIPELINE_TOKENIZER_HPP

#include "core/config.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <utility>

class BPETokenizer {
public:
    // Load vocabulary and merge rules from a TokenizerConfig.
    void load(const TokenizerConfig& cfg);

    // Encode text to a sequence of token ids.
    std::vector<int> encode(const std::string& text) const;

    // Decode a sequence of token ids back to UTF-8 text.
    std::string decode(const std::vector<int>& ids) const;

    int vocab_size()     const { return vocab_size_; }
    int eos_token_id()   const { return eos_token_id_; }
    bool is_loaded()     const { return vocab_size_ > 0; }

private:
    int vocab_size_    = 0;
    int eos_token_id_  = 0;

    std::vector<std::string>           id_to_token_;
    std::unordered_map<std::string,int> token_to_id_;

    // merge_rank_[{a,b}] = rank (lower = applied first)
    std::map<std::pair<std::string,std::string>, int> merge_rank_;

    // byte ↔ unicode-char string (GPT-2 byte encoding)
    std::string                           byte_enc_[256]; // byte  → unicode string
    std::unordered_map<std::string,uint8_t> byte_dec_;   // unicode string → byte

    // Build the standard GPT-2 byte encoder/decoder tables.
    void build_byte_encoder();

    // Convert a unicode codepoint to a UTF-8 std::string.
    static std::string cp_to_utf8(int cp);

    // Split text into pre-tokens (whitespace-aware, preserving space prefix).
    std::vector<std::string> pre_tokenize(const std::string& text) const;

    // Apply BPE merges to a list of per-byte unicode-string pieces.
    std::vector<std::string> bpe_apply(std::vector<std::string> pieces) const;
};

#endif // FUNASR_PIPELINE_TOKENIZER_HPP
