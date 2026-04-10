// pipeline/tokenizer.hpp
// GPT-2 byte-level BPE tokenizer for Qwen3.5
// Loads vocabulary and merge rules directly from GGUF metadata (TokenizerConfig).
//
// Usage:
//   BPETokenizer tok;
//   tok.load(model.config.tokenizer);
//   auto ids = tok.encode("Hello, world!");
//   printf("%s\n", tok.decode(ids).c_str());
//
#ifndef QWEN35MOE_PIPELINE_TOKENIZER_HPP
#define QWEN35MOE_PIPELINE_TOKENIZER_HPP

#include "core/config.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <sstream>
#include <climits>

class BPETokenizer {
public:
    BPETokenizer() = default;

    // ============================================================
    // Load vocab + merges from TokenizerConfig
    // ============================================================
    bool load(const TokenizerConfig& cfg) {
        if (cfg.ggml_tokens.empty()) {
            fprintf(stderr, "[Tokenizer] ERROR: no tokens in config\n");
            return false;
        }

        vocab_size_ = (int)cfg.ggml_tokens.size();
        id_to_token_.resize(vocab_size_);

        for (int i = 0; i < vocab_size_; i++) {
            const std::string& tok = cfg.ggml_tokens[i];
            id_to_token_[i]  = tok;
            token_to_id_[tok] = i;
        }

        // Load merges: each merge is a string like "a b" (two tokens separated by space)
        for (int i = 0; i < (int)cfg.ggml_merges.size(); i++) {
            const std::string& m = cfg.ggml_merges[i];
            // Find the single space separator
            size_t sep = m.find(' ');
            if (sep == std::string::npos) continue;
            std::string left  = m.substr(0, sep);
            std::string right = m.substr(sep + 1);
            merge_rank_[{left, right}] = i;
        }

        // Build byte-to-unicode lookup table (GPT-2 style)
        build_byte_to_unicode();

        // Find special token IDs
        eos_id_     = (int)cfg.ggml_eos_token_id;
        pad_id_     = (int)cfg.ggml_padding_token_id;
        bos_id_     = -1; // Qwen3.5 has no explicit BOS, uses chat format

        // Find <|im_start|> and <|im_end|> by searching the vocab
        auto it_start = token_to_id_.find("<|im_start|>");
        auto it_end   = token_to_id_.find("<|im_end|>");
        im_start_id_ = (it_start != token_to_id_.end()) ? it_start->second : -1;
        im_end_id_   = (it_end   != token_to_id_.end()) ? it_end->second   : -1;

        // Find <|endoftext|>
        auto it_eot = token_to_id_.find("<|endoftext|>");
        if (it_eot != token_to_id_.end()) eot_id_ = it_eot->second;
        else eot_id_ = eos_id_;

        fprintf(stderr, "[Tokenizer] vocab_size=%d merges=%zu eos=%d im_start=%d im_end=%d\n",
                vocab_size_, cfg.ggml_merges.size(), eos_id_, im_start_id_, im_end_id_);
        return true;
    }

    // ============================================================
    // Encode text → token IDs
    // ============================================================
    std::vector<int> encode(const std::string& text, bool add_bos = false) const {
        std::vector<int> result;
        if (add_bos && bos_id_ >= 0) result.push_back(bos_id_);

        // Pre-tokenize: split text into "word pieces"
        // Each piece is a UTF-8 string encoded as GPT-2 unicode chars
        auto words = pretokenize(text);

        for (const auto& word : words) {
            auto tokens = bpe_encode_word(word);
            for (const auto& tok : tokens) {
                auto it = token_to_id_.find(tok);
                if (it != token_to_id_.end()) {
                    result.push_back(it->second);
                } else {
                    // Fallback: encode as individual bytes
                    for (unsigned char c : tok) {
                        // Try to find single byte token
                        std::string byte_tok = byte_to_unicode_str(c);
                        auto it2 = token_to_id_.find(byte_tok);
                        if (it2 != token_to_id_.end()) {
                            result.push_back(it2->second);
                        }
                        // else: skip unknown byte (shouldn't happen with full vocab)
                    }
                }
            }
        }
        return result;
    }

    // ============================================================
    // Encode a pre-formatted chat prompt (already includes special tokens as text)
    // ============================================================
    std::vector<int> encode_special(const std::string& text) const {
        // Handle special tokens by splitting on them first
        std::vector<int> result;

        // Known special tokens to detect
        static const char* SPECIAL_TOKS[] = {
            "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", nullptr
        };

        std::string remaining = text;
        while (!remaining.empty()) {
            // Find the earliest special token
            size_t earliest_pos = std::string::npos;
            const char* earliest_tok = nullptr;
            for (const char** st = SPECIAL_TOKS; *st; ++st) {
                size_t p = remaining.find(*st);
                if (p != std::string::npos && (earliest_pos == std::string::npos || p < earliest_pos)) {
                    earliest_pos = p;
                    earliest_tok = *st;
                }
            }

            if (earliest_pos == std::string::npos) {
                // No more special tokens
                auto ids = encode(remaining);
                result.insert(result.end(), ids.begin(), ids.end());
                break;
            }

            // Encode text before the special token
            if (earliest_pos > 0) {
                auto ids = encode(remaining.substr(0, earliest_pos));
                result.insert(result.end(), ids.begin(), ids.end());
            }

            // Encode the special token itself
            std::string sp(earliest_tok);
            auto it = token_to_id_.find(sp);
            if (it != token_to_id_.end()) {
                result.push_back(it->second);
            }

            remaining = remaining.substr(earliest_pos + sp.size());
        }
        return result;
    }

    // ============================================================
    // Decode token IDs → text
    // ============================================================
    std::string decode(const std::vector<int>& ids) const {
        std::string result;
        for (int id : ids) {
            if (id < 0 || id >= vocab_size_) continue;
            result += decode_token(id_to_token_[id]);
        }
        return result;
    }

    // Decode a single token ID to string
    std::string decode_one(int id) const {
        if (id < 0 || id >= vocab_size_) return "";
        return decode_token(id_to_token_[id]);
    }

    // ============================================================
    // Build chat-format prompt for Qwen3.5
    // ============================================================
    std::string make_chat_prompt(const std::string& user_msg,
                                  const std::string& system_msg = "You are a helpful assistant.") const {
        std::string prompt;
        if (im_start_id_ >= 0) {
            prompt += "<|im_start|>system\n" + system_msg + "<|im_end|>\n";
            prompt += "<|im_start|>user\n" + user_msg + "<|im_end|>\n";
            prompt += "<|im_start|>assistant\n";
        } else {
            // Fallback: plain prompt
            prompt = user_msg;
        }
        return prompt;
    }

    // ============================================================
    // Accessors
    // ============================================================
    int vocab_size()  const { return vocab_size_; }
    int eos_id()      const { return eos_id_; }
    int pad_id()      const { return pad_id_; }
    int bos_id()      const { return bos_id_; }
    int im_start_id() const { return im_start_id_; }
    int im_end_id()   const { return im_end_id_; }
    int eot_id()      const { return eot_id_; }

    bool is_stop_token(int id) const {
        return id == eos_id_ || id == eot_id_ || id == im_end_id_;
    }

private:
    // ============================================================
    // GPT-2 byte-to-unicode mapping
    // ============================================================
    uint32_t byte_to_unicode_[256] = {};  // byte → unicode codepoint
    uint8_t unicode_to_byte_[65536] = {}; // unicode codepoint → byte (for decode)
    bool has_unicode_to_byte_[65536] = {};

    void build_byte_to_unicode() {
        // Standard GPT-2 byte_to_unicode mapping
        // Printable ASCII: 33-126, 161-172, 174-255 → same codepoints
        std::vector<int> bs;
        for (int i = '!'; i <= '~'; i++) bs.push_back(i);
        for (int i = 0xA1; i <= 0xAC; i++) bs.push_back(i);
        for (int i = 0xAE; i <= 0xFF; i++) bs.push_back(i);

        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n);
                n++;
            }
        }
        for (int i = 0; i < 256; i++) {
            byte_to_unicode_[bs[i]] = cs[i];
        }
        // Build reverse mapping
        for (int i = 0; i < 256; i++) {
            uint32_t cp = byte_to_unicode_[i];
            if (cp < 65536) {
                unicode_to_byte_[cp] = (uint8_t)i;
                has_unicode_to_byte_[cp] = true;
            }
        }
    }

    // Convert a single byte to its GPT-2 unicode string (UTF-8 encoded)
    std::string byte_to_unicode_str(unsigned char b) const {
        uint32_t cp = byte_to_unicode_[b];
        return codepoint_to_utf8(cp);
    }

    // Convert a unicode codepoint to its UTF-8 string
    static std::string codepoint_to_utf8(uint32_t cp) {
        std::string result;
        if (cp < 0x80) {
            result += (char)cp;
        } else if (cp < 0x800) {
            result += (char)(0xC0 | (cp >> 6));
            result += (char)(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            result += (char)(0xE0 | (cp >> 12));
            result += (char)(0x80 | ((cp >> 6) & 0x3F));
            result += (char)(0x80 | (cp & 0x3F));
        } else {
            result += (char)(0xF0 | (cp >> 18));
            result += (char)(0x80 | ((cp >> 12) & 0x3F));
            result += (char)(0x80 | ((cp >> 6) & 0x3F));
            result += (char)(0x80 | (cp & 0x3F));
        }
        return result;
    }

    // ============================================================
    // Pre-tokenizer: split text into word pieces
    // GPT-2 style: space is a prefix of the following word (Ġ = U+0120)
    // We split on whitespace boundaries, keeping the space as prefix.
    // ============================================================
    std::vector<std::string> pretokenize(const std::string& text) const {
        std::vector<std::string> words;
        if (text.empty()) return words;

        // Convert each byte to GPT-2 unicode char, then split into words
        // where each word starts at position 0 or after a non-space unicode char
        // that followed the space character (Ġ U+0120).
        //
        // Simple approach: iterate over UTF-8 codepoints, convert bytes → GPT2 unicode,
        // split when we encounter the Ġ character (U+0120, the space mapping).

        // First convert raw bytes to GPT-2 unicode string
        std::string gpt2_text;
        gpt2_text.reserve(text.size() * 2);
        for (unsigned char c : text) {
            gpt2_text += codepoint_to_utf8(byte_to_unicode_[c]);
        }

        // Now split into words: a new word starts at the beginning OR when we
        // encounter Ġ (U+0120, encoded as 0xC4 0xA0 in UTF-8).
        // Each new word starts with Ġ (except the very first word if text doesn't start with space).
        const std::string SPACE_CP = codepoint_to_utf8(0x0120); // Ġ = U+0120

        std::string current;
        size_t i = 0;
        while (i < gpt2_text.size()) {
            // Read one UTF-8 codepoint
            unsigned char first = (unsigned char)gpt2_text[i];
            int len = 1;
            if (first >= 0xF0) len = 4;
            else if (first >= 0xE0) len = 3;
            else if (first >= 0xC0) len = 2;

            std::string cp_str = gpt2_text.substr(i, len);
            i += len;

            // Check if this codepoint is Ġ (the space)
            if (cp_str == SPACE_CP) {
                // Start new word; save current word if non-empty
                if (!current.empty()) {
                    words.push_back(current);
                }
                current = SPACE_CP; // start new word with space prefix
            } else {
                current += cp_str;
            }
        }
        if (!current.empty()) {
            words.push_back(current);
        }

        return words;
    }

    // ============================================================
    // BPE encoding of a single word piece
    // Returns vector of token strings
    // ============================================================
    std::vector<std::string> bpe_encode_word(const std::string& word) const {
        // Split word into individual UTF-8 codepoints
        std::vector<std::string> symbols;
        size_t i = 0;
        while (i < word.size()) {
            unsigned char first = (unsigned char)word[i];
            int len = 1;
            if (first >= 0xF0) len = 4;
            else if (first >= 0xE0) len = 3;
            else if (first >= 0xC0) len = 2;
            symbols.push_back(word.substr(i, len));
            i += len;
        }

        if (symbols.size() <= 1) return symbols;

        // Iteratively apply the highest-priority merge
        while (true) {
            int best_rank = INT_MAX;
            int best_pos  = -1;

            for (int j = 0; j + 1 < (int)symbols.size(); j++) {
                auto it = merge_rank_.find({symbols[j], symbols[j+1]});
                if (it != merge_rank_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_pos  = j;
                }
            }

            if (best_pos < 0) break; // No more merges possible

            // Merge symbols[best_pos] and symbols[best_pos+1]
            symbols[best_pos] = symbols[best_pos] + symbols[best_pos + 1];
            symbols.erase(symbols.begin() + best_pos + 1);
        }

        return symbols;
    }

    // ============================================================
    // Decode a single token string back to UTF-8 bytes
    // Reverses the GPT-2 byte-to-unicode mapping
    // ============================================================
    std::string decode_token(const std::string& token) const {
        // Convert GPT-2 unicode chars back to bytes
        std::string result;
        size_t i = 0;
        while (i < token.size()) {
            unsigned char first = (unsigned char)token[i];
            int len = 1;
            if (first >= 0xF0) len = 4;
            else if (first >= 0xE0) len = 3;
            else if (first >= 0xC0) len = 2;

            // Decode UTF-8 codepoint
            uint32_t cp = 0;
            if (len == 1) {
                cp = first;
            } else if (len == 2) {
                cp = ((first & 0x1F) << 6) | ((unsigned char)token[i+1] & 0x3F);
            } else if (len == 3) {
                cp = ((first & 0x0F) << 12) |
                     (((unsigned char)token[i+1] & 0x3F) << 6) |
                     ((unsigned char)token[i+2] & 0x3F);
            } else if (len == 4) {
                cp = ((first & 0x07) << 18) |
                     (((unsigned char)token[i+1] & 0x3F) << 12) |
                     (((unsigned char)token[i+2] & 0x3F) << 6) |
                     ((unsigned char)token[i+3] & 0x3F);
            }
            i += len;

            // Check if this codepoint maps back to a byte
            if (cp < 65536 && has_unicode_to_byte_[cp]) {
                result += (char)unicode_to_byte_[cp];
            } else if (cp < 0x80) {
                result += (char)cp;
            } else {
                // Keep as-is (might be a special character)
                result += token.substr(i - len, len);
            }
        }
        return result;
    }

    // ============================================================
    // Data members
    // ============================================================
    int vocab_size_ = 0;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;

    // merge_rank_[{left, right}] = priority index (lower = higher priority)
    struct PairHash {
        size_t operator()(const std::pair<std::string,std::string>& p) const {
            size_t h1 = std::hash<std::string>{}(p.first);
            size_t h2 = std::hash<std::string>{}(p.second);
            // Combine using boost-style hash_combine (avoids UB from large shifts)
            h1 ^= h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
            return h1;
        }
    };
    std::unordered_map<std::pair<std::string,std::string>, int, PairHash> merge_rank_;

    int eos_id_      = -1;
    int pad_id_      = -1;
    int bos_id_      = -1;
    int im_start_id_ = -1;
    int im_end_id_   = -1;
    int eot_id_      = -1;
};

#endif // QWEN35MOE_PIPELINE_TOKENIZER_HPP
