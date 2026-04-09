// pipeline/tokenizer.cpp
// BPE tokenizer implementation
//
#include "pipeline/tokenizer.hpp"
#include <cstdio>
#include <climits>
#include <cassert>

// ============================================================
// Public API
// ============================================================

void BPETokenizer::load(const TokenizerConfig& cfg) {
    vocab_size_   = (int)cfg.ggml_tokens.size();
    eos_token_id_ = (int)cfg.ggml_eos_token_id;

    // Build vocab maps
    id_to_token_ = cfg.ggml_tokens;
    for (int i = 0; i < vocab_size_; i++) {
        token_to_id_[cfg.ggml_tokens[i]] = i;
    }

    // Build BPE merge rank table
    // Each merge string looks like "piece1 piece2" (space-separated pair)
    for (int r = 0; r < (int)cfg.ggml_merges.size(); r++) {
        const std::string& m = cfg.ggml_merges[r];
        size_t sp = m.find(' ');
        if (sp == std::string::npos) continue;
        merge_rank_[{m.substr(0, sp), m.substr(sp + 1)}] = r;
    }

    build_byte_encoder();

    printf("[Tokenizer] Loaded: vocab_size=%d, merges=%d, eos=%d\n",
           vocab_size_, (int)cfg.ggml_merges.size(), eos_token_id_);
}

// ============================================================
// encode(): text → token ids
// ============================================================
std::vector<int> BPETokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    for (const std::string& word : pre_tokenize(text)) {
        // Map each byte of the word to its unicode string representation
        std::vector<std::string> chars;
        chars.reserve(word.size());
        for (unsigned char c : word) {
            chars.push_back(byte_enc_[c]);
        }
        // Apply BPE merges
        for (const auto& piece : bpe_apply(std::move(chars))) {
            auto it = token_to_id_.find(piece);
            if (it != token_to_id_.end()) {
                ids.push_back(it->second);
            }
            // If a piece is unknown, skip it (rare for byte-level BPE)
        }
    }
    return ids;
}

// ============================================================
// decode(): token ids → text
// ============================================================
std::string BPETokenizer::decode(const std::vector<int>& ids) const {
    std::string out;
    for (int id : ids) {
        if (id < 0 || id >= vocab_size_) continue;
        const std::string& tok = id_to_token_[id];
        // Iterate over UTF-8 characters in the token and map each back to a byte
        size_t i = 0;
        while (i < tok.size()) {
            unsigned char c0 = (unsigned char)tok[i];
            int len = 1;
            if      (c0 >= 0xF0) len = 4;
            else if (c0 >= 0xE0) len = 3;
            else if (c0 >= 0xC0) len = 2;
            std::string uch = tok.substr(i, len);
            i += len;
            auto it = byte_dec_.find(uch);
            if (it != byte_dec_.end()) {
                out += (char)it->second;
            } else {
                // Unknown mapping: pass-through the UTF-8 character as-is
                out += uch;
            }
        }
    }
    return out;
}

// ============================================================
// Private helpers
// ============================================================

void BPETokenizer::build_byte_encoder() {
    // Standard GPT-2 bytes_to_unicode:
    // Bytes that map to themselves (as unicode codepoints):
    //   33-126  (printable ASCII, excl. space)
    //   161-172 (¡¢£¤¥¦§¨©ª«¬)
    //   174-255 (®¯°±...ÿ)
    //
    // All other bytes (0-32, 127-160, 173) map to codepoints 256, 257, ...

    std::vector<bool> covered(256, false);

    auto add = [&](uint8_t b, int cp) {
        std::string s = cp_to_utf8(cp);
        byte_enc_[b] = s;
        byte_dec_[s] = b;
        covered[b]   = true;
    };

    for (int b = 33;  b <= 126; b++) add((uint8_t)b, b);
    for (int b = 161; b <= 172; b++) add((uint8_t)b, b);
    for (int b = 174; b <= 255; b++) add((uint8_t)b, b);

    int extra = 256;
    for (int b = 0; b < 256; b++) {
        if (!covered[b]) {
            add((uint8_t)b, extra++);
        }
    }
}

std::string BPETokenizer::cp_to_utf8(int cp) {
    std::string s;
    if (cp < 0x80) {
        s += (char)cp;
    } else if (cp < 0x800) {
        s += (char)(0xC0 | (cp >> 6));
        s += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += (char)(0xE0 | (cp >> 12));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    } else {
        s += (char)(0xF0 | (cp >> 18));
        s += (char)(0x80 | ((cp >> 12) & 0x3F));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    }
    return s;
}

// Pre-tokenize: split text into pre-tokens following the GPT-2 convention.
// Words that follow a space get the space prepended.
std::vector<std::string> BPETokenizer::pre_tokenize(const std::string& text) const {
    std::vector<std::string> result;
    std::string cur;
    bool after_space = false; // last byte was whitespace

    // Whitespace-preserving pre-tokenisation (space-prefix convention):
    // a leading space is attached to the following word, matching the GPT-2
    // tokeniser behaviour where Ġ (the visual symbol) represents a space-prefixed token.
    for (size_t i = 0; i < text.size(); ) {
        unsigned char c = (unsigned char)text[i];

        // Handle multi-byte UTF-8 – pass through as one chunk attached to current token
        int mb_len = 1;
        if      (c >= 0xF0) mb_len = 4;
        else if (c >= 0xE0) mb_len = 3;
        else if (c >= 0xC0) mb_len = 2;

        if (mb_len > 1) {
            // Multi-byte char: attach to current token
            cur += text.substr(i, mb_len);
            i += mb_len;
            after_space = false;
            continue;
        }

        // ASCII
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (!cur.empty()) {
                result.push_back(cur);
                cur.clear();
            }
            // Start the next token with the space (GPT-2 style Ġ prefix)
            cur += (char)c;
            after_space = true;
        } else {
            cur += (char)c;
            after_space = false;
        }
        i++;
    }
    if (!cur.empty()) result.push_back(cur);
    return result;
}

// Apply BPE merges iteratively (standard greedy lowest-rank merge).
std::vector<std::string> BPETokenizer::bpe_apply(std::vector<std::string> pieces) const {
    if (pieces.size() <= 1) return pieces;

    while (true) {
        int best_rank = INT_MAX;
        int best_idx  = -1;

        for (int i = 0; i + 1 < (int)pieces.size(); i++) {
            auto it = merge_rank_.find({pieces[i], pieces[i + 1]});
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx  = i;
            }
        }
        if (best_idx < 0) break;

        pieces[best_idx] += pieces[best_idx + 1];
        pieces.erase(pieces.begin() + best_idx + 1);
    }
    return pieces;
}
