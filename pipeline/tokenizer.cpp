#include <algorithm>
#include <limits>
#include <cstdio>
#include "tokenizer.h"

// ---------------------------------------------------------------------------
// GPT-2 byte-to-unicode mapping
// ---------------------------------------------------------------------------
// GPT-2 BPE uses a byte-to-unicode mapping where printable ASCII bytes map to
// themselves and non-printable bytes map to higher Unicode code points.
// This function builds the byte→char and char→byte tables.

static std::string byte_to_unicode_char(uint8_t b) {
    // GPT-2 byte-to-unicode: bijection from 256 byte values to 256 Unicode code points.
    //
    // Printable ASCII (33-126): map to U+0021-U+007E (same value, 1-byte UTF-8)
    // Latin-1 supplement (161-172, 174-255): map to U+00A1-U+00AC, U+00AE-U+00FF
    //   These are 2-byte UTF-8 sequences (U+0080-U+00FF → 0xC2/0xC3 prefix)
    // Remapped bytes (0-32, 127-160, 173): map to U+0100-U+0143 (2-byte UTF-8)
    //
    // CRITICAL: Bytes 161-172 and 174-255 map to Unicode code points that require
    // 2-byte UTF-8 encoding. Returning a raw byte (char(b)) would produce invalid
    // UTF-8 and break BPE symbol splitting for non-ASCII input (e.g., Japanese).

    if (b >= 33 && b <= 126) {
        // ASCII printable: code point == byte value, 1-byte UTF-8
        return std::string(1, static_cast<char>(b));
    }

    // For all other bytes, compute the Unicode code point and encode as UTF-8
    int cp;
    if ((b >= 161 && b <= 172) || b >= 174) {
        // Latin-1 supplement: code point == byte value (U+00A1-U+00FF)
        cp = b;
    } else if (b <= 32) {
        cp = 256 + b;            // bytes 0-32 → U+0100-U+0120
    } else if (b == 127) {
        cp = 256 + 33;           // byte 127 → U+0121
    } else if (b >= 128 && b <= 160) {
        cp = 256 + 34 + (b - 128); // bytes 128-160 → U+0122-U+0142
    } else {
        cp = 256 + 67;           // byte 173 → U+0143
    }

    // UTF-8 encode the code point (all code points here are U+00A1-U+0143, 2-byte UTF-8)
    char buf[4];
    buf[0] = static_cast<char>(0xC0 | ((cp >> 6) & 0x1F));
    buf[1] = static_cast<char>(0x80 | (cp & 0x3F));
    buf[2] = '\0';
    return std::string(buf, 2);
}

static std::unordered_map<std::string, uint8_t> build_unicode_to_byte() {
    std::unordered_map<std::string, uint8_t> result;
    for (int b = 0; b < 256; b++) {
        std::string u = byte_to_unicode_char(static_cast<uint8_t>(b));
        result[u] = static_cast<uint8_t>(b);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

bool Tokenizer::init(const TokenizerInfo& tokenizer_info) {
    if (tokenizer_info.ggml_tokens.empty()) {
        fprintf(stderr, "No vocabulary tokens in model config\n");
        return false;
    }

    // Build vocabulary mappings
    id_to_token_ = tokenizer_info.ggml_tokens;
    token_to_id_.reserve(id_to_token_.size());
    for (size_t i = 0; i < id_to_token_.size(); ++i) {
        token_to_id_[id_to_token_[i]] = static_cast<int>(i);
    }

    // Build merge rank map
    merge_ranks_.reserve(tokenizer_info.ggml_merges.size());
    for (size_t i = 0; i < tokenizer_info.ggml_merges.size(); ++i) {
        merge_ranks_[tokenizer_info.ggml_merges[i]] = static_cast<int>(i);
    }

    eos_token_id_ = tokenizer_info.ggml_eos_token_id;
    bos_token_id_ = -1;

    // Resolve chat template special tokens from vocabulary
    auto resolve = [&](const std::string& name) -> int {
        auto it = token_to_id_.find(name);
        return (it != token_to_id_.end()) ? it->second : -1;
    };
    im_start_token_id_ = resolve("<|im_start|>");
    im_end_token_id_ = resolve("<|im_end|>");

    initialized_ = true;
    fprintf(stderr, "Tokenizer initialized: vocab=%zu, merges=%zu, eos=%d, im_start=%d, im_end=%d\n",
             id_to_token_.size(), merge_ranks_.size(), eos_token_id_,
             im_start_token_id_, im_end_token_id_);
    return true;
}

// ---------------------------------------------------------------------------
// BPE encode a single word (already converted to unicode representation)
// ---------------------------------------------------------------------------

std::vector<int> Tokenizer::bpe_encode_word(const std::string& word) const {
    // Split word into individual characters (UTF-8 aware for the unicode-mapped bytes)
    std::vector<std::string> symbols;
    size_t i = 0;
    while (i < word.size()) {
        uint8_t c = static_cast<uint8_t>(word[i]);
        size_t char_len = 1;
        if ((c & 0x80) == 0) char_len = 1;
        else if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else char_len = 4;
        if (i + char_len > word.size()) char_len = 1;
        symbols.push_back(word.substr(i, char_len));
        i += char_len;
    }

    if (symbols.size() <= 1) {
        // Single character or empty — look up directly
        auto it = token_to_id_.find(word);
        if (it != token_to_id_.end()) {
            return {it->second};
        }
        // Unknown token — return empty
        return {};
    }

    // Iteratively merge the pair with the lowest merge rank
    while (symbols.size() > 1) {
        int best_rank = std::numeric_limits<int>::max();
        size_t best_pos = std::string::npos;

        for (size_t j = 0; j + 1 < symbols.size(); ++j) {
            std::string pair = symbols[j] + " " + symbols[j + 1];
            auto it = merge_ranks_.find(pair);
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos = j;
            }
        }

        if (best_pos == std::string::npos) break;  // No more merges possible

        // Apply the merge: combine symbols[best_pos] and symbols[best_pos+1]
        symbols[best_pos] = symbols[best_pos] + symbols[best_pos + 1];
        symbols.erase(symbols.begin() + static_cast<long>(best_pos) + 1);
    }

    // Convert symbols to token IDs
    std::vector<int> ids;
    ids.reserve(symbols.size());
    for (const auto& sym : symbols) {
        auto it = token_to_id_.find(sym);
        if (it != token_to_id_.end()) {
            ids.push_back(it->second);
        } else {
            fprintf(stderr, "Unknown BPE symbol: '%s'\n", sym.c_str());
        }
    }
    return ids;
}

// ---------------------------------------------------------------------------
// Encode: text → token IDs
// ---------------------------------------------------------------------------

std::vector<int> Tokenizer::encode(const std::string& text) const {
    if (!initialized_ || text.empty()) return {};

    std::vector<int> result;

    // GPT-2 pre-tokenization: split text into words using a regex pattern.
    // The pattern captures: contractions, letters, digits, non-letter/non-digit,
    // and whitespace sequences. Each match is independently BPE-encoded.
    //
    // Simplified pattern (captures whitespace-prefixed words and punctuation):
    // We convert each byte to its GPT-2 unicode representation before BPE.

    // Simple word splitting: split on whitespace boundaries, keeping space as
    // prefix (Ġ) on non-first words — matching GPT-2 convention.
    std::vector<std::string> words;
    std::string current_word;

    for (size_t i = 0; i < text.size(); ++i) {
        uint8_t c = static_cast<uint8_t>(text[i]);

        if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
            if (!current_word.empty()) {
                words.push_back(current_word);
                current_word.clear();
            }
            // Prefix the space character to the NEXT word (GPT-2 Ġ convention)
            current_word = byte_to_unicode_char(c);
        } else {
            current_word += byte_to_unicode_char(c);
        }
    }
    if (!current_word.empty()) {
        words.push_back(current_word);
    }

    // BPE-encode each word independently
    for (const auto& word : words) {
        auto ids = bpe_encode_word(word);
        result.insert(result.end(), ids.begin(), ids.end());
    }

    return result;
}

// ---------------------------------------------------------------------------
// Decode: token IDs → text
// ---------------------------------------------------------------------------

std::string Tokenizer::decode(const std::vector<int>& token_ids) const {
    if (!initialized_) return "";

    static const auto unicode_to_byte = build_unicode_to_byte();

    std::string raw;
    for (int id : token_ids) {
        if (id >= 0 && id < static_cast<int>(id_to_token_.size())) {
            raw += id_to_token_[static_cast<size_t>(id)];
        }
    }

    // Convert GPT-2 unicode representation back to bytes
    std::string result;
    size_t i = 0;
    while (i < raw.size()) {
        // Try multi-byte UTF-8 first (2-byte for the Ġ-style mapped bytes)
        uint8_t c = static_cast<uint8_t>(raw[i]);
        size_t char_len = 1;
        if ((c & 0x80) == 0) char_len = 1;
        else if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else char_len = 4;
        if (i + char_len > raw.size()) char_len = 1;

        std::string u = raw.substr(i, char_len);
        auto it = unicode_to_byte.find(u);
        if (it != unicode_to_byte.end()) {
            result += static_cast<char>(it->second);
        } else {
            // Pass through as-is (shouldn't happen with valid tokens)
            result += u;
        }
        i += char_len;
    }

    return result;
}

std::string Tokenizer::decode(int token_id) const {
    return decode(std::vector<int>{token_id});
}
