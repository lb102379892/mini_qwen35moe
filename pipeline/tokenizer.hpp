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
#include "unicode.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
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
    /**
     * @brief 加载分词器配置
     * 
     * 该方法负责从 TokenizerConfig 中加载分词器所需的所有配置信息，
     * 包括词汇表、合并规则、特殊标记等。
     * 
     * @param cfg 分词器配置对象
     * @return bool 是否成功加载配置
     */
    bool load(const TokenizerConfig& cfg) {
        // 检查配置中是否包含 tokens
        if (cfg.ggml_tokens.empty()) {
            fprintf(stderr, "[Tokenizer] ERROR: no tokens in config\n");
            return false;
        }

        // 初始化词汇表大小和 ID 到 token 的映射
        vocab_size_ = (int)cfg.ggml_tokens.size();
        id_to_token_.resize(vocab_size_);

        // 构建 token 到 ID 和 ID 到 token 的双向映射
        for (int i = 0; i < vocab_size_; i++) {
            const std::string& tok = cfg.ggml_tokens[i];
            id_to_token_[i]  = tok;      // ID 到 token 的映射
            token_to_id_[tok] = i;        // token 到 ID 的映射
        }

        // 加载合并规则：每个合并规则是一个形如 "a b" 的字符串（两个 token 用空格分隔）
        int n_merges = (int)cfg.ggml_merges.size();
        for (int i = 0; i < n_merges; i++) {
            const std::string& m = cfg.ggml_merges[i];

            size_t sep = m.find(' ');
            if (sep == std::string::npos) continue;
            std::string left  = m.substr(0, sep);
            std::string right = m.substr(sep + 1);
            merge_rank_[{left, right}] = i;
        }

        // 构建字节到 unicode 的查找表（GPT-2 风格）
        build_byte_to_unicode();

        // 查找特殊 token ID
        eos_id_     = (int)cfg.ggml_eos_token_id;     // 结束标记 ID
        pad_id_     = (int)cfg.ggml_padding_token_id; // 填充标记 ID
        bos_id_     = -1; // Qwen3.5 没有显式的 BOS 标记，使用聊天格式

        // 通过搜索词汇表查找 <|im_start|> 和 <|im_end|> 标记
        auto it_start = token_to_id_.find("<|im_start|>");
        auto it_end   = token_to_id_.find("<|im_end|>");
        im_start_id_ = (it_start != token_to_id_.end()) ? it_start->second : -1;
        im_end_id_   = (it_end   != token_to_id_.end()) ? it_end->second   : -1;

        // 查找 <|endoftext|> 标记
        auto it_eot = token_to_id_.find("<|endoftext|>");
        if (it_eot != token_to_id_.end()) eot_id_ = it_eot->second;
        else eot_id_ = eos_id_; // 如果不存在，则使用 eos_id_

        // 加载 token 类型并构建特殊 token 列表
        if (!cfg.ggml_token_type.empty()) {
            token_type_.resize(vocab_size_, 1); // 默认: NORMAL
            for (int i = 0; i < vocab_size_ && i < (int)cfg.ggml_token_type.size(); i++) {
                token_type_[i] = cfg.ggml_token_type[i];
            }
            // 收集类型为 3 (CONTROL) 和 4 (USER_DEFINED) 的标记作为特殊标记
            for (int i = 0; i < vocab_size_; i++) {
                int32_t t = token_type_[i];
                if (t == TOKEN_TYPE_CONTROL || t == TOKEN_TYPE_USER_DEFINED) {
                    special_tokens_sorted_.emplace_back(id_to_token_[i], i);
                }
            }
            // 按字符串长度降序排序，用于贪心最长匹配
            std::sort(special_tokens_sorted_.begin(), special_tokens_sorted_.end(),
                [](const std::pair<std::string,int>& a, const std::pair<std::string,int>& b) {
                    return a.first.size() > b.first.size();
                });
        } else {
            token_type_.assign(vocab_size_, 1); // 如果没有 token 类型信息，全部设为 NORMAL
        }

        // 自动构建 EOG (end-of-generation) 标记集合
        eog_ids_.clear();
        if (eos_id_ >= 0) eog_ids_.insert(eos_id_);
        if (eot_id_ >= 0) eog_ids_.insert(eot_id_);
        if (im_end_id_ >= 0) eog_ids_.insert(im_end_id_);

        // 扫描 CONTROL(type==3) 和 USER_DEFINED(type==4) 标记，查找已知的 EOG 模式
        for (int i = 0; i < vocab_size_; i++) {
            if (token_type_[i] != TOKEN_TYPE_CONTROL && token_type_[i] != TOKEN_TYPE_USER_DEFINED) continue;
            const std::string& text = id_to_token_[i];
            if (kEogTokenPatterns.count(text) > 0) {
                eog_ids_.insert(i);
            }
        }

        // 打印分词器信息
        fprintf(stderr, "[Tokenizer] vocab_size=%d merges=%zu eos=%d im_start=%d im_end=%d special=%zu\n",
                vocab_size_, cfg.ggml_merges.size(), eos_id_, im_start_id_, im_end_id_,
                special_tokens_sorted_.size());
        chat_template_ = cfg.chat_template;
        return true;
    }

    // ============================================================
    // Encode text → token IDs
    // ============================================================
    std::vector<int> encode(const std::string& text, bool add_bos = false) const {
        std::vector<int> result;
        if (add_bos && bos_id_ >= 0) result.push_back(bos_id_);

        if (special_tokens_sorted_.empty()) {
            // No special tokens: just pretokenize + BPE
            auto words = pretokenize(text);
            for (const auto& word : words) {
                bpe_encode_and_push(word, result);
            }
            return result;
        }

        // Greedy scan: find special tokens with longest-match at earliest position,
        // run BPE on non-special segments between them.
        size_t pos = 0;
        while (pos < text.size()) {
            size_t earliest_pos = std::string::npos;
            int    earliest_id  = -1;
            size_t earliest_len = 0;

            for (const auto& sp : special_tokens_sorted_) {
                if (sp.first.empty()) continue;
                size_t p = text.find(sp.first, pos);
                if (p == std::string::npos) continue;
                if (earliest_pos == std::string::npos || p < earliest_pos ||
                    (p == earliest_pos && sp.first.size() > earliest_len)) {
                    earliest_pos = p;
                    earliest_id  = sp.second;
                    earliest_len = sp.first.size();
                }
            }

            if (earliest_pos == std::string::npos) {
                // No more special tokens — encode remainder with BPE
                if (pos < text.size()) {
                    auto words = pretokenize(text.substr(pos));
                    for (const auto& word : words) {
                        bpe_encode_and_push(word, result);
                    }
                }
                break;
            }

            // Encode non-special prefix with BPE
            if (earliest_pos > pos) {
                auto words = pretokenize(text.substr(pos, earliest_pos - pos));
                for (const auto& word : words) {
                    bpe_encode_and_push(word, result);
                }
            }

            // Emit the special token directly
            result.push_back(earliest_id);
            pos = earliest_pos + earliest_len;
        }
        return result;
    }

    // ============================================================
    // Decode token IDs → text
    // ============================================================
    std::string decode(const std::vector<int>& ids) const {
        std::string result;
        for (int id : ids) {
            result += decode_one(id);
        }
        return result;
    }

    // Decode a single token ID to string
    std::string decode_one(int id) const {
        if (id < 0 || id >= vocab_size_) return "";
        if (!token_type_.empty() && id < (int)token_type_.size()) {
            int32_t t = token_type_[id];
            if (t == 6) {
                // Byte token: format is <0xXX> (6 chars: '<','0','x',hi,lo,'>')
                // Parse the two hex digits at positions 3 and 4.
                const std::string& tok = id_to_token_[id];
                if (tok.size() == 6 && tok[0] == '<' && tok[1] == '0' && tok[2] == 'x' && tok[5] == '>') {
                    auto hex_digit = [](char c) -> unsigned char {
                        if (c >= '0' && c <= '9') return (unsigned char)(c - '0');
                        if (c >= 'a' && c <= 'f') return (unsigned char)(c - 'a' + 10);
                        if (c >= 'A' && c <= 'F') return (unsigned char)(c - 'A' + 10);
                        return 0;
                    };
                    unsigned char byte_val = (unsigned char)((hex_digit(tok[3]) << 4) | hex_digit(tok[4]));
                    return std::string(1, (char)byte_val);
                }
                return "";
            }
            if (t == 3 || t == 4) {
                // Control/user-defined special token: return raw string (no GPT-2 decoding)
                return id_to_token_[id];
            }
        }
        return decode_token(id_to_token_[id]);
    }

    // ============================================================
    // Build chat-format prompt for Qwen3.5
    // ============================================================
    std::string make_chat_prompt(const std::string& user_msg) const {
        std::string prompt;
        std::stringstream ss;
        if (im_start_id_ >= 0) {
            ss << "<|im_start|>user\n" << user_msg << "<|im_end|>\n";
            ss << "<|im_start|>assistant\n";
            prompt = ss.str();
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
        return eog_ids_.count(id) > 0;
    }

    bool has_chat_template() const {
        return !chat_template_.empty();
    }

    bool should_skip_output_token(int id) const {
        if (id < 0 || id >= vocab_size_) return true;
        if (!token_type_.empty() && id < (int)token_type_.size()) {
            int32_t t = token_type_[id];
            if (t == TOKEN_TYPE_CONTROL || t == TOKEN_TYPE_USER_DEFINED) {
                return true;
            }
        }
        return false;
    }

private:
    static constexpr int32_t TOKEN_TYPE_CONTROL = 3;
    static constexpr int32_t TOKEN_TYPE_USER_DEFINED = 4;
    inline static const std::unordered_set<std::string> kEogTokenPatterns = {
        "<|im_end|>",
        "<|endoftext|>",
        "<|end_of_text|>",
        "<|eot_id|>"
    };

    struct ChatMessage {
        std::string role;
        std::string content;
    };

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
    // Pre-tokenizer: split text into word pieces using qwen35 regex rules.
    // Returns GPT-2 byte-encoded word pieces (ready for BPE).
    // Regex: LLAMA_VOCAB_PRE_TYPE_QWEN2 from llama.cpp
    // ============================================================
    std::vector<std::string> pretokenize(const std::string& text) const {
        if (text.empty()) return {};
        // unicode_regex_split with byte_encode=true returns GPT-2 byte-encoded pieces
        return unicode_regex_split(text, {QWEN35_PRETOKENIZE_PATTERN}, true);
    }

    // ============================================================
    // BPE-encode one GPT-2 byte-encoded word piece and push token IDs to result.
    // ============================================================
    void bpe_encode_and_push(const std::string& word, std::vector<int>& result) const {
        auto tokens = bpe_encode_word(word);
        for (const auto& tok : tokens) {
            auto it = token_to_id_.find(tok);
            if (it != token_to_id_.end()) {
                result.push_back(it->second);
            } else {
                // Fallback: encode as individual bytes
                for (unsigned char c : tok) {
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

    // Pretokenizer regex for Qwen2/3.5 (LLAMA_VOCAB_PRE_TYPE_QWEN2 from llama.cpp)
    // Note: uses explicit case variants instead of (?i:...) which is not supported
    // by all regex engines (e.g. POSIX / std::regex ERE).
    static constexpr const char* QWEN35_PRETOKENIZE_PATTERN =
        "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    int vocab_size_ = 0;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;

    // token_type_[id]: 1=NORMAL, 3=CONTROL, 4=USER_DEFINED, 6=BYTE
    std::vector<int32_t> token_type_;
    // Special tokens (type==3 or type==4), sorted by string length descending
    std::vector<std::pair<std::string,int>> special_tokens_sorted_;

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
    std::string chat_template_;
    std::unordered_set<int> eog_ids_;
};

#endif // QWEN35MOE_PIPELINE_TOKENIZER_HPP
