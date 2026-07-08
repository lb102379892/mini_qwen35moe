#pragma once

#include <string>

struct ChatTemplateApplyResult {
    std::string prompt;
    std::string generation_prompt;
    bool enable_thinking = false;
    bool ok = false;
    std::string error;
};

struct ChatTemplateParseResult {
    std::string content;
    std::string reasoning_content;
};

// Renders GGUF tokenizer.chat_template (Jinja) like llama.cpp --jinja.
class ChatTemplateApplier {
public:
    ChatTemplateApplier(const std::string& template_src,
                        const std::string& bos_token,
                        const std::string& eos_token);

    bool ready() const { return ready_; }

    ChatTemplateApplyResult apply_request_json(const std::string& request_json) const;
    ChatTemplateParseResult parse_assistant_output(const std::string& raw,
                                                   bool enable_thinking) const;

private:
    bool ready_ = false;
    std::string src_;
    std::string bos_token_;
    std::string eos_token_;
    struct CompiledTemplate;
    CompiledTemplate* compiled_ = nullptr;

public:
    ~ChatTemplateApplier();
    ChatTemplateApplier(const ChatTemplateApplier&) = delete;
    ChatTemplateApplier& operator=(const ChatTemplateApplier&) = delete;
};
