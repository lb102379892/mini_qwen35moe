#include "pipeline/chat_template.h"

#include <algorithm>
#include <cctype>
#include <nlohmann/json.hpp>
#include "jinja/parser.h"
#include "jinja/runtime.h"

using json = nlohmann::ordered_json;

namespace {

std::string trim_copy(std::string s) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

void map_developer_role_to_system(json& messages) {
    if (!messages.is_array()) {
        return;
    }
    for (auto& msg : messages) {
        if (msg.is_object() && msg.value("role", "") == "developer") {
            msg["role"] = "system";
        }
    }
}

bool kwargs_enable_thinking(const json& kwargs) {
    if (kwargs.contains("enable_thinking") && kwargs["enable_thinking"].is_boolean()) {
        return kwargs["enable_thinking"].get<bool>();
    }
    return false;
}

// Qwen GGUF Jinja closes an empty thinking block on the assistant prefix when
// enable_thinking=false, but many Qwen checkpoints still need /no_think on the
// last user turn to suppress verbose non-code output (llama.cpp users often rely
// on both; the template itself does not inject /no_think).
void apply_qwen_no_think_hint(json& messages, bool enable_thinking) {
    if (enable_thinking || !messages.is_array()) {
        return;
    }
    for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
        if (!it->is_object() || it->value("role", "") != "user") {
            continue;
        }
        if (!it->contains("content") || !(*it)["content"].is_string()) {
            return;
        }
        std::string content = (*it)["content"].get<std::string>();
        if (content.find("/think") != std::string::npos ||
            content.find("/no_think") != std::string::npos) {
            return;
        }
        if (!content.empty() && content.back() == '\n') {
            content += "/no_think";
        } else {
            content += "\n/no_think";
        }
        (*it)["content"] = content;
        return;
    }
}

std::string render_template(const jinja::program& prog,
                            const std::string& src,
                            const std::string& bos_token,
                            const std::string& eos_token,
                            const json& messages,
                            const json& template_kwargs,
                            bool add_generation_prompt) {
    jinja::context ctx(src);

    json inp = json{
        {"messages", messages},
        {"bos_token", bos_token},
        {"eos_token", eos_token},
        {"enable_thinking", kwargs_enable_thinking(template_kwargs)},
    };
    if (template_kwargs.is_object()) {
        for (auto it = template_kwargs.begin(); it != template_kwargs.end(); ++it) {
            inp[it.key()] = it.value();
        }
    }
    if (add_generation_prompt) {
        inp["add_generation_prompt"] = true;
    }

    jinja::global_from_json(ctx, inp, true);
    jinja::runtime runtime(ctx);
    const jinja::value results = runtime.execute(prog);
    auto parts = jinja::runtime::gather_string_parts(results);
    return parts->as_string().str();
}

std::string compute_generation_prompt(const jinja::program& prog,
                                    const std::string& src,
                                    const std::string& bos_token,
                                    const std::string& eos_token,
                                    const json& messages,
                                    const json& template_kwargs) {
    const std::string no_gen = render_template(
        prog, src, bos_token, eos_token, messages, template_kwargs, false);
    const std::string with_gen = render_template(
        prog, src, bos_token, eos_token, messages, template_kwargs, true);

    size_t prefix_len = 0;
    const size_t min_size = std::min(no_gen.size(), with_gen.size());
    while (prefix_len < min_size && no_gen[prefix_len] == with_gen[prefix_len]) {
        prefix_len++;
    }
    return with_gen.substr(prefix_len);
}

void strip_trailing_chat_control_tokens(std::string& text) {
    static const char* kSuffixes[] = {
        "<|endoftext|>",
        "<|im_end|>",
        "<|im_start|>",
    };
    bool changed = true;
    while (changed) {
        changed = false;
        text = trim_copy(text);
        for (const char* suffix : kSuffixes) {
            const std::string token(suffix);
            if (text.size() >= token.size() &&
                text.compare(text.size() - token.size(), token.size(), token) == 0) {
                text.erase(text.size() - token.size());
                changed = true;
            }
        }
    }
}

} // namespace

struct ChatTemplateApplier::CompiledTemplate {
    jinja::program prog;
};

ChatTemplateApplier::ChatTemplateApplier(const std::string& template_src,
                                         const std::string& bos_token,
                                         const std::string& eos_token)
    : src_(template_src),
      bos_token_(bos_token),
      eos_token_(eos_token) {
    if (template_src.empty()) {
        return;
    }
    try {
        jinja::lexer lexer;
        auto tokens = lexer.tokenize(template_src);
        auto* compiled = new CompiledTemplate();
        compiled->prog = jinja::parse_from_tokens(tokens);
        compiled_ = compiled;
        ready_ = true;
    } catch (const std::exception& ex) {
        ready_ = false;
    }
}

ChatTemplateApplier::~ChatTemplateApplier() {
    delete compiled_;
    compiled_ = nullptr;
}

ChatTemplateApplyResult ChatTemplateApplier::apply_request_json(const std::string& request_json) const {
    ChatTemplateApplyResult out{};
    if (!ready_ || compiled_ == nullptr) {
        out.error = "chat template unavailable";
        return out;
    }

    json body;
    try {
        body = json::parse(request_json);
    } catch (const std::exception& ex) {
        out.error = std::string("invalid JSON: ") + ex.what();
        return out;
    }

    if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty()) {
        out.error = "messages array required";
        return out;
    }

    json messages = body["messages"];
    map_developer_role_to_system(messages);

    json template_kwargs = json::object();
    if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object()) {
        template_kwargs = body["chat_template_kwargs"];
    }

    const bool enable_thinking = kwargs_enable_thinking(template_kwargs);
    apply_qwen_no_think_hint(messages, enable_thinking);

    try {
        out.prompt = render_template(
            compiled_->prog, src_, bos_token_, eos_token_, messages, template_kwargs, true);
        out.generation_prompt = compute_generation_prompt(
            compiled_->prog, src_, bos_token_, eos_token_, messages, template_kwargs);
        out.enable_thinking = kwargs_enable_thinking(template_kwargs);
        out.ok = !out.prompt.empty();
        if (!out.ok) {
            out.error = "rendered prompt is empty";
        }
    } catch (const std::exception& ex) {
        out.error = std::string("chat template render failed: ") + ex.what();
    }
    return out;
}

ChatTemplateParseResult ChatTemplateApplier::parse_assistant_output(const std::string& raw,
                                                                    bool enable_thinking) const {
    ChatTemplateParseResult out{};
    std::string text = raw;
    strip_trailing_chat_control_tokens(text);

    static const std::string kThinkStart = std::string("<") + "|redacted_thinking>";
    static const std::string kThinkEnd = std::string("</") + "|redacted_thinking>";

    std::string reasoning;
    for (;;) {
        const size_t start = text.find(kThinkStart);
        if (start == std::string::npos) {
            break;
        }
        const size_t reasoning_begin = start + kThinkStart.size();
        const size_t end = text.find(kThinkEnd, reasoning_begin);
        if (end == std::string::npos) {
            reasoning += text.substr(reasoning_begin);
            text.erase(start);
            break;
        }
        reasoning += text.substr(reasoning_begin, end - reasoning_begin);
        text.erase(start, end + kThinkEnd.size() - start);
    }

    out.reasoning_content = trim_copy(reasoning);
    out.content = trim_copy(text);

    if (!enable_thinking) {
        out.reasoning_content.clear();
    }
    return out;
}
