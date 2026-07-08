#include <string>
#include <regex>
#include <cstdio>
#include <cstdlib>
#include "common.h"
#include "ggml.h"

int extractNumber(const std::string& str) {
    std::regex pattern(R"(blk\.(\d+)\.)");
    std::smatch match;
    
    if (std::regex_search(str, match, pattern)) {
        return std::stoi(match[1]);
    }
    return -1; // 未找到数字
}

std::string getAfterNumber(const std::string& str) {
    std::regex pattern(R"(blk\.\d+(.*))");
    std::smatch match;
    
    if (std::regex_search(str, match, pattern) && match.size() > 1) {
        return match[1]; // 返回数字后面的部分
    }
    return "";
}

static void qwen_ggml_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void)user_data;
    if (level == GGML_LOG_LEVEL_DEBUG) {
        const char* enable = std::getenv("QWEN35MOE_GGML_DEBUG");
        if (!enable || enable[0] == '0') {
            return;
        }
    }
    fputs(text, stderr);
    fflush(stderr);
}

void init_ggml_logging() {
    static bool installed = false;
    if (installed) {
        return;
    }
    installed = true;
    ggml_log_set(qwen_ggml_log_callback, nullptr);
}
