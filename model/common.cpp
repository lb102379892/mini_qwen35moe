#include <string>
#include <regex>
#include "common.h"

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
