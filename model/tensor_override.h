#pragma once

#include <regex>
#include <string>
#include <vector>

enum class WeightDevice {
    Default,
    CPU,
    GPU,
};

class TensorOverrideRules {
public:
    void clear();
    bool add_rule(const std::string& spec, std::string* error_out = nullptr);
    bool empty() const { return rules_.empty(); }
    size_t size() const { return rules_.size(); }

    // Last matching rule wins (llama.cpp --override-tensor semantics).
    WeightDevice resolve(const std::string& tensor_name, WeightDevice default_device) const;

    void log_rules() const;

private:
    struct Rule {
        std::string pattern_str;
        std::string device_str;
        WeightDevice device = WeightDevice::Default;
        std::regex pattern;
    };

    std::vector<Rule> rules_;
};

bool parse_override_device(const std::string& device_str, WeightDevice* out, std::string* error_out);
