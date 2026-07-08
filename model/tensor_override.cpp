#include "model/tensor_override.h"

#include <cstdio>
#include <stdexcept>

namespace {

std::string trim_copy(const std::string& value) {
    const size_t start = value.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return {};
    }
    const size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(start, end - start + 1);
}

}  // namespace

bool parse_override_device(const std::string& device_str, WeightDevice* out, std::string* error_out) {
    if (out == nullptr) {
        return false;
    }
    const std::string device = trim_copy(device_str);
    if (device.empty()) {
        if (error_out) {
            *error_out = "override device is empty";
        }
        return false;
    }
    if (device == "CPU" || device == "cpu") {
        *out = WeightDevice::CPU;
        return true;
    }
    if (device == "GPU" || device == "gpu" ||
        device.rfind("CUDA", 0) == 0 || device.rfind("cuda", 0) == 0) {
        *out = WeightDevice::GPU;
        return true;
    }
    if (error_out) {
        *error_out = "unsupported override device '" + device + "' (use CPU or GPU/CUDA0)";
    }
    return false;
}

void TensorOverrideRules::clear() {
    rules_.clear();
}

bool TensorOverrideRules::add_rule(const std::string& spec, std::string* error_out) {
    const size_t eq = spec.rfind('=');
    if (eq == std::string::npos || eq == 0 || eq + 1 >= spec.size()) {
        if (error_out) {
            *error_out = "invalid override spec '" + spec + "' (expected pattern=DEVICE)";
        }
        return false;
    }

    Rule rule;
    rule.pattern_str = trim_copy(spec.substr(0, eq));
    rule.device_str = trim_copy(spec.substr(eq + 1));
    if (rule.pattern_str.empty()) {
        if (error_out) {
            *error_out = "override regex pattern is empty";
        }
        return false;
    }
    if (!parse_override_device(rule.device_str, &rule.device, error_out)) {
        return false;
    }

    try {
        rule.pattern = std::regex(rule.pattern_str);
    } catch (const std::regex_error& ex) {
        if (error_out) {
            *error_out = std::string("invalid override regex '") + rule.pattern_str + "': " + ex.what();
        }
        return false;
    }

    rules_.push_back(std::move(rule));
    return true;
}

WeightDevice TensorOverrideRules::resolve(const std::string& tensor_name, WeightDevice default_device) const {
    WeightDevice resolved = default_device;
    for (const Rule& rule : rules_) {
        if (std::regex_search(tensor_name, rule.pattern)) {
            resolved = rule.device;
        }
    }
    return resolved;
}

void TensorOverrideRules::log_rules() const {
    if (rules_.empty()) {
        return;
    }
    std::fprintf(stderr, "[Loader] tensor override rules (%zu):\n", rules_.size());
    for (const Rule& rule : rules_) {
        std::fprintf(stderr, "  %s=%s\n", rule.pattern_str.c_str(), rule.device_str.c_str());
    }
}
