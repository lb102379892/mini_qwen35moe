#include "model/dev_defaults.h"

#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

void set_env_if_unset(const char* name, const char* value) {
    const char* existing = std::getenv(name);
    if (existing != nullptr && existing[0] != '\0') {
        return;
    }
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 0);
#endif
}

const char* dev_mode_label(DevMode mode) {
    switch (mode) {
        case DevMode::CPU_MODE:
            return "cpu";
        case DevMode::GPU_MODE:
            return "gpu";
        case DevMode::AUTO_MODE:
            return "auto";
    }
    return "cpu";
}

int default_cpu_threads() {
    const unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) {
        return 4;
    }
    return static_cast<int>(hw);
}

}  // namespace

void apply_dev_mode_env_defaults(DevMode mode, const CParam& param) {
    (void)param;
    std::vector<const char*> applied;

    auto mark = [&](const char* name, const char* value) {
        const char* before = std::getenv(name);
        set_env_if_unset(name, value);
        if (before == nullptr || before[0] == '\0') {
            applied.push_back(name);
        }
    };

    switch (mode) {
        case DevMode::CPU_MODE:
            // CPU decode is already segmented-cache heavy; LOCK_CTX pins a
            // context-wide attention graph and makes short-prompt prefill very
            // slow. Fine decode buckets + prefill_token_bucket_capacity() are
            // the better CPU default.
            break;
        case DevMode::GPU_MODE:
            // Log once on first DeltaNet fused decode hit so GPU runs can confirm
            // the fast path is active (logging only; path is CUDA-gated separately).
            mark("QWEN35MOE_DN_DECODE_VERIFY", "1");
            mark("QWEN35MOE_DECODE_KV_LOCK_CTX", "1");
            break;
        case DevMode::AUTO_MODE:
            mark("QWEN35MOE_DECODE_KV_LOCK_CTX", "1");
            break;
    }

    if (!applied.empty()) {
        std::fprintf(stderr, "[dev-defaults] mode=%s env:", dev_mode_label(mode));
        for (const char* name : applied) {
            std::fprintf(stderr, " %s", name);
        }
        std::fprintf(stderr, "\n");
    }
}

void apply_dev_mode_cparam_defaults(CParam& param, const CParamUserFlags& flags) {
    const char* mode = dev_mode_label(param.dev_mode);
    std::vector<const char*> notes;

    auto note = [&](const char* text) {
        notes.push_back(text);
    };

    switch (param.dev_mode) {
        case DevMode::CPU_MODE:
            if (!flags.flash_attention && !param.flash_attention) {
                note("flash_attn=off");
            }
            if (!flags.enable_paged_kv) {
                if (param.enable_paged_kv) {
                    note("paged_kv=off");
                }
                param.enable_paged_kv = false;
            }
            if (!flags.max_sequences) {
                param.max_sequences = 1;
                note("parallel=1");
            }
            if (!flags.n_threads && param.n_threads == 4) {
                param.n_threads = default_cpu_threads();
                note("threads=hw");
            }
            break;

        case DevMode::GPU_MODE:
            if (!flags.flash_attention && !param.flash_attention) {
                param.flash_attention = true;
                note("flash_attn=on");
            }
            if (!flags.enable_paged_kv && !param.enable_paged_kv && param.max_sequences > 1) {
                param.enable_paged_kv = true;
                note("paged_kv=on(parallel>1)");
            }
            if (!flags.top_k && param.top_k == 50) {
                param.top_k = 40;
                note("top_k=40");
            }
            if (!flags.top_p && param.top_p == 0.9f) {
                param.top_p = 0.95f;
                note("top_p=0.95");
            }
            break;

        case DevMode::AUTO_MODE:
            if (!flags.flash_attention && !param.flash_attention) {
                param.flash_attention = true;
                note("flash_attn=on");
            }
            if (!flags.enable_paged_kv && param.enable_paged_kv) {
                param.enable_paged_kv = false;
                note("paged_kv=off");
            }
            if (!flags.max_sequences) {
                if (!param.tensor_overrides.empty()) {
                    param.max_sequences = 1;
                    note("parallel=1(split override)");
                } else if (param.max_sequences == 4) {
                    param.max_sequences = 1;
                    note("parallel=1");
                }
            }
            if (!flags.n_threads && param.n_threads == 4) {
                param.n_threads = default_cpu_threads();
                note("threads=hw");
            }
            break;
    }

    if (!notes.empty()) {
        std::fprintf(stderr, "[dev-defaults] mode=%s cparam:", mode);
        for (const char* text : notes) {
            std::fprintf(stderr, " %s", text);
        }
        std::fprintf(stderr, "\n");
    }
}
