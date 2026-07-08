#!/usr/bin/env bash
# tests/functional/test_matrix.sh — end-to-end CLI parameter correctness.
#
# For each row in the matrix below:
#   1. Launch test_qwen35moe with a unique --port and the row's extra args.
#   2. Wait for the HTTP server to come up.
#   3. Optionally send a deterministic greedy probe prompt and capture the
#      assistant text. CPU 35B generation is very slow, so CPU rows default to
#      start-only checks unless RUN_CPU_GENERATION=1 is set.
#   4. Send SIGTERM, wait, harvest stderr.
#   5. Assert: no runtime errors; response non-empty; the stderr contains
#      every required marker for this configuration; absent markers (if any)
#      are not present.
#
# A failure in any row sets the suite exit code to 1; failures are listed
# at the end. Each row's stderr log stays in $TEST_TMPDIR for inspection.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../lib.sh"

# Default probe prompt. We intentionally do NOT assert exact content by
# default: Qwen3.5 can emit a <think> block even when enable_thinking=false, and
# quantization / sampler changes can alter wording. Functional tests are about
# CLI wiring and server/runtime health, not model intelligence.
PROBE_PROMPT="${PROBE_PROMPT:-Say hello in one short sentence.}"
PROBE_MAX_TOKENS="${PROBE_MAX_TOKENS:-8}"
PROBE_NEEDLE="${PROBE_NEEDLE:-}"       # optional: set to enforce semantic text
RUN_CPU_GENERATION="${RUN_CPU_GENERATION:-0}"

# port_for <n>  -> BASE_PORT + n
port_for() { echo $(( BASE_PORT + $1 )); }

# ── run_case ──────────────────────────────────────────────────────────────────
#
# run_case <name> <extra_args> <required_logs> <absent_log> [<env>] [<probe_mode>]
#
# - <extra_args>     space-separated CLI args to append (in addition to those
#                    set by start_server: --model, --host, --port, --threads,
#                    --ctx-size, --seed). Pass "" for none.
# - <required_logs>  comma-separated list of substrings; the test passes if
#                    ANY ONE of them appears in stderr. Use "" to skip.
# - <absent_log>     substring that must NOT appear in stderr. Use "" to skip.
# - <env>            optional VAR=val pairs prefixed via env(1).
# - <probe_mode>     "probe" (default) sends one chat request and requires a
#                    non-empty response; "start-only" only verifies boot/logs.

CASE_INDEX=0
run_case() {
    local name="$1"
    local extra_args="$2"
    local required_logs="$3"
    local absent_log="$4"
    local env_pairs="${5:-}"
    local probe_mode="${6:-probe}"

    CASE_INDEX=$((CASE_INDEX + 1))
    local port; port=$(port_for "$CASE_INDEX")
    local log_file="$TEST_TMPDIR/${CASE_INDEX}_${name}.log"

    section "Case $CASE_INDEX: $name (port=$port)"
    log "  args: $extra_args"
    [[ -n "$env_pairs" ]] && log "  env:  $env_pairs"

    # Expand extra_args as separate words; we accept simple " "-separated values.
    # Quoting compound values is not needed for the CLI surface we exercise.
    # shellcheck disable=SC2206
    local args_array=( $extra_args )

    local pid
    if [[ -n "$env_pairs" ]]; then
        # shellcheck disable=SC2206
        local env_array=( $env_pairs )
        pid=$(BIN="$BIN" MODEL="$MODEL" THREADS="$THREADS" CTX="$CTX" env "${env_array[@]}" bash -c '
            "$BIN" --model "$MODEL" --host 127.0.0.1 --port "'"$port"'" \
                   --threads "$THREADS" --ctx-size "$CTX" --seed 42 "$@" \
                   > "'"$log_file"'" 2>&1 &
            echo $!
        ' _ "${args_array[@]}")
    else
        pid=$(start_server "$port" "$log_file" "${args_array[@]}")
    fi

    if ! wait_for_server "$port" "$log_file"; then
        fail "$name: server failed to start within ${SERVER_READY_TIMEOUT_SEC}s"
        log "  last 30 lines of $log_file:"
        tail -30 "$log_file" 2>/dev/null | sed 's/^/    /' >&2
        stop_server "$pid"
        return 1
    fi

    local resp=""
    if [[ "$probe_mode" == "probe" ]]; then
        resp=$(chat_complete "$port" "$PROBE_PROMPT" 0.0 1 "$PROBE_MAX_TOKENS") || resp=""
    fi

    stop_server "$pid"

    assert_no_runtime_errors "$name" "$log_file"
    if [[ "$probe_mode" == "probe" ]]; then
        assert_response_nonempty "$name" "$resp"
    else
        pass "$name: start-only check (chat probe skipped)"
    fi
    if [[ -n "$resp" && -n "$PROBE_NEEDLE" ]]; then
        assert_response_contains "$name" "$resp" "$PROBE_NEEDLE"
    fi
    if [[ -n "$required_logs" ]]; then
        # shellcheck disable=SC2206
        local needles=( ${required_logs//,/ } )
        assert_log_contains "$name" "$log_file" "${needles[@]}"
    fi
    if [[ -n "$absent_log" ]]; then
        assert_log_absent "$name" "$log_file" "$absent_log"
    fi
}

# ──────────────────────────────────────────────────────────────────────────────
# The matrix.
#
# Per row: (name, extra args, required-any-of log substrings, absent log).
# Adjust the matrix to your environment — disable GPU/AUTO rows if you built
# without QWEN35MOE_CUDA, etc. via SKIP_* env vars below.
# ──────────────────────────────────────────────────────────────────────────────

: "${SKIP_GPU:=0}"        # set to 1 if the build/host has no CUDA
: "${SKIP_AUTO:=$SKIP_GPU}"  # AUTO needs both backends

# ── Smoke: minimal CPU run with chat template + greedy sampling ──────────────
run_case "smoke_cpu_default" \
    "--dev-mode cpu" \
    "Loading model:,Starting HTTP server" \
    "Segmentation fault" \
    "" \
    "$([[ "$RUN_CPU_GENERATION" == "1" ]] && echo probe || echo start-only)"

# ── Dev modes ────────────────────────────────────────────────────────────────
if [[ "$SKIP_GPU" != "1" ]]; then
    run_case "dev_mode_gpu" \
        "--dev-mode gpu --parallel 1" \
        "Starting HTTP server" \
        "mixed GPU/CPU detected"
fi

if [[ "$SKIP_AUTO" != "1" ]]; then
    # On a small model that fits entirely on GPU, AUTO may never print
    # "mixed GPU/CPU detected" — that's logged informationally by lib.sh.
    run_case "dev_mode_auto" \
        "--dev-mode auto --parallel 1 --gpu-layer 600" \
        "Starting HTTP server" \
        ""
fi

# ── Parallel / continuous batching ───────────────────────────────────────────
run_case "parallel_4_cpu" \
    "--dev-mode cpu --parallel 4" \
    "Starting HTTP server" \
    "" \
    "" \
    "$([[ "$RUN_CPU_GENERATION" == "1" ]] && echo probe || echo start-only)"

if [[ "$SKIP_GPU" != "1" ]]; then
    run_case "parallel_4_gpu" \
        "--dev-mode gpu --parallel 4" \
        "Starting HTTP server" \
        ""
fi

# ── flash-attn ───────────────────────────────────────────────────────────────
if [[ "$SKIP_GPU" != "1" ]]; then
    run_case "flash_attn_gpu" \
        "--dev-mode gpu --flash-attn --parallel 1" \
        "Starting HTTP server" \
        ""
fi

# ── paged-kv ─────────────────────────────────────────────────────────────────
if [[ "$SKIP_GPU" != "1" ]]; then
    # GPU + flash-attn + paged-kv: the only configuration where the fused-FA
    # decode path is actually exercised. Look for the activation log line
    # (only printed when QWEN35MOE_PAGED_FUSED_DIAG / DEV_CHECK enables diag).
    run_case "paged_kv_gpu_fa" \
        "--dev-mode gpu --flash-attn --paged-kv --paged-kv-block 16 --parallel 1" \
        "paged-kv] enabled" \
        "" \
        "QWEN35MOE_PAGED_FUSED_DIAG=1"
fi

# paged-kv requested under AUTO mode: must auto-disable (see graph.cpp).
if [[ "$SKIP_AUTO" != "1" ]]; then
    run_case "paged_kv_auto_autodisable" \
        "--dev-mode auto --paged-kv --gpu-layer 600 --parallel 1" \
        "disabled automatically: model is running in AUTO_MODE" \
        "paged-kv] enabled"

    # The escape hatch: force-enable under AUTO and verify the warning fires.
    run_case "paged_kv_auto_force_override" \
        "--dev-mode auto --paged-kv --gpu-layer 600 --parallel 1" \
        "force-enabled under AUTO_MODE" \
        "disabled automatically: model is running in AUTO_MODE" \
        "QWEN35MOE_PAGED_KV_FORCE_MIXED=1"
fi

# ── Batch / micro-batch ──────────────────────────────────────────────────────
run_case "small_ubatch" \
    "--dev-mode cpu --n-batch 128 --n-ubatch 32" \
    "Starting HTTP server" \
    "" \
    "" \
    "$([[ "$RUN_CPU_GENERATION" == "1" ]] && echo probe || echo start-only)"

# ── Sampling: temperature path ───────────────────────────────────────────────
# Probes the temperature/top-k/top-p code path; we don't check determinism
# (it's stochastic) but do require a non-empty response and no errors.
run_case "sampling_temperature" \
    "--dev-mode cpu --parallel 1" \
    "Starting HTTP server" \
    "" \
    "" \
    "$([[ "$RUN_CPU_GENERATION" == "1" ]] && echo probe || echo start-only)"

# ── Mixed-mode DN decode batched (the second-priority fix) ───────────────────
if [[ "$SKIP_AUTO" != "1" ]]; then
    run_case "mixed_batched_eager_default" \
        "--dev-mode auto --parallel 4 --gpu-layer 600" \
        "batched eager scheduler path,Starting HTTP server" \
        "sequential per-token segmented path"

    run_case "mixed_batched_sequential_opt_in" \
        "--dev-mode auto --parallel 4 --gpu-layer 600" \
        "sequential per-token segmented path" \
        "batched eager scheduler path" \
        "QWEN35MOE_MIXED_BATCHED_SEQUENTIAL=1"
fi

print_summary
