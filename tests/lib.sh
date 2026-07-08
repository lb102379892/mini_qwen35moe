#!/usr/bin/env bash
# tests/lib.sh — shared helpers for end-to-end test_qwen35moe testing.
#
# Source from any test script:
#   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#   source "$SCRIPT_DIR/../lib.sh"

set -u  # do NOT set -e here — we want callers to inspect exit codes

# ── Configuration via env vars ────────────────────────────────────────────────
: "${MODEL:?ERROR: set MODEL=/path/to/model.gguf}"
: "${BIN:=./build/test_qwen35moe}"
: "${BASE_PORT:=7700}"     # tests pick BASE_PORT + offset to avoid clashes
: "${THREADS:=4}"
: "${CTX:=512}"
: "${SERVER_READY_TIMEOUT_SEC:=600}"   # 35B + CUDA cold load can take minutes
: "${HTTP_TIMEOUT_SEC:=300}"
: "${TEST_TMPDIR:=$(mktemp -d -t qwen35moe-tests.XXXXXX)}"

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: binary not found / not executable: $BIN" >&2
    echo "Build it first (e.g. cmake .. && make -j)." >&2
    exit 1
fi
if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: model file not found: $MODEL" >&2
    exit 1
fi

# Per-suite counters
PASS_COUNT=0
FAIL_COUNT=0
FAIL_LOG=()

log()  { printf '%s\n' "$*" >&2; }
pass() { printf '[PASS] %s\n' "$*" >&2; PASS_COUNT=$((PASS_COUNT + 1)); }
fail() {
    printf '[FAIL] %s\n' "$*" >&2
    FAIL_COUNT=$((FAIL_COUNT + 1))
    FAIL_LOG+=("$*")
}
section() {
    printf '\n=== %s ===\n' "$*" >&2
}

# ── Server lifecycle ──────────────────────────────────────────────────────────
#
# start_server <port> <log_file> [extra args...]
#   Spawns BIN with the given args in background. Returns the PID.
#   The server writes stderr (where all interesting logs go) to <log_file>.
#
# wait_for_server <port> <log_file>
#   Polls /health until 200 or until SERVER_READY_TIMEOUT_SEC elapses.
#   Falls back to checking stderr for "Starting HTTP server" if /health
#   is not yet wired.
#
# stop_server <pid>
#   Sends SIGTERM, then SIGKILL after a few seconds.

start_server() {
    local port="$1"; shift
    local log_file="$1"; shift
    local extra_args=( "$@" )

    : > "$log_file"

    # All test runs use deterministic seed and greedy/near-greedy sampling so
    # output is comparable across runs. Tests override --temp / --top-k as needed.
    # We capture both stdout (where [Timing] / [batch] lines go) and stderr
    # (where [paged-kv], [mixed-mode] etc. logs go) into the same file. The
    # functional matrix asserts on stderr keywords; the perf benchmark parses
    # the stdout [Timing] lines. Combining them simplifies both consumers.
    "$BIN" \
        --model "$MODEL" \
        --host 127.0.0.1 \
        --port "$port" \
        --threads "$THREADS" \
        --ctx-size "$CTX" \
        --seed 42 \
        "${extra_args[@]}" \
        > "$log_file" 2>&1 &

    local pid=$!
    echo "$pid"
}

wait_for_server() {
    local port="$1"
    local log_file="$2"
    local deadline=$(( $(date +%s) + SERVER_READY_TIMEOUT_SEC ))

    while (( $(date +%s) < deadline )); do
        # Prefer /health if it responds; fall back to checking the stderr banner.
        if curl -fsS --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            return 0
        fi
        if grep -q "Starting HTTP server" "$log_file" 2>/dev/null; then
            # Banner printed but /health may still need a second; one more probe.
            sleep 1
            if curl -fsS --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
                return 0
            fi
            # Some early failures still print the banner — re-check the process.
        fi
        sleep 1
    done
    return 1
}

stop_server() {
    local pid="$1"
    if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    kill -TERM "$pid" 2>/dev/null || true
    local deadline=$(( $(date +%s) + 8 ))
    while kill -0 "$pid" 2>/dev/null && (( $(date +%s) < deadline )); do
        sleep 0.2
    done
    if kill -0 "$pid" 2>/dev/null; then
        kill -KILL "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
}

# ── HTTP probe ────────────────────────────────────────────────────────────────
#
# chat_complete <port> <user_text> [temperature] [top_k] [max_tokens]
#   Sends a non-streaming /v1/chat/completions request and prints the assistant
#   message text. Empty string on failure (caller should also check $?).
#
# Defaults are greedy (temp=0.0, top_k=1) so the output is deterministic and
# easy to assert on.

chat_complete() {
    local port="$1"
    local user_text="$2"
    local temperature="${3:-0.0}"
    local top_k="${4:-1}"
    local max_tokens="${5:-32}"

    local body
    body=$(cat <<EOF
{
  "messages":[{"role":"user","content":$(json_quote "$user_text")}],
  "temperature": ${temperature},
  "top_p": 1,
  "top_k": ${top_k},
  "max_tokens": ${max_tokens},
  "chat_template_kwargs": {"enable_thinking": false}
}
EOF
)

    local resp
    resp=$(curl -fsS --max-time "$HTTP_TIMEOUT_SEC" \
        -X POST "http://127.0.0.1:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$body" 2>/dev/null) || return 1

    # Minimal extractor for choices[0].message.content (server emits no nested
    # newlines that would break this).
    printf '%s' "$resp" \
        | sed -nE 's/.*"content"[[:space:]]*:[[:space:]]*"((\\.|[^"\\])*)".*/\1/p' \
        | head -1
}

json_quote() {
    # Minimal JSON string quoting. Good enough for short prompts we use here.
    local s="$1"
    s=${s//\\/\\\\}
    s=${s//\"/\\\"}
    s=${s//$'\n'/\\n}
    s=${s//$'\r'/\\r}
    s=${s//$'\t'/\\t}
    printf '"%s"' "$s"
}

# ── Assertions ────────────────────────────────────────────────────────────────
#
# assert_response_nonempty <name> <text>
# assert_response_contains <name> <text> <needle>
# assert_log_contains      <name> <log_file> <pattern...>   (any-of)
# assert_log_absent        <name> <log_file> <pattern>

assert_response_nonempty() {
    local name="$1" text="$2"
    if [[ -n "$text" ]]; then
        pass "$name: response is non-empty"
    else
        fail "$name: response is empty"
    fi
}

assert_response_contains() {
    local name="$1" text="$2" needle="$3"
    if [[ "$text" == *"$needle"* ]]; then
        pass "$name: response contains '$needle'"
    else
        fail "$name: response missing '$needle' (got: ${text:0:120}...)"
    fi
}

assert_log_contains() {
    local name="$1" log_file="$2"; shift 2
    local pat
    for pat in "$@"; do
        if grep -qF -- "$pat" "$log_file" 2>/dev/null; then
            pass "$name: log contains '$pat'"
            return 0
        fi
    done
    fail "$name: log missing all of: $*"
    return 1
}

assert_log_absent() {
    local name="$1" log_file="$2" pat="$3"
    if grep -qF -- "$pat" "$log_file" 2>/dev/null; then
        fail "$name: log unexpectedly contains '$pat'"
    else
        pass "$name: log does NOT contain '$pat'"
    fi
}

assert_no_runtime_errors() {
    local name="$1" log_file="$2"
    if grep -qiE "CUDA error|illegal memory access|device mismatch|Segmentation fault|Aborted|terminate called" "$log_file"; then
        fail "$name: runtime error in log:"
        grep -iE "CUDA error|illegal memory access|device mismatch|Segmentation fault|Aborted|terminate called" "$log_file" | sed 's/^/    /' >&2
        return 1
    fi
    pass "$name: no runtime errors"
}

# ── Summary ───────────────────────────────────────────────────────────────────
print_summary() {
    local total=$(( PASS_COUNT + FAIL_COUNT ))
    printf '\n────────────────────────────────────────\n' >&2
    printf 'Summary: %d/%d assertions passed\n' "$PASS_COUNT" "$total" >&2
    if (( FAIL_COUNT > 0 )); then
        printf '\nFailures:\n' >&2
        local f
        for f in "${FAIL_LOG[@]}"; do
            printf '  - %s\n' "$f" >&2
        done
        return 1
    fi
    return 0
}

# Make logs directory predictable but per-suite.
mkdir -p "$TEST_TMPDIR"
log "Test logs and artifacts will be written to: $TEST_TMPDIR"
