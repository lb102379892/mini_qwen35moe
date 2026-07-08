#!/usr/bin/env bash
# tests/perf/run_bench.sh — opinionated wrapper around bench.py.
#
# Runs the most informative comparison matrix for mini vs llama.cpp:
#   1. Single-stream GPU latency (decode-bound)
#   2. Concurrent (parallel=4) GPU throughput
#   3. Mixed GPU+CPU vs llama.cpp CPU offload (only mini side; llama row is
#      best-effort, using --split-mode layer + --n-gpu-layers)
#
# Each cell produces one row in the markdown table. JSON-per-line is also
# emitted to $OUT_DIR/<scenario>.jsonl for later analysis.
#
# Required env:
#   MODEL          — path to the .gguf model (same file for both backends)
#   LLAMA_CPP_BIN  — path to llama.cpp's llama-server binary
#
# Optional env:
#   BIN            — path to mini binary (default ./build/test_qwen35moe)
#   THREADS        — CPU threads (default 4)
#   CTX            — KV cache context length (default 2048)
#   MAX_TOKENS     — completion length (default 200)
#   RUNS           — measured requests per scenario (default 5)
#   OUT_DIR        — where to write logs+jsonl (default $PWD/bench-results)

set -u

: "${MODEL:?ERROR: set MODEL=/path/to/model.gguf}"
: "${LLAMA_CPP_BIN:?ERROR: set LLAMA_CPP_BIN=/path/to/llama-server}"
: "${BIN:=./build/test_qwen35moe}"
: "${THREADS:=4}"
: "${CTX:=2048}"
: "${MAX_TOKENS:=200}"
: "${RUNS:=5}"
: "${OUT_DIR:=$(pwd)/bench-results}"

mkdir -p "$OUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$SCRIPT_DIR/bench.py"

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: mini binary not executable: $BIN" >&2
    exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 is required to run bench.py" >&2
    exit 1
fi

run_scenario() {
    local name="$1"; shift
    echo
    echo "========================================================================"
    echo "Scenario: $name"
    echo "  args: $*"
    echo "  out:  $OUT_DIR/${name}.jsonl"
    echo "========================================================================"
    python3 "$BENCH" \
        --runs "$RUNS" \
        --max-tokens "$MAX_TOKENS" \
        --threads "$THREADS" \
        --ctx "$CTX" \
        --json-out "$OUT_DIR/${name}.jsonl" \
        "$@" \
        | tee "$OUT_DIR/${name}.md"
}

# ─── Scenario 1: single-stream GPU latency (the headline number for batch=1) ─
run_scenario "01_gpu_single" \
    --target both --concurrency 1 \
    --mini-args  "--dev-mode gpu --flash-attn --paged-kv --parallel 1" \
    --llama-args "--n-gpu-layers 999 -fa --parallel 1"

# ─── Scenario 2: concurrent GPU throughput (4 parallel chats, the headline ──
# ─── number for `--parallel 4`) ───────────────────────────────────────────────
run_scenario "02_gpu_parallel4" \
    --target both --concurrency 4 \
    --mini-args  "--dev-mode gpu --flash-attn --paged-kv --parallel 4" \
    --llama-args "--n-gpu-layers 999 -fa --parallel 4"

# ─── Scenario 3: mixed GPU+CPU (mini-only is meaningful — llama's row is a ───
# ─── best-effort comparison using --n-gpu-layers split) ──────────────────────
# Note: under --dev-mode auto, paged-kv auto-disables (correct behaviour);
# the warning is informational, not a failure.
run_scenario "03_mixed_gpu_cpu" \
    --target both --concurrency 4 \
    --mini-args  "--dev-mode auto --flash-attn --parallel 4 --gpu-layer 600" \
    --llama-args "--n-gpu-layers 24 -fa --parallel 4"

# ─── Scenario 4 (optional): pure-CPU floor, useful as a sanity check that ────
# ─── the mini CPU path is not regressed against llama.cpp CPU-only ────────────
if [[ "${INCLUDE_CPU_SCENARIO:-0}" == "1" ]]; then
    run_scenario "04_cpu_only" \
        --target both --concurrency 1 \
        --mini-args  "--dev-mode cpu --flash-attn --parallel 1" \
        --llama-args "--n-gpu-layers 0 -fa --parallel 1"
fi

echo
echo "Done. Per-scenario markdown tables live in $OUT_DIR/*.md;"
echo "raw per-request JSON in $OUT_DIR/*.jsonl."
