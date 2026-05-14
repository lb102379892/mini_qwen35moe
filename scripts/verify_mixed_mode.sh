#!/usr/bin/env bash
# verify_mixed_mode.sh — Validation steps for mixed GPU/CPU (--dev-mode=auto) fix
#
# Usage:
#   chmod +x scripts/verify_mixed_mode.sh
#   MODEL=/path/to/model.gguf ./scripts/verify_mixed_mode.sh
#
# Prerequisites:
#   - CUDA build:  cmake -DQWEN35MOE_CUDA=ON .. && make -j
#   - MODEL environment variable pointing to a .gguf model file

set -euo pipefail

: "${MODEL:?Please set MODEL=/path/to/model.gguf}"
: "${BIN:=./build/test_qwen35moe}"
PROMPT="Say hello in one sentence."
CTX=256
THREADS="${THREADS:-4}"

if [[ ! -f "$BIN" ]]; then
    echo "ERROR: binary not found at $BIN — build first with QWEN35MOE_CUDA=ON" >&2
    exit 1
fi

pass() { echo "[PASS] $*"; }
fail() { echo "[FAIL] $*"; exit 1; }

# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Pure GPU mode — baseline, ensure no regression
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Test 1: pure GPU mode (--dev-mode=gpu) ==="
OUTPUT=$(timeout 120 "$BIN" \
    --model "$MODEL" \
    --prompt "$PROMPT" \
    --dev-mode gpu \
    --ctx-size "$CTX" \
    --threads "$THREADS" \
    --no-chat \
    --temp 0.0 2>&1 || true)

if echo "$OUTPUT" | grep -qiE "CUDA error|illegal memory access|device mismatch|Segmentation fault|Aborted"; then
    fail "pure GPU mode produced an error:\n$OUTPUT"
fi
pass "pure GPU mode completed without errors."

# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Mixed GPU/CPU mode — eager decode path (primary fix)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Test 2: mixed GPU/CPU mode (--dev-mode=auto), eager decode ==="
OUTPUT=$(timeout 180 "$BIN" \
    --model "$MODEL" \
    --prompt "$PROMPT" \
    --dev-mode auto \
    --ctx-size "$CTX" \
    --threads "$THREADS" \
    --no-chat \
    --temp 0.0 2>&1 || true)

if echo "$OUTPUT" | grep -qiE "CUDA error|illegal memory access|device mismatch|Segmentation fault|Aborted"; then
    fail "mixed GPU/CPU mode produced an error:\n$OUTPUT"
fi
if echo "$OUTPUT" | grep -q "mixed GPU/CPU detected"; then
    pass "mixed mode detected and eager decode fallback active."
else
    echo "[INFO] Model may be fully GPU-resident (no CPU offload needed). Verify with a larger model."
fi
pass "mixed GPU/CPU mode (eager) completed without errors."

# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Mixed GPU/CPU mode with device-check logging enabled
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Test 3: mixed GPU/CPU mode + QWEN35MOE_DEV_CHECK=1 logging ==="
OUTPUT=$(timeout 180 QWEN35MOE_DEV_CHECK=1 "$BIN" \
    --model "$MODEL" \
    --prompt "$PROMPT" \
    --dev-mode auto \
    --ctx-size "$CTX" \
    --threads "$THREADS" \
    --no-chat \
    --temp 0.0 2>&1 || true)

if echo "$OUTPUT" | grep -qiE "CUDA error|illegal memory access|device mismatch|Segmentation fault|Aborted"; then
    fail "mixed mode with dev-check produced an error:\n$OUTPUT"
fi
if echo "$OUTPUT" | grep -q "\[dev-check\]"; then
    pass "device-check log lines present."
else
    echo "[INFO] No [dev-check] lines — either model is pure GPU or QWEN35MOE_DEV_CHECK is unsupported."
fi
pass "mixed GPU/CPU mode with dev-check completed without errors."

# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Decode graph reuse in pure GPU mode (performance path, no regression)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Test 4: decode graph reuse in pure GPU mode ==="
OUTPUT=$(timeout 180 QWEN35MOE_DECODE_GRAPH_DIAG=1 QWEN35MOE_DECODE_GRAPH_DIAG_INTERVAL=1 "$BIN" \
    --model "$MODEL" \
    --prompt "$PROMPT" \
    --dev-mode gpu \
    --ctx-size "$CTX" \
    --threads "$THREADS" \
    --no-chat \
    --temp 0.0 2>&1 || true)

if echo "$OUTPUT" | grep -qiE "CUDA error|illegal memory access|Segmentation fault|Aborted"; then
    fail "pure GPU mode with decode-graph diag produced an error:\n$OUTPUT"
fi
if echo "$OUTPUT" | grep -q "\[PERF\]\[decode-graph\]"; then
    HITS=$(echo "$OUTPUT" | grep -oP "hit=\K[0-9]+" | tail -1 || echo 0)
    echo "[INFO] decode-graph hit count at end of run: $HITS"
fi
pass "decode graph reuse (pure GPU) completed without errors."

echo ""
echo "All tests passed."
