#!/usr/bin/env bash
# tests/perf/run_bench_cpp.sh
#
# Matrix driver for the C++ qwen35moe_bench harness.
#
# For each parameter combination below, this script:
#   1. spawns one qwen35moe_bench process,
#   2. measures (per request) prefill / decode tok/s and (overall) wall tok/s,
#   3. appends a Markdown row to $OUT_FILE so you get a single comparable
#      table at the end.
#
# Every process loads the model fresh, so wall time is "model_load + warmup +
# runs". If you need fine-grained sweeps without paying model-load cost every
# time, edit `BENCH_MATRIX_ROWS` / `AUTO_GPU_LAYER_ROWS` (or the RANGE_* arrays)
# to sweep only the dimension you care about and keep the rest fixed.
#
# Required environment:
#   MODEL  - absolute path to a .gguf model file
#
# Optional environment (with defaults):
#   BIN              - path to qwen35moe_bench   (auto-detected)
#   OUT_FILE         - results.md path           (default: ./bench_results.md)
#   RUNS             - measured request count    (default: 6)
#   WARMUP           - warmup request count      (default: 2)
#   MAX_TOKENS       - generated tokens / req    (default: 128)
#   PROMPT           - user prompt text          (default: built-in)
#   PROMPT_REPEAT    - repeat factor for prompt  (default: 1)
#   CTX              - --ctx-size                (default: 4096)
#   THREADS          - --threads (CPU cases)     (default: nproc)
#   ONLY             - regex to include cases    (default: all)
#   SKIP             - regex to skip cases       (default: none)
#
# Example:
#   MODEL=/path/Q5_K_M.gguf RUNS=8 WARMUP=2 ./tests/perf/run_bench_cpp.sh
#

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

: "${MODEL:?MODEL=/path/to/model.gguf is required}"
if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: MODEL file not found: $MODEL" >&2
    exit 1
fi

: "${OUT_FILE:=$REPO_ROOT/bench_results.md}"
: "${RUNS:=6}"
: "${WARMUP:=2}"
: "${MAX_TOKENS:=4096}"
: "${PROMPT:=写个linux用c++语言实现的server示例，只要求代码。}"
: "${PROMPT_REPEAT:=1}"
: "${ONLY:=}"
: "${SKIP:=}"

# Locate the binary. The repo doesn't fix a build dir, so search the obvious
# places and fail with a clear message if we can't find it.
detect_bin() {
    if [[ -n "${BIN:-}" && -x "$BIN" ]]; then
        echo "$BIN"; return
    fi
    for candidate in \
        "$REPO_ROOT/build/qwen35moe_bench" \
        "$REPO_ROOT/build-cuda/qwen35moe_bench" \
        "$REPO_ROOT/cmake-build-debug/qwen35moe_bench" \
        "$REPO_ROOT/cmake-build-release/qwen35moe_bench" \
        "$REPO_ROOT/qwen35moe_bench" ; do
        if [[ -x "$candidate" ]]; then
            echo "$candidate"; return
        fi
    done
    echo ""; return
}
BIN="$(detect_bin)"
if [[ -z "$BIN" ]]; then
    echo "ERROR: qwen35moe_bench not found. Build it first, e.g.:" >&2
    echo "  cmake -B build -DQWEN35MOE_CUDA=ON && cmake --build build -j --target qwen35moe_bench" >&2
    echo "Or set BIN=/path/to/qwen35moe_bench." >&2
    exit 1
fi

echo "Bench harness:    $BIN"
echo "Model:            $MODEL"
echo "Results file:     $OUT_FILE"
echo "Per-case runs:    $RUNS (warmup=$WARMUP, max_tokens=$MAX_TOKENS)"
echo

# Start clean so the file's header is written by the first qwen35moe_bench run.
rm -f "$OUT_FILE"

# -----------------------------------------------------------------------------
# Parameter ranges — each swept flag and its allowed values.
# build_cases() validates every matrix row against these arrays before run.
# -----------------------------------------------------------------------------
RANGE_DEV_MODE=(cpu gpu auto)
RANGE_FLASH_ATTN=(0 1)          # 0=off, 1=--flash-attn
RANGE_PARALLEL=(1 4)
RANGE_BENCH_CONCURRENCY=(1 4)   # --bench-concurrency (usually matches --parallel)
RANGE_PAGED_KV=(0 1)            # 0=omit --paged-kv
RANGE_PAGED_KV_BLOCK=(16 32 64)
RANGE_N_BATCH=(512 1024)
RANGE_N_UBATCH=(512 1024)
RANGE_THREADS=(1 2 3 4 7 9 13 17)  # CPU threads; ignored for gpu/auto
RANGE_CTX_SIZE=(1024 2048 4096)
RANGE_GPU_ID=(1)
RANGE_GPU_LAYER=(2 4 6 8 10 20 30 999)

# Shared sampling defaults (identical across all rows).
BENCH_TEMP=0.01
BENCH_TOP_P=1
BENCH_TOP_K=1
DEFAULT_GPU_LAYER=6

# Matrix rows × dev_mode; labels are built by build_case_label() (not row numbers).
# Columns: flash_attn parallel bench_conc paged_kv paged_kv_block n_batch n_ubatch threads ctx_size
# Label tokens: dev, fa|nofa, parN, [concN], pkv|nopkv, [pkvbN], nbN, nubN, thrN, ctxN, [gidN], [glN]
# Use "-" for paged_kv_block when paged_kv=0 (case 15).
BENCH_MATRIX_ROWS=(
    "0 1 1 1 16 512 512 4 4096"
    "1 1 1 1 16 512 512 4 4096"
    "0 4 4 1 16 512 512 4 4096"   # parallel=4 + paged-kv (batched decode stress)
    "1 1 1 1 16 512 512 4 4096"
    "1 1 1 1 32 512 512 4 4096"
    "1 1 1 1 64 512 512 4 4096"
    "1 1 1 1 16 1024 512 4 4096"
    "1 1 1 1 16 1024 1024 4 4096"
    "1 1 1 1 16 512 512 17 4096"
    "1 1 1 1 16 512 512 13 4096"
    "1 1 1 1 16 512 512 9 4096"
    "1 1 1 1 16 512 512 7 4096"
    "1 1 1 1 16 512 512 4 4096"
    "1 1 1 1 16 512 512 3 4096"
    "1 1 1 1 16 512 512 2 4096"
    "1 1 1 1 16 512 512 1 4096"
    "1 1 1 1 16 512 512 4 2048"
    "1 1 1 1 16 512 512 4 1024"
    "1 1 1 0 - 512 512 4 1024"
)

# Extra auto gpu-layer sweep (other dims match matrix row 14 / ctx1024 baseline).
AUTO_GPU_LAYER_ROWS=(6 10 20 30 999)

validate_in_range() {
    local label="$1" param="$2" value="$3"
    shift 3
    local allowed v
    for v in "$@"; do
        if [[ "$v" == "$value" ]]; then
            return 0
        fi
    done
    echo "ERROR: $label: $param=$value not in [${*// /, }]" >&2
    return 1
}

# shellcheck disable=SC2034  # fields assigned for clarity / validation
build_extra_args() {
    local label="$1" dev_mode="$2"
    local flash parallel conc paged_kv paged_kv_block n_batch n_ubatch threads ctx_size gpu_layer

    flash="$3"
    parallel="$4"
    conc="$5"
    paged_kv="$6"
    paged_kv_block="$7"
    n_batch="$8"
    n_ubatch="$9"
    threads="${10}"
    ctx_size="${11}"
    gpu_layer="${12:-$DEFAULT_GPU_LAYER}"

    validate_in_range "$label" flash_attn "$flash" "${RANGE_FLASH_ATTN[@]}" || return 1
    validate_in_range "$label" parallel "$parallel" "${RANGE_PARALLEL[@]}" || return 1
    validate_in_range "$label" bench_concurrency "$conc" "${RANGE_BENCH_CONCURRENCY[@]}" || return 1
    validate_in_range "$label" paged_kv "$paged_kv" "${RANGE_PAGED_KV[@]}" || return 1
    if [[ "$paged_kv" == 1 ]]; then
        validate_in_range "$label" paged_kv_block "$paged_kv_block" "${RANGE_PAGED_KV_BLOCK[@]}" || return 1
    elif [[ "$paged_kv_block" != "-" ]]; then
        echo "ERROR: $label: paged_kv_block must be '-' when paged_kv=0" >&2
        return 1
    fi
    validate_in_range "$label" n_batch "$n_batch" "${RANGE_N_BATCH[@]}" || return 1
    validate_in_range "$label" n_ubatch "$n_ubatch" "${RANGE_N_UBATCH[@]}" || return 1
    validate_in_range "$label" threads "$threads" "${RANGE_THREADS[@]}" || return 1
    validate_in_range "$label" ctx_size "$ctx_size" "${RANGE_CTX_SIZE[@]}" || return 1
    validate_in_range "$label" dev_mode "$dev_mode" "${RANGE_DEV_MODE[@]}" || return 1

    local extra="--dev-mode $dev_mode --temp $BENCH_TEMP --top-p $BENCH_TOP_P --top-k $BENCH_TOP_K"
    extra+=" --ctx-size $ctx_size --threads $threads --n-batch $n_batch --n-ubatch $n_ubatch"
    extra+=" --parallel $parallel"
    if [[ "$paged_kv" == 1 ]]; then
        extra+=" --paged-kv --paged-kv-block $paged_kv_block"
    fi
    if [[ "$flash" == 1 ]]; then
        extra+=" --flash-attn"
    fi
    if [[ "$dev_mode" == gpu || "$dev_mode" == auto ]]; then
        local gpu_id="${RANGE_GPU_ID[0]}"
        validate_in_range "$label" gpu_id "$gpu_id" "${RANGE_GPU_ID[@]}" || return 1
        extra+=" --gpu-id $gpu_id"
    fi
    if [[ "$dev_mode" == auto ]]; then
        validate_in_range "$label" gpu_layer "$gpu_layer" "${RANGE_GPU_LAYER[@]}" || return 1
        extra+=" --gpu-layer $gpu_layer"
    fi

    echo "$extra"
}

# Build a stable bench row label from abbreviated parameter names + values.
# Example: cpu_fa_par1_pkv_pkvb16_nb512_nub512_thr4_ctx4096_gid1_gl600
build_case_label() {
    local dev_mode="$1"
    local flash="$2"
    local parallel="$3"
    local conc="$4"
    local paged_kv="$5"
    local paged_kv_block="$6"
    local n_batch="$7"
    local n_ubatch="$8"
    local threads="$9"
    local ctx_size="${10}"
    local gpu_layer="${11:-}"

    local -a parts=("$dev_mode")
    if [[ "$flash" == 1 ]]; then
        parts+=(fa)
    else
        parts+=(nofa)
    fi
    parts+=("par${parallel}")
    if [[ "$conc" != "$parallel" ]]; then
        parts+=("conc${conc}")
    fi
    if [[ "$paged_kv" == 1 ]]; then
        parts+=(pkv "pkvb${paged_kv_block}")
    else
        parts+=(nopkv)
    fi
    parts+=("nb${n_batch}" "nub${n_ubatch}" "thr${threads}" "ctx${ctx_size}")
    if [[ "$dev_mode" == gpu || "$dev_mode" == auto ]]; then
        parts+=("gid${RANGE_GPU_ID[0]}")
    fi
    if [[ "$dev_mode" == auto ]]; then
        parts+=("gl${gpu_layer:-$DEFAULT_GPU_LAYER}")
    fi
    local IFS=_
    echo "${parts[*]}"
}

build_cases() {
    CASES=()
    local dev_mode row label extra conc
    local flash parallel paged_kv paged_kv_block n_batch n_ubatch threads ctx_size gpu_layer

    for dev_mode in "${RANGE_DEV_MODE[@]}"; do
        for row in "${BENCH_MATRIX_ROWS[@]}"; do
            # shellcheck disable=SC2086
            read -r flash parallel conc paged_kv paged_kv_block n_batch n_ubatch threads ctx_size <<<"$row"
            label="$(build_case_label "$dev_mode" \
                "$flash" "$parallel" "$conc" "$paged_kv" "$paged_kv_block" \
                "$n_batch" "$n_ubatch" "$threads" "$ctx_size")"
            extra="$(build_extra_args "$label" "$dev_mode" \
                "$flash" "$parallel" "$conc" "$paged_kv" "$paged_kv_block" \
                "$n_batch" "$n_ubatch" "$threads" "$ctx_size")" || return 1
            CASES+=("$label | $extra | $conc")
        done
    done

    # gpu-layer sweep: same dims as matrix row 14 (ctx 1024), varying gl only.
    row="${BENCH_MATRIX_ROWS[13]}"
    for gpu_layer in "${AUTO_GPU_LAYER_ROWS[@]}"; do
        # shellcheck disable=SC2086
        read -r flash parallel conc paged_kv paged_kv_block n_batch n_ubatch threads ctx_size <<<"$row"
        label="$(build_case_label auto \
            "$flash" "$parallel" "$conc" "$paged_kv" "$paged_kv_block" \
            "$n_batch" "$n_ubatch" "$threads" "$ctx_size" "$gpu_layer")"
        extra="$(build_extra_args "$label" auto \
            "$flash" "$parallel" "$conc" "$paged_kv" "$paged_kv_block" \
            "$n_batch" "$n_ubatch" "$threads" "$ctx_size" "$gpu_layer")" || return 1
        CASES+=("$label | $extra | $conc")
    done
}

build_cases || exit 1

run_case() {
    local label="$1"
    local extra="$2"
    local conc="$3"

    if [[ -n "$ONLY" && ! "$label" =~ $ONLY ]]; then
        echo "  SKIP $label (does not match ONLY=$ONLY)"
        return 0
    fi
    if [[ -n "$SKIP" && "$label" =~ $SKIP ]]; then
        echo "  SKIP $label (matches SKIP=$SKIP)"
        return 0
    fi

    echo "==> $label"
    echo "    args: $extra (conc=$conc)"

    # Intentionally word-split $extra so individual flags become argv entries.
    # The whole point of "extra args" is to be a shell-style flag list.
    # shellcheck disable=SC2086
    "$BIN" \
        --model "$MODEL" \
        --bench-runs "$RUNS" \
        --bench-warmup "$WARMUP" \
        --bench-concurrency "$conc" \
        --bench-max-tokens "$MAX_TOKENS" \
        --bench-prompt "$PROMPT" \
        --bench-prompt-repeat "$PROMPT_REPEAT" \
        --bench-label "$label" \
        --bench-output "$OUT_FILE" \
        $extra \
        || { echo "  FAILED: $label" >&2; return 1; }
    echo
}

failures=0
for raw in "${CASES[@]}"; do
    # Split "label | extra | conc" on '|'; trim surrounding spaces.
    label="$(echo "$raw"  | awk -F'|' '{print $1}' | xargs)"
    extra="$(echo "$raw"  | awk -F'|' '{print $2}' | sed -e 's/^ *//' -e 's/ *$//')"
    conc="$(echo  "$raw"  | awk -F'|' '{print $3}' | xargs)"
    if ! run_case "$label" "$extra" "$conc"; then
        failures=$((failures + 1))
    fi
done

echo
echo "Done. Results table: $OUT_FILE"
if (( failures > 0 )); then
    echo "$failures case(s) failed." >&2
    exit 1
fi
