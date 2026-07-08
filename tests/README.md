# Tests

End-to-end validation and benchmarking for `test_qwen35moe`.

## Layout

Three layers, picked according to what you want to learn:

- `lib.sh` + `functional/test_matrix.sh`
  Black-box correctness matrix. Each row launches a fresh
  `test_qwen35moe` HTTP server with one CLI combination, probes
  `/v1/chat/completions`, asserts on response + stderr log markers,
  kills the server. Best for "does flag X still wire through end-to-end?"

- `perf/run_bench.sh` (+ `perf/bench.py`)
  HTTP-driven throughput comparison against an external
  `llama-server` (llama.cpp) using the OpenAI-compatible
  `/v1/chat/completions` API. Best for apples-to-apples comparison
  with another inference engine.

- `perf/run_bench_cpp.sh` (+ `perf/bench_main.cpp` → `qwen35moe_bench`)
  **In-process C++ benchmark.** Calls `ChatEngine::run_complete`
  directly with the same `CParam` that the CLI would produce, so the
  whole `--dev-mode / --parallel / --flash-attn / --paged-kv /
  --n-batch / --n-ubatch / ...` flag set behaves 1:1 with the main
  binary but without any HTTP/JSON overhead in the measurement loop.
  Per-request prefill / decode tok/s come from `TimingStats` published
  by `ChatEngine` itself, so they're as accurate as the engine's own
  `[Timing]` log line. Best for **"how does flag X change t/s?"**

C++ unit tests are still useful for **data structures and algorithms** —
see `graph/paged_kv_cache_test.cpp` (registered with `ctest`). Add more
of those when you touch `simple_kv_cache`, `DeltaNetState`, or token
sampling math, not when you change CLI parsing or pipeline wiring.

## Prerequisites

Environment variables every script reads:

| Var                | Default                              | Required |
|--------------------|--------------------------------------|----------|
| `MODEL`            | (none)                               | yes      |
| `BIN`              | `./build/test_qwen35moe`             | no       |
| `LLAMA_CPP_BIN`    | (none — perf comparison skipped)     | only for perf |
| `BASE_PORT`        | `7700`                               | no       |
| `THREADS`          | `4`                                  | no       |
| `CTX`              | `512`                                | no       |

Build the binary once with the same CMake flags you use in production
(e.g. `-DQWEN35MOE_CUDA=ON`).

## Quick start

```bash
export MODEL=/home/xc/3rd/Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_M.gguf

# All correctness tests (server-mode, HTTP probes)
./tests/functional/test_matrix.sh

# In-process C++ throughput sweep across common flag combinations.
# Build qwen35moe_bench first (same CMake config as test_qwen35moe).
cmake -B build -DQWEN35MOE_CUDA=ON
cmake --build build -j --target qwen35moe_bench
./tests/perf/run_bench_cpp.sh
# -> writes one Markdown table to ./bench_results.md, one row per case.

# Optional: head-to-head HTTP comparison vs llama.cpp
export LLAMA_CPP_BIN=/path/to/llama-server
./tests/perf/run_bench.sh
```

Each script returns non-zero on failure and prints `[PASS]` / `[FAIL]`
markers so you can use it in CI.

### Reading the C++ bench output

Each row of `bench_results.md` is one process / one parameter set. The
columns are:

| column | meaning |
|--------|---------|
| `label` | name from `CASES` (or `--bench-label`) |
| `dev / parallel / flash / paged_kv / n_batch / n_ubatch` | the actual `CParam` used |
| `conc` | `--bench-concurrency` (in-flight requests) |
| `prompt_tok / gen_tok` | median prompt / generated token counts |
| `runs` | measured request count (warmup discarded) |
| `prefill tok/s med/p95` | per-request prefill throughput |
| `decode tok/s med/p95` | per-request decode throughput |
| `e2e ms med` | median per-request total inference time |
| `wall tok/s` | aggregate `total_generated_tokens / wall_clock_ms` — system throughput |

When `conc=1` the `wall tok/s` ≈ `decode tok/s med`; when `conc>1` the
gap shows how much the continuous batcher actually helps.

### Sweeping one dimension

Edit `CASES` in `tests/perf/run_bench_cpp.sh`, or invoke
`qwen35moe_bench` directly:

```bash
./build/qwen35moe_bench --model "$MODEL" \
    --dev-mode auto --gpu-layer 999 \
    --override-tensor ".*ffn_.*_exps.*=CPU" \
    --flash-attn --parallel 4 --n-ubatch 256 \
    --bench-runs 8 --bench-warmup 2 --bench-max-tokens 128 \
    --bench-label auto_expert_offload \
    --bench-output results.md
```

Run it once per value of the dimension you want to sweep; rows
accumulate into the same `results.md`.

## Adding a new flag test

**Correctness (does the flag still wire through?)** — add one row to
`functional/test_matrix.sh`:

```bash
run_case "my_flag_on" \
    "--my-flag value --dev-mode gpu" \
    "[my-component] activated" \      # expected stderr substring
    ""                                # expected_absent (optional)
```

`lib.sh` handles lifecycle, timeout, log capture, cleanup.

**Performance (does the flag actually help t/s?)** — add a row to
`BENCH_MATRIX_ROWS` in `tests/perf/run_bench_cpp.sh` (or extend the
`RANGE_*` arrays). The script builds each label from parameter
abbreviations + values via `build_case_label()`, e.g.
`gpu_fa_par4_pkv_pkvb16_nb512_nub512_thr4_ctx4096_gid1`. Runtime format is
still `"label | <extra args> | concurrency"` inside `CASES`:

```bash
# After build_cases: label is auto-generated; you only edit the matrix row.
"gpu_fa_par4_pkv_pkvb16_nb512_nub512_thr4_ctx4096_gid1 | --dev-mode gpu ... | 4"
```

The wrapper passes `--bench-runs / --bench-warmup / --bench-output / ...`
itself, so matrix rows only need the dimensions that change between cases.
