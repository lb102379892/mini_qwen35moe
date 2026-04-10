# mini_qwen35moe — CPU-only inference for Qwen3.5 35B MoE

A minimal, dependency-light implementation of Qwen3.5-35B-A3B MoE CPU inference
built on top of the bundled **ggml** library.

## Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
# Binary: build/test_qwen35moe
```

Requires: CMake >= 3.14, C++17, Linux or macOS.

## Run

### Single prompt (one-shot generation)
```bash
./build/test_qwen35moe \
  --model Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_M.gguf \
  --prompt "Hello, explain quantum computing" \
  --n-predict 256 \
  --temp 0.7 \
  --threads 8
```

### Interactive REPL (omit --prompt)
```bash
./build/test_qwen35moe \
  --model Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_M.gguf \
  --threads 8
```

### Greedy decoding (deterministic)
```bash
./build/test_qwen35moe \
  --model Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q5_K_M.gguf \
  --prompt "2 + 2 = ?" \
  --temp 0 \
  --n-predict 64
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| --model \<path\> | (required) | Path to .gguf model file |
| --prompt \<text\> | (interactive) | Input prompt |
| --system \<text\> | "You are a helpful assistant." | Chat system message |
| --n-predict \<N\> | 256 | Max new tokens to generate |
| --temp \<f\> | 0.8 | Temperature (0 = greedy) |
| --top-p \<f\> | 0.95 | Top-p sampling nucleus |
| --top-k \<N\> | 40 | Top-k sampling (0 = off) |
| --threads \<N\> | 4 | CPU threads |
| --no-chat | false | Disable chat template, pass prompt verbatim |
| --verbose | false | Print tokenization and timing info |

## Architecture

- **40 transformer blocks**: every 4th block (il % 4 == 3) is GQA full-attention;
  all others are DeltaNet SSM (gated linear attention).
- **MoE FFN** in all layers: 256 experts, top-8, ff_dim=512, plus a shared expert.
- **No KV cache**: every forward pass reprocesses the entire token sequence.
- **Tokenizer**: GPT-2 byte-level BPE loaded from GGUF metadata.
- **Sampling**: greedy, temperature + top-k + top-p.

## Memory requirements

The Q5_K_M quantized model requires approximately 24 GB RAM to load all weights.
