#!/usr/bin/env python3
"""
bench.py — End-to-end throughput benchmark for test_qwen35moe vs llama.cpp.

For each target (mini / llama / both):
  1. Spawn the server with the requested arg set.
  2. Wait for /health (with a generous timeout — model load is slow).
  3. Send N requests (optionally concurrent) of identical prompt / sampler.
  4. Parse the server's own per-request timing lines (those are *exact*: we
     don't have to guess token counts client-side).
  5. Report median prefill tok/s, median decode tok/s, p50 / p95 end-to-end
     latency, and aggregate throughput under concurrency.

Why parse the server log instead of measuring wall-clock and dividing by
strlen(reply)?  Tokenization differs between models / templates / quant
schemes; only the server can report the *real* prompt_tokens and
generated_tokens, which is what tok/s should be normalised by.

Both `test_qwen35moe` and `llama-server` (llama.cpp) print one timing line
per request to stdout/stderr; we extract numbers with simple regex per
backend. Add a new backend by writing a new TimingParser.

This file deliberately uses only the Python 3 standard library so the
benchmark runs anywhere the project already builds (no pip install).

Usage examples:

    # Compare mini (GPU) vs llama.cpp (GPU) on identical params:
    MODEL=/path/to/model.gguf \\
    LLAMA_CPP_BIN=/path/to/llama-server \\
        tests/perf/bench.py --target both --runs 5 --max-tokens 200

    # mini-only sweep over dev modes:
    MODEL=/path/to/model.gguf tests/perf/bench.py \\
        --target mini --mini-args '--dev-mode gpu --flash-attn --paged-kv' \\
        --runs 3 --concurrency 1

    # Concurrent throughput (4 parallel chat sessions):
    tests/perf/bench.py --target both --concurrency 4 --runs 8

Exit code: 0 if every request succeeded, 1 otherwise.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Iterable, Optional


# ── Server lifecycle ──────────────────────────────────────────────────────────


@dataclass
class ServerHandle:
    name: str                       # "mini" / "llama"
    proc: subprocess.Popen
    port: int
    log_path: str

    def stop(self, grace: float = 8.0) -> None:
        if self.proc.poll() is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            self.proc.terminate()
        try:
            self.proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                self.proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.proc.wait(timeout=grace)


def wait_for_health(port: int, log_path: str, timeout_s: float) -> bool:
    """Probe /health every 0.5s until 200 or until timeout."""
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        # Useful diagnostic for users debugging a server that exited early:
        # if the log contains a fatal error there's no point continuing to wait.
        if os.path.exists(log_path):
            with contextlib.suppress(OSError):
                tail = _tail(log_path, 2000)
                if any(p in tail for p in (
                    "Segmentation fault", "terminate called",
                    "CUDA error", "illegal memory access", "Aborted",
                    "HTTP server initialization failed",
                    "Engine initialization failed",
                )):
                    return False
        time.sleep(0.5)
    return False


def _tail(path: str, n_bytes: int) -> str:
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - n_bytes))
        return f.read().decode("utf-8", errors="replace")


def start_mini(bin_path: str, model: str, port: int, log_path: str,
               threads: int, ctx: int, extra_args: list[str]) -> ServerHandle:
    cmd = [
        bin_path,
        "--model", model,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--threads", str(threads),
        "--ctx-size", str(ctx),
        "--seed", "42",
        *extra_args,
    ]
    log_file = open(log_path, "wb", buffering=0)
    proc = subprocess.Popen(
        cmd, stdout=log_file, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )
    return ServerHandle("mini", proc, port, log_path)


def start_llama(bin_path: str, model: str, port: int, log_path: str,
                threads: int, ctx: int, extra_args: list[str]) -> ServerHandle:
    """Spawn llama.cpp's `llama-server`. Args mirror the mini server's intent.

    We force `--metrics` on so eval timing is verbose enough to parse without
    relying on streaming. Pass --n-gpu-layers / -fa / etc. through extra_args.
    """
    cmd = [
        bin_path,
        "--model", model,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--threads", str(threads),
        "--ctx-size", str(ctx),
        "--seed", "42",
        "--metrics",
        *extra_args,
    ]
    log_file = open(log_path, "wb", buffering=0)
    proc = subprocess.Popen(
        cmd, stdout=log_file, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )
    return ServerHandle("llama", proc, port, log_path)


# ── Timing extractors ────────────────────────────────────────────────────────
#
# Each backend prints exactly one timing line per request (mini's is
# `[Timing][slot=...] ...`, llama's is `prompt eval time = ...` followed by
# `eval time = ...`). We parse those tokens after the request completes and
# match them back to the request via a monotonic counter — the parsers consume
# only the segment of log written *after* the last consumed offset.


@dataclass
class Sample:
    prompt_tokens: int
    generated_tokens: int
    prefill_ms: float
    decode_ms: float
    e2e_ms: float
    prefill_tok_s: float
    decode_tok_s: float
    ok: bool = True
    raw_log_line: str = ""


class TimingParser:
    """Streaming parser for a backend's stdout/stderr log."""

    def __init__(self, log_path: str) -> None:
        self.log_path = log_path
        self._offset = 0

    def _read_new(self) -> str:
        if not os.path.exists(self.log_path):
            return ""
        with open(self.log_path, "rb") as f:
            f.seek(self._offset)
            chunk = f.read()
        self._offset += len(chunk)
        return chunk.decode("utf-8", errors="replace")

    def parse_new(self) -> list[Sample]:
        raise NotImplementedError


# Format example (chat.cpp:269):
# [Timing][slot=0][ok] prompt_tokens=42 generated_tokens=80 queue=0.10 ms
# prefill=120.30 ms (349.13 tok/s) decode=820.40 ms (97.51 tok/s)
# total=940.80 ms e2e=940.91 ms output=85.05 tok/s all=129.71 tok/s
_MINI_RE = re.compile(
    r"\[Timing\]\[slot=\d+\]\[(?P<ok>ok|error)\]\s+"
    r"prompt_tokens=(?P<pt>\d+)\s+generated_tokens=(?P<gt>\d+)\s+"
    r"queue=(?P<queue>[\d.]+) ms\s+"
    r"prefill=(?P<prefill>[\d.]+) ms \((?P<prefill_ts>[\d.]+) tok/s\)\s+"
    r"decode=(?P<decode>[\d.]+) ms \((?P<decode_ts>[\d.]+) tok/s\)\s+"
    r"total=(?P<total>[\d.]+) ms\s+e2e=(?P<e2e>[\d.]+) ms"
)


class MiniParser(TimingParser):
    def parse_new(self) -> list[Sample]:
        out: list[Sample] = []
        chunk = self._read_new()
        for m in _MINI_RE.finditer(chunk):
            out.append(Sample(
                prompt_tokens=int(m.group("pt")),
                generated_tokens=int(m.group("gt")),
                prefill_ms=float(m.group("prefill")),
                decode_ms=float(m.group("decode")),
                e2e_ms=float(m.group("e2e")),
                prefill_tok_s=float(m.group("prefill_ts")),
                decode_tok_s=float(m.group("decode_ts")),
                ok=(m.group("ok") == "ok"),
                raw_log_line=m.group(0),
            ))
        return out


# llama.cpp llama-server prints (one per request):
#   prompt eval time =   123.45 ms /    42 tokens (    2.94 ms per token,   340.20 tokens per second)
#   eval time =          820.40 ms /    80 runs   (   10.26 ms per token,    97.51 tokens per second)
#   total time =         943.85 ms /   122 tokens
_LLAMA_PREFILL_RE = re.compile(
    r"prompt eval time\s*=\s*(?P<ms>[\d.]+)\s*ms\s*/\s*(?P<tokens>\d+)\s*tokens"
    r".*?(?P<tps>[\d.]+)\s*tokens per second",
    re.IGNORECASE,
)
_LLAMA_DECODE_RE = re.compile(
    r"\beval time\s*=\s*(?P<ms>[\d.]+)\s*ms\s*/\s*(?P<tokens>\d+)\s*runs"
    r".*?(?P<tps>[\d.]+)\s*tokens per second",
    re.IGNORECASE,
)
_LLAMA_TOTAL_RE = re.compile(
    r"total time\s*=\s*(?P<ms>[\d.]+)\s*ms",
    re.IGNORECASE,
)


class LlamaParser(TimingParser):
    def parse_new(self) -> list[Sample]:
        out: list[Sample] = []
        chunk = self._read_new()
        # llama-server prints these three lines together per request. Walk the
        # chunk in order; pair (prefill, decode, total) triplets.
        idx = 0
        while True:
            m_pre = _LLAMA_PREFILL_RE.search(chunk, idx)
            if not m_pre:
                break
            m_dec = _LLAMA_DECODE_RE.search(chunk, m_pre.end())
            if not m_dec:
                break
            m_tot = _LLAMA_TOTAL_RE.search(chunk, m_dec.end())
            if not m_tot:
                break
            out.append(Sample(
                prompt_tokens=int(m_pre.group("tokens")),
                generated_tokens=int(m_dec.group("tokens")),
                prefill_ms=float(m_pre.group("ms")),
                decode_ms=float(m_dec.group("ms")),
                e2e_ms=float(m_tot.group("ms")),
                prefill_tok_s=float(m_pre.group("tps")),
                decode_tok_s=float(m_dec.group("tps")),
                ok=True,
                raw_log_line=chunk[m_pre.start():m_tot.end()].strip(),
            ))
            idx = m_tot.end()
        return out


# ── HTTP client ──────────────────────────────────────────────────────────────


_DEFAULT_PROMPT = (
    "Write a single short paragraph (max 6 sentences) explaining what "
    "matrix multiplication is, suitable for a high school student. "
    "Be concise and direct."
)


def chat_complete(port: int, prompt: str, *, max_tokens: int, temperature: float,
                  top_k: int, top_p: float, timeout_s: float) -> bool:
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            _ = resp.read()
            return True
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        sys.stderr.write(f"[bench] HTTP error on port {port}: {e}\n")
        return False


# ── Benchmark drivers ────────────────────────────────────────────────────────


@dataclass
class BackendReport:
    name: str
    args: list[str]
    samples: list[Sample] = field(default_factory=list)
    failures: int = 0
    wall_clock_s: float = 0.0
    concurrency: int = 1

    def succeeded(self) -> list[Sample]:
        return [s for s in self.samples if s.ok]


def run_workload(handle: ServerHandle, parser: TimingParser, *, prompt: str,
                 runs: int, concurrency: int, max_tokens: int, temperature: float,
                 top_k: int, top_p: float, request_timeout: float) -> BackendReport:
    report = BackendReport(name=handle.name, args=[], concurrency=concurrency)

    # Warm-up request (not counted) — first invocation can pay extra cost
    # for graph capture / kernel JIT / paged-KV bucket allocation.
    sys.stderr.write(f"[bench:{handle.name}] warmup ...\n")
    chat_complete(handle.port, prompt, max_tokens=max(8, max_tokens // 4),
                  temperature=0.0, top_k=1, top_p=1.0, timeout_s=request_timeout)
    time.sleep(0.5)
    parser.parse_new()  # discard warm-up samples

    sys.stderr.write(
        f"[bench:{handle.name}] running {runs} requests "
        f"(concurrency={concurrency}, max_tokens={max_tokens}) ...\n"
    )
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                chat_complete, handle.port, prompt,
                max_tokens=max_tokens, temperature=temperature,
                top_k=top_k, top_p=top_p, timeout_s=request_timeout,
            )
            for _ in range(runs)
        ]
        for fut in concurrent.futures.as_completed(futures):
            if not fut.result():
                report.failures += 1
    report.wall_clock_s = time.perf_counter() - t0

    # Give the server a beat to flush its [Timing] lines.
    time.sleep(0.5)
    report.samples = parser.parse_new()
    return report


# ── Reporting ────────────────────────────────────────────────────────────────


def _p(xs: Iterable[float], q: float) -> float:
    xs = sorted(xs)
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]
    idx = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
    return xs[idx]


def summarise(report: BackendReport) -> dict:
    succ = report.succeeded()
    if not succ:
        return {
            "name": report.name, "n": 0, "failures": report.failures,
            "wall_clock_s": report.wall_clock_s,
        }
    pre_tps = [s.prefill_tok_s for s in succ]
    dec_tps = [s.decode_tok_s for s in succ]
    e2e_ms = [s.e2e_ms for s in succ]
    total_gen = sum(s.generated_tokens for s in succ)
    return {
        "name": report.name,
        "n": len(succ),
        "failures": report.failures,
        "wall_clock_s": report.wall_clock_s,
        "concurrency": report.concurrency,
        "prompt_tokens_median": int(statistics.median(s.prompt_tokens for s in succ)),
        "generated_tokens_median": int(statistics.median(s.generated_tokens for s in succ)),
        "prefill_tok_s_median": round(statistics.median(pre_tps), 2),
        "prefill_tok_s_p95":   round(_p(pre_tps, 0.95), 2),
        "decode_tok_s_median": round(statistics.median(dec_tps), 2),
        "decode_tok_s_p95":    round(_p(dec_tps, 0.95), 2),
        "e2e_ms_median":       round(statistics.median(e2e_ms), 2),
        "e2e_ms_p95":          round(_p(e2e_ms, 0.95), 2),
        "aggregate_gen_tok_s": round(total_gen / report.wall_clock_s, 2) if report.wall_clock_s > 0 else 0,
    }


def print_markdown_table(rows: list[dict]) -> None:
    headers = [
        "backend", "runs", "fail", "conc",
        "prompt_tok (med)", "gen_tok (med)",
        "prefill tok/s (med)", "decode tok/s (med)",
        "e2e ms (med, p95)", "aggregate gen tok/s",
    ]
    print()
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        if r["n"] == 0:
            print(f"| {r['name']} | {0} | {r['failures']} | – | – | – | – | – | – | – |")
            continue
        print(
            f"| {r['name']} | {r['n']} | {r['failures']} | {r['concurrency']} | "
            f"{r['prompt_tokens_median']} | {r['generated_tokens_median']} | "
            f"{r['prefill_tok_s_median']} | {r['decode_tok_s_median']} | "
            f"{r['e2e_ms_median']} / {r['e2e_ms_p95']} | "
            f"{r['aggregate_gen_tok_s']} |"
        )
    print()


# ── Main ─────────────────────────────────────────────────────────────────────


def split_arg_string(s: Optional[str]) -> list[str]:
    if not s:
        return []
    # Mirrors how shell splits — good enough since our test args never
    # contain quoted spaces.
    return s.split()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=["mini", "llama", "both"], default="both",
                    help="Which servers to benchmark.")
    ap.add_argument("--runs", type=int, default=5,
                    help="Number of measured requests per backend (excludes warmup).")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Number of concurrent in-flight requests.")
    ap.add_argument("--max-tokens", type=int, default=200,
                    help="max_tokens for each request.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--prompt", default=_DEFAULT_PROMPT)
    ap.add_argument("--mini-args", default="--dev-mode gpu --flash-attn --paged-kv",
                    help="Extra CLI args appended when spawning test_qwen35moe.")
    ap.add_argument("--llama-args", default="--n-gpu-layers 999 -fa",
                    help="Extra CLI args appended when spawning llama-server.")
    ap.add_argument("--threads", type=int,
                    default=int(os.environ.get("THREADS", "4")))
    ap.add_argument("--ctx", type=int,
                    default=int(os.environ.get("CTX", "2048")))
    ap.add_argument("--mini-port", type=int,
                    default=int(os.environ.get("BASE_PORT", "7700")) + 100)
    ap.add_argument("--llama-port", type=int,
                    default=int(os.environ.get("BASE_PORT", "7700")) + 101)
    ap.add_argument("--ready-timeout", type=float, default=300.0,
                    help="Server start-up timeout in seconds (model load).")
    ap.add_argument("--request-timeout", type=float, default=180.0)
    ap.add_argument("--json-out", default=None,
                    help="If set, write the full report (one JSON per line) here.")
    args = ap.parse_args()

    model = os.environ.get("MODEL")
    if not model:
        sys.stderr.write("ERROR: set MODEL=/path/to/model.gguf\n")
        return 2
    if not os.path.isfile(model):
        sys.stderr.write(f"ERROR: model not found: {model}\n")
        return 2

    mini_bin = os.environ.get("BIN", "./build/test_qwen35moe")
    llama_bin = os.environ.get("LLAMA_CPP_BIN", "")
    if args.target in ("mini", "both") and not (os.path.isfile(mini_bin) and os.access(mini_bin, os.X_OK)):
        sys.stderr.write(f"ERROR: mini binary not found or not executable: {mini_bin}\n")
        return 2
    if args.target in ("llama", "both"):
        if not llama_bin:
            sys.stderr.write(
                "ERROR: LLAMA_CPP_BIN is not set. Either set it to a built "
                "llama-server, or pass --target mini to skip the comparison.\n"
            )
            return 2
        if not (os.path.isfile(llama_bin) and os.access(llama_bin, os.X_OK)):
            sys.stderr.write(f"ERROR: llama binary not found or not executable: {llama_bin}\n")
            return 2

    log_dir = tempfile.mkdtemp(prefix="qwen35moe-bench.")
    sys.stderr.write(f"[bench] logs: {log_dir}\n")

    reports: list[BackendReport] = []
    handles: list[ServerHandle] = []
    try:
        if args.target in ("mini", "both"):
            log_path = os.path.join(log_dir, "mini.log")
            h = start_mini(mini_bin, model, args.mini_port, log_path,
                           args.threads, args.ctx, split_arg_string(args.mini_args))
            handles.append(h)
            if not wait_for_health(h.port, h.log_path, args.ready_timeout):
                sys.stderr.write(f"[bench:mini] server failed to come up — check {log_path}\n")
            else:
                report = run_workload(
                    h, MiniParser(h.log_path),
                    prompt=args.prompt, runs=args.runs, concurrency=args.concurrency,
                    max_tokens=args.max_tokens, temperature=args.temperature,
                    top_k=args.top_k, top_p=args.top_p,
                    request_timeout=args.request_timeout,
                )
                report.args = split_arg_string(args.mini_args)
                reports.append(report)

        if args.target in ("llama", "both"):
            log_path = os.path.join(log_dir, "llama.log")
            h = start_llama(llama_bin, model, args.llama_port, log_path,
                            args.threads, args.ctx, split_arg_string(args.llama_args))
            handles.append(h)
            if not wait_for_health(h.port, h.log_path, args.ready_timeout):
                sys.stderr.write(f"[bench:llama] server failed to come up — check {log_path}\n")
            else:
                report = run_workload(
                    h, LlamaParser(h.log_path),
                    prompt=args.prompt, runs=args.runs, concurrency=args.concurrency,
                    max_tokens=args.max_tokens, temperature=args.temperature,
                    top_k=args.top_k, top_p=args.top_p,
                    request_timeout=args.request_timeout,
                )
                report.args = split_arg_string(args.llama_args)
                reports.append(report)
    finally:
        for h in handles:
            h.stop()

    summaries = [summarise(r) for r in reports]
    print_markdown_table(summaries)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            for r, s in zip(reports, summaries):
                f.write(json.dumps({
                    "name": r.name,
                    "args": r.args,
                    "summary": s,
                    "samples": [s_.__dict__ for s_ in r.samples],
                }) + "\n")

    # Exit non-zero if any backend failed all requests or had failures.
    total_failures = sum(r.failures for r in reports)
    return 0 if total_failures == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.stderr.write("\n[bench] interrupted\n")
        sys.exit(130)
