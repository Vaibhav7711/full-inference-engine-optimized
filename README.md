# Full Inference Engine: T4 and Ada runtimes

A single-GPU LLM inference engine with continuous batching, paged KV cache, chunked
prefill, CUDA graphs, Triton kernels, prefix reuse, streaming, and an
OpenAI-compatible HTTP API. This public repository contains **two standalone,
hardware-specific runtime trees**, not one package that silently chooses between
unrelated measured configurations:

| Path | GPU used for validation | Default serving policy | Main validated checkpoint |
| --- | --- | --- | --- |
| [`t4-sm75/`](t4-sm75/) | NVIDIA Tesla T4, Turing `sm_75` | FP16, graphed Triton paged decode, SDPA prefill | Qwen3-0.6B |
| [`ada-sm89/`](ada-sm89/) | NVIDIA GeForce RTX 4060, Ada `sm_89`, 8 GB | Graphed Triton paged decode, tiled prefill on 16-token pages; explicit FP16 or quantized profile | Qwen3-0.6B and 1.7B; separate 4B-AWQ path |

Each directory has its own `engine/`, `examples/`, `scripts/`, `pyproject.toml`,
`README.md`, and `LICENSE`. Install and run **from one directory at a time**: both
distributions import as `engine` and must not share a Python environment. The
repository root is documentation, not an installable package. Model weights are
downloaded separately; no checkpoints, benchmark datasets, or raw profiler dumps
are bundled here.

## Measured results

These are workload-specific observations, not universal speedups or cross-GPU
comparisons. TTFT is time to first token, ITL is inter-token latency, and throughput
counts delivered output tokens. The [full Ada report](https://github.com/Vaibhav7711/full-inference-engine/blob/rtx/ada-sm89/docs/rtx4060-ada-sm89-complete-report.md)
records setup, acceptance gates, negative results, and links to raw JSON.

### Ada `sm_89`: Qwen3-1.7B serving

At an equal **5 GiB weights-plus-KV budget**, three interleaved end-to-end runs
compared FP16 with a stack of W8A16 MLP and vocabulary projections, calibrated
INT8 KV, tiled INT8 prefill, and packed fused rows. Chat used 32 requests at 4/s
with up to eight active requests; long-context used 12 requests, 3–4k-token
prompts at 1.5/s, and up to eight active requests.

| Metric | FP16 | Measured stack | Change |
| --- | ---: | ---: | ---: |
| Chat throughput | 318.2 tok/s | 439.4 tok/s | **+38.1%** |
| Chat TTFT p50 | 1,890 ms | 852 ms | **−54.9%** |
| Chat ITL p50 / p99 | 17.40 / 43.09 ms | 10.84 / 26.09 ms | **−37.7% / −39.5%** |
| Long-context throughput | 83.7 tok/s | 128.5 tok/s | **+53.6%** |
| Long-context TTFT p50 | 6,235 ms | 3,497 ms | **−43.9%** |
| Long-context ITL p50 / p99 | 21.06 / 37.78 ms | 27.42 / 35.34 ms | **+30.2% / −6.5%** |
| Long-context preemptions | 9 | 0 | eliminated |
| KV token capacity | 16,800 | 45,872 | **2.73×** |

The long-context median ITL rises because the larger KV pool keeps more requests
active instead of queueing or preempting them. The stack is **lossy**: teacher-forced
top-1 agreement with FP16 was 97.3% at 1k context and 97.7% at 4k; KL divergence
was 0.0041/0.0038. Use the FP16 `lossless` profile if those differences matter.
See the [chat](https://github.com/Vaibhav7711/full-inference-engine/blob/rtx/ada-sm89/results/rtx4060/stacked_vs_fp16_chat_1.7b.json)
and [long-context](https://github.com/Vaibhav7711/full-inference-engine/blob/rtx/ada-sm89/results/rtx4060/stacked_vs_fp16_long_c8_1.7b.json)
measurements.

**Comparability limit:** the shipped 1.7B `optimized` factory also enables W8A16
attention projections. They were *not* in the stacked A/B above. Their decode
benefit was measured, but their additional loaded-service benefit was unresolved.
The table's equal-budget pool sizes are also benchmark settings, not the factory's
default block counts. Do not attribute the table verbatim to the default server.

Other RTX 4060 results:

| Case | Measured result | Scope |
| --- | --- | --- |
| Qwen3-0.6B, long prompts, equal 3 GiB weights-plus-KV budget | Calibrated INT8 KV: 166.5 → 190.2 tok/s (**+14.2%**); TTFT p50 1,383 → 998 ms (**−27.9%**); KV capacity 17,680 → 34,832 tokens (**1.97×**) | [Loaded-service A/B](https://github.com/Vaibhav7711/full-inference-engine/blob/rtx/ada-sm89/results/rtx4060/optimized_vs_fp16_long_c8_0.6b.json); this comparison isolates the KV-oriented stack, not every default 0.6B profile feature. |
| Qwen3-4B-AWQ, one 2k-token prompt and 32 output tokens | Hybrid packed-INT4 prefill TTFT 2,364 → 620 ms (**3.81×**) versus tinygemm-only | [Single-prompt prefill A/B](https://github.com/Vaibhav7711/full-inference-engine/blob/rtx/ada-sm89/results/rtx4060/awq_prefill_path_ab_4b.json), **not** loaded-service throughput. AWQ checkpoint is external; KV remains FP16. |
| Qwen3-1.7B, one 128-token prompt and 128 output tokens | Tuned path: 8.92 ms ITL, 112 tok/s, 2.68 GiB peak allocation; explicit naive decoder: 17.14 ms, 58 tok/s, 3.26 GiB | [Tuned](https://github.com/Vaibhav7711/full-inference-engine/blob/rtx/ada-sm89/results/rtx4060/single_request/Qwen3-1.7B_current_optimized.json) / [naive](https://github.com/Vaibhav7711/full-inference-engine/blob/rtx/ada-sm89/results/rtx4060/single_request/Qwen3-1.7B_naive.json). Separate single-request harness; no W8 attention. |

### T4 `sm_75`: historical Qwen3-0.6B FP16 chat

On the T4 with roughly 656-token prompts, the early-to-later engine measurements
were as follows. They span development stages and are **not** a controlled single
interleaved comparison of two final releases.

| Metric | Early T4 measurement | Later T4 measurement |
| --- | ---: | ---: |
| Prefill-carrying step p50 | 98 ms | 20.5 ms |
| Decode-only step p50, batch about five | 9.7 ms | 10.2 ms |
| ITL p50 / p99 | about 25 / 295 ms | 19.8 / 33 ms |
| TTFT p50 | 2.8 s | about 0.3–0.4 s |

The improvement mostly came from prefill and mixed-step scheduling, not a faster
decode-only step. The [T4 optimization journal](https://github.com/Vaibhav7711/full-inference-engine/blob/main/docs/optimization-journal.md)
and [attribution plan](https://github.com/Vaibhav7711/full-inference-engine/blob/main/docs/t4-reevaluation-plan.md)
contain the detailed experiment history and qualifications. Do not compare the T4
numbers directly with the Ada table: GPU, workload, model configuration, and
measurement period differ.

## Install and run

Requires Linux, an NVIDIA CUDA GPU, Python 3.10+, and a CUDA-compatible PyTorch
installation. Use a CUDA/PyTorch/Triton combination supported by your driver;
installing a mismatched wheel can fail before the engine runs. Hugging Face access
is needed to download the Qwen3 checkpoint. Each tree has the same distribution
and import name, so use **separate virtual environments**.

For T4:

```bash
cd t4-sm75
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
# Install a CUDA-compatible PyTorch build for this machine first.
.venv/bin/python -m pip install -e '.[server]'
./scripts/run_local_chatbot.sh
```

For RTX 4060:

```bash
cd ada-sm89
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
# Install a CUDA-compatible PyTorch build for this machine first.
.venv/bin/python -m pip install -e '.[server]'
./scripts/run_local_chatbot.sh
```

The scripts bind to `127.0.0.1:8000` and provide a local chat UI. The T4 script
loads Qwen3-0.6B FP16; the Ada script loads Qwen3-1.7B `optimized` by default.
From inside `ada-sm89/`, choose FP16 instead with:

```bash
.venv/bin/python -m uvicorn engine.server.api:create_rtx4060_lossless_app --factory --host 127.0.0.1 --port 8000
```

Or choose the measured page-256 FP16 long-context Flash **prefill** profile with
`engine.server.api:create_rtx4060_flash_app`. It requires a compatible, separately
installed FlashAttention-2 build; Flash **decode** is not the default and lost
end-to-end against graphed Triton in the measured workloads. The normal 16-token
page geometry uses tiled prefill on Ada and SDPA on T4.

Both trees expose the same HTTP surface:

```bash
curl -s http://127.0.0.1:8000/ready
curl -s http://127.0.0.1:8000/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"Explain paged KV caching briefly.","max_new_tokens":64}'
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3-1.7B","messages":[{"role":"user","content":"Hello"}],"max_tokens":64,"stream":true}'
curl -s http://127.0.0.1:8000/metrics
```

For T4, use `"model":"Qwen/Qwen3-0.6B"` in the chat request. Routes include
`/generate`, `/generate/stream`, `/v1/completions`, `/v1/chat/completions`,
`/health`, `/ready`, and `/metrics`. The OpenAI-style routes support streaming,
usage, logprobs, stop strings and per-request sampling controls. Startup warms
eligible CUDA graph buckets before readiness, so the first live request does not
pay their compilation/capture cost.

For library use, run `python examples/library_usage.py` from `t4-sm75/` or
`python examples/profile_usage.py --profile optimized` from `ada-sm89/`.
The Ada example also accepts `--profile lossless`, `--model Qwen/Qwen3-0.6B`,
and `--num-blocks N`. The `build_engine()` factory in
[`ada-sm89/engine/profiles.py`](ada-sm89/engine/profiles.py) is the main entry point
for selecting FP16 or the calibrated quantized configuration. It uses default KV
block counts of 1,024/2,048 for 1.7B lossless/optimized and 1,536/2,560 for
0.6B lossless/optimized on the 8 GB card. These are deployable defaults, not
the equal-budget experiment's block counts. The 4B-AWQ profile uses 1,024 blocks
and needs a separately obtained compatible AWQ checkpoint.

## What is implemented

`HTTP/SSE → service worker → FCFS scheduler → packed decode/prefill step →
model forward → paged attention → sampling → per-request KV commit`. The service
admits by KV capacity, chunks long prompts under a token budget, and can preempt
or cancel requests. A refcounted prefix cache reuses exact prompt pages; partial
tails are copy-on-write. CUDA graphs replay eligible decode, prefill, and mixed
steps. Metrics cover request latency, throughput, allocations, prefix hits,
preemptions, and spill/restore. The runtime rejects unsupported model geometry
with a reason instead of silently selecting an incompatible attention kernel.

| Capability | T4 tree | Ada tree |
| --- | --- | --- |
| Continuous batching, paged KV, prefix reuse, streaming, metrics | Yes | Yes |
| Graphed Triton paged decode | Yes | Yes, tuned dispatch |
| Normal page-16 prefill | SDPA | Tiled `tl.dot` |
| Lossless FP16 weights and KV | Default | Explicit `lossless` profile |
| Calibrated INT8 KV and W8A16 weight-only profiles | Kernel code is present, not the measured T4 serving default | Qwen3-0.6B/1.7B `optimized` profiles |
| Packed INT4 Qwen3-4B-AWQ path | No | Yes; external checkpoint |
| Direct paged FlashAttention-2 prefill | Not T4 default | Opt-in page-256 FP16 path |
| Integrated n-gram/draft speculative serving | Present in T4 tree | Not integrated in Ada service; experimental batched code exists |

The T4 and Ada policies reflect different hardware. On T4, the early tiled
prefill kernel missed tensor cores and lost badly; SDPA was chosen. On Ada,
the tiled path uses tensor cores and wins at normal 16-token pages. The
[Ada report](https://github.com/Vaibhav7711/full-inference-engine/blob/rtx/ada-sm89/docs/rtx4060-ada-sm89-complete-report.md)
also documents rejected split-K settings, Flash decode, dense-gather Flash,
unfavorable weight-quantization shapes, and why isolated kernel wins were not
accepted without end-to-end evidence.

### Optimization decisions on Ada

| Area | Accepted runtime choice | What was rejected or limited |
| --- | --- | --- |
| Decode attention | Graph-safe Triton paged decode with measured launch/geometry policy | Flash decode lost end-to-end; split-K helped the isolated batch-1/4k cell but was not a broad default; higher occupancy alone did not guarantee lower latency. |
| FP16 prefill | Tensor-core tiled prefill on normal 16-token pages; direct paged Flash prefill only with 256-token pages and long prompts | Dense-gather Flash on page-16 and forced Flash decode split counts lost after setup/copy costs. |
| INT8 KV | Per-head calibrated key smoothing, native paged INT8 decode, tiled INT8 prefill | Plain per-token INT8 keys failed quality; batch-8 tuned decode rule did not win. AWQ 4B still uses FP16 KV. |
| Weights | Searched W8A16 MLP and vocabulary kernels on 0.6B/1.7B; W8A16 attention only on the 1.7B `optimized` profile | W8A16 attention on 0.6B failed the early-token quality gate; W8A8 and installed-runtime FP8 were not accepted. |
| AWQ 4B | Packed INT4 weights, three-tier prefill dispatch, graph capture with packed-width rows | Earlier scalar-gather INT4 prefill and tinygemm/padded prefill graphs lost. |
| Scheduling | Packed fused decode/prefill rows, 128-token chunk/budget, idle whole-prompt path, prefix reuse | Larger prefill budgets and GPU-resident metadata did not establish a general service win. |

The T4 tree also contains an **explicit optional two-T4 draft/target
speculative profile**; it is separate from the default single-GPU service.
The Ada tree has experimental batched speculative code but does not wire it
into its serving engine. Neither tree claims general tensor/model parallelism.

## Validation and provenance

The Ada runtime was copied from the research repository's `rtx/ada-sm89`
snapshot at commit `8495d24`; its full gate there passed **316 non-CUDA tests
and 265 CUDA tests**, with one expected skip for an 8 GB speculative case.
The T4 tree is derived from this repository's `ca288c0` runtime, with its
misplaced RTX 4060-only Flash app factory removed. This public repo contains
working runtime code, examples and launchers, while the
[research repository](https://github.com/Vaibhav7711/full-inference-engine)
holds tests, benchmark harnesses, raw result JSON, chronological journals, and
the [complete Ada technical report](https://github.com/Vaibhav7711/full-inference-engine/blob/rtx/ada-sm89/docs/rtx4060-ada-sm89-complete-report.md).
The small directory-specific README files provide focused quick starts. Only
the Ada tree has been validated on the local RTX 4060 during this publication
pass; this machine cannot re-run T4-specific performance measurements. Results
are established for the listed Qwen3 checkpoints, not arbitrary models or GPUs.

The project is beta software. It does not claim general multi-GPU serving, guaranteed
performance on other hardware, or exact FP16 token identity in quantized modes.
Licensed under [Apache 2.0](LICENSE).
