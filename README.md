# Full Inference Engine — optimized runtime

A clean, usage-focused distribution of the single-GPU LLM inference engine. This
repository contains the runnable engine and its public entry points only: no tests,
benchmarks, result artifacts, experiment notebooks, or research journals.

## Included optimizations

- Continuous batching with per-request sampling
- Paged KV cache with prefix reuse and preemption
- Chunked prefill interleaved with decode
- CUDA graphs for decode, prefill, and fused steps
- Triton paged-attention, RMSNorm, RoPE, and SwiGLU kernels
- GPU-specific attention backend selection
- Optional INT8 KV cache
- Optional n-gram or draft-model speculative decoding
- OpenAI-compatible HTTP and streaming API

The runtime is designed for NVIDIA CUDA GPUs. It has been validated primarily with
Qwen3 and other dense Llama-style causal language models.

## Requirements

- Linux with an NVIDIA GPU and a working CUDA driver
- Python 3.10 or newer
- A CUDA-compatible PyTorch installation
- Enough GPU memory for the model, KV pool, activations, and graph captures

## Install

Create an environment and install the PyTorch build appropriate for your CUDA setup
first. Then install this repository:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[server]"
```

PyTorch's CUDA distribution supplies Triton on supported Linux installations.

## Use as a library

```python
from engine.batching.continuous_batching import ContinuousBatchingEngine
from engine.model import load_model

loaded = load_model("Qwen/Qwen3-0.6B", dtype="float16")
runtime = ContinuousBatchingEngine(
    loaded.model,
    loaded.tokenizer,
    loaded.device,
    max_active=8,
    num_blocks=1024,
    block_size=16,
    cuda_graph_batch_sizes=(1, 2, 4, 8),
)
runtime.warmup()

token_batches = runtime.generate(
    ["Explain KV caching in one sentence."],
    max_new_tokens=64,
)
print(loaded.tokenizer.decode(token_batches[0], skip_special_tokens=True))
```

The complete usage example covers automatic KV-pool sizing, backend selection,
batched generation, seeded sampling, staggered arrivals, and latency reporting:

```bash
python examples/library_usage.py --model Qwen/Qwen3-0.6B
```

## Run the server

Start the default Qwen3-0.6B server:

```bash
uvicorn engine.server.api:create_app --factory --host 127.0.0.1 --port 8000
```

Or, after installing into `.venv`, use the included launcher:

```bash
./scripts/run_local_chatbot.sh
```

Wait until the model and CUDA graphs are ready:

```bash
until curl -sf http://127.0.0.1:8000/ready >/dev/null; do sleep 2; done
```

Send a native request:

```bash
curl -s http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain paged attention.","max_new_tokens":64}'
```

Or use the OpenAI-compatible endpoint:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"Qwen/Qwen3-0.6B",
    "messages":[{"role":"user","content":"Explain paged attention."}],
    "max_tokens":64
  }'
```

The browser UI is available at `http://127.0.0.1:8000/`, readiness at `/ready`,
health at `/health`, and Prometheus-format metrics at `/metrics`.

## Repository contents

```text
engine/                    Runtime implementation
examples/library_usage.py End-to-end library usage
scripts/run_local_chatbot.sh
                           Local server launcher
pyproject.toml             Package and dependency metadata
LICENSE                    Apache-2.0 license
```

## Capacity note

KV capacity is `num_blocks × block_size` tokens shared across active requests. Reduce
`num_blocks` if startup reports that the KV pool cannot fit. The engine deliberately
warms Triton kernels and captures configured CUDA graph shapes before reporting ready,
so initial startup takes longer than an ordinary eager model load.

## License

Apache-2.0. See [LICENSE](LICENSE).
