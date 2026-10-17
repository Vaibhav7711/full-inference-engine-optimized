# Tesla T4 / sm_75 runtime

This directory is a standalone install of the validated T4 runtime. It uses FP16,
graph-replayed Triton paged decode, and SDPA prefill on normal 16-token KV pages.

Create a Python environment in this directory, install a CUDA-compatible PyTorch build,
then run `pip install -e ".[server]"`. Start the Qwen3-0.6B service with
`./scripts/run_local_chatbot.sh` or run `python examples/library_usage.py`.

The [repository README](../README.md) contains the full setup, measured T4 numbers,
feature matrix, API examples, and links to the research evidence.
