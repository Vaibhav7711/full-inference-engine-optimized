# RTX 4060 / Ada sm_89 runtime

This directory is a standalone install of the RTX 4060 runtime. It includes the FP16
lossless profile, calibrated INT8 KV and W8A16 profiles for Qwen3-0.6B/1.7B, and the
packed INT4 path for a separately obtained Qwen3-4B-AWQ checkpoint.

Create a Python environment in this directory, install a CUDA-compatible PyTorch build,
then run `pip install -e ".[server]"`. Start the optimized Qwen3-1.7B service with
`./scripts/run_local_chatbot.sh`; start FP16 service with
`uvicorn engine.server.api:create_rtx4060_lossless_app --factory`.
Use `python examples/profile_usage.py --profile lossless` for a library example.

The [repository README](../README.md) contains the full setup, measured Ada numbers,
quality limits, API examples, and links to the research evidence.
