"""Generate with a measured RTX 4060 profile in this standalone directory.

Run from ``ada-sm89`` after installation::

    python examples/profile_usage.py --profile optimized
    python examples/profile_usage.py --profile lossless
"""

from __future__ import annotations

import argparse

from engine.profiles import PROFILES, build_engine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--profile", choices=PROFILES, default="optimized")
    parser.add_argument("--prompt", default="Explain paged KV caching in one sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-blocks", type=int, default=None)
    args = parser.parse_args()

    engine = build_engine(args.model, profile=args.profile, num_blocks=args.num_blocks)
    tokens = engine.generate([args.prompt], max_new_tokens=args.max_new_tokens)[0]
    print(f"profile={args.profile}, KV={engine.kv_cache_dtype}, tokens={len(tokens)}")
    print(engine.loaded_model.tokenizer.decode(tokens, skip_special_tokens=True))


if __name__ == "__main__":
    main()
