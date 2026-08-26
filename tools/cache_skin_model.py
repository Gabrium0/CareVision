#!/usr/bin/env python3
"""Provision the pinned PyTorch fallback model into a deployment cache."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL = "LaurianeMD/vit-skin-disease"
REVISION = "1b4fccab2c8b83bf6964e394c40f0dcdd21b1d1c"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("runtime-models/huggingface"))
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=MODEL,
        revision=REVISION,
        cache_dir=str(args.cache_dir),
        allow_patterns=["config.json", "preprocessor_config.json", "*.safetensors"],
    )
    print(f"Cached pinned skin model at {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
