#!/usr/bin/env python3
"""Build a Jetson-local FP16 TensorRT engine and bind it to metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path,
                        default=Path("assets/skin_models/vit_skin.onnx"))
    parser.add_argument("--engine", type=Path,
                        default=Path("runtime-models/vit_skin_orin_fp16.engine"))
    parser.add_argument("--workspace-mib", type=int, default=2048)
    args = parser.parse_args()
    if platform.machine().lower() not in {"aarch64", "arm64"}:
        raise SystemExit("TensorRT engines must be built on the target Jetson")
    trtexec = shutil.which("trtexec")
    if not trtexec:
        raise SystemExit("trtexec is unavailable; install/activate the JetPack TensorRT tools")
    onnx_metadata_path = Path(str(args.onnx) + ".json")
    if not args.onnx.is_file() or not onnx_metadata_path.is_file():
        raise SystemExit("ONNX model and its metadata are required")
    metadata = json.loads(onnx_metadata_path.read_text(encoding="utf-8"))
    if _sha256(args.onnx) != metadata.get("onnx_sha256"):
        raise SystemExit("ONNX hash does not match export metadata")
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    command = [
        trtexec,
        f"--onnx={args.onnx.resolve()}",
        f"--saveEngine={args.engine.resolve()}",
        "--fp16",
        f"--memPoolSize=workspace:{max(256, args.workspace_mib)}MiB",
        "--skipInference",
    ]
    subprocess.run(command, check=True)
    try:
        import tensorrt as trt
        trt_version = trt.__version__
    except Exception:
        trt_version = "unknown"
    jetpack = os.environ.get("JETPACK_VERSION", "record-on-device")
    metadata.update({
        "precision": "fp16",
        "engine_sha256": _sha256(args.engine),
        "tensorrt_version": trt_version,
        "jetpack_version": jetpack,
        "platform": platform.platform(),
    })
    engine_metadata = Path(str(args.engine) + ".json")
    engine_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Built {args.engine}")
    print(f"Metadata {engine_metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
