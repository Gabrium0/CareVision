#!/usr/bin/env python3
"""Export the pinned reference skin classifier to ONNX plus metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MODEL = "LaurianeMD/vit-skin-disease"
REVISION = "1b4fccab2c8b83bf6964e394c40f0dcdd21b1d1c"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("assets/skin_models/vit_skin.onnx"))
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("runtime-models/huggingface"))
    parser.add_argument("--local-files-only", action="store_true",
                        help="fail rather than downloading the pinned model")
    args = parser.parse_args()

    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained(
        MODEL, revision=REVISION, local_files_only=args.local_files_only,
        cache_dir=str(args.cache_dir))
    model = AutoModelForImageClassification.from_pretrained(
        MODEL, revision=REVISION, local_files_only=args.local_files_only,
        cache_dir=str(args.cache_dir))
    model.eval().cpu()

    class LogitsOnly(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values):
            return self.inner(pixel_values=pixel_values).logits

    export_model = LogitsOnly(model)
    size = processor.size
    height = int(size.get("height", size.get("shortest_edge", 224)))
    width = int(size.get("width", size.get("shortest_edge", 224)))
    sample = torch.zeros((1, 3, height, width), dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        export_model,
        (sample,),
        str(args.output),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )
    metadata = {
        "schema_version": 1,
        "model": MODEL,
        "model_revision": REVISION,
        "onnx_sha256": _sha256(args.output),
        "opset": args.opset,
        "input_name": "pixel_values",
        "output_name": "logits",
        "output_dtype": "float32",
        "input_shape": [1, 3, height, width],
        "id2label": {str(k): str(v) for k, v in model.config.id2label.items()},
        "processor": MODEL,
        "processor_config": processor.to_dict(),
    }
    metadata_path = Path(str(args.output) + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {args.output}")
    print(f"Metadata {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
