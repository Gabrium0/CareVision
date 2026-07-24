#!/usr/bin/env python3
"""Run the skin VLM over a folder of images and report hits, misses, refusals.

The live pipeline only ever sees whatever the camera happens to catch, which
makes it a poor way to answer "what can this actually recognize?". This bench
feeds prepared images through the *same* client, model, prompt and validator the
`skin_vision` module uses for a close-up, so the answer is the product's answer.

Refusals are results, not failures: the schema validator rejects low-confidence
or malformed responses on purpose, and an image the model declines to call is
exactly the kind of limit worth showing alongside the successes.

    python tools/skin_bench.py demo-images/ --out bench.html --embed-images
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import json
from pathlib import Path
import sys
import time

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.env import nvidia_api_key  # noqa: E402
from integrations.nvidia_vlm import NvidiaVLMClient, NvidiaVLMError  # noqa: E402
from modules.skin_vision import (  # noqa: E402
    _SCHEMA_TEXT, _extract_json, _response_format, _schema_contract,
    validate_analysis,
)

SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
MODEL = "meta/llama-3.2-11b-vision-instruct"

# Mirrors SkinVision._prompt(stage="closeup"): a supplied condition photo is
# precisely the "user-provided close-up" case.
TASK = ("Inspect this user-provided close-up for visible skin changes. "
        "Be conservative and non-diagnostic.")
PROMPT = TASK + "\n\n" + _SCHEMA_TEXT + "\nJSON contract:\n" + _schema_contract("closeup")


def screen(client: NvidiaVLMClient, path: Path, min_confidence: float) -> dict:
    """Return one row describing what the model made of a single image."""
    row = {"image": path.name, "status": "error", "error": None,
           "finding_present": None, "visible_features": [], "body_region": None,
           "possible_conditions": [], "image_quality": None, "confidence": None,
           "latency_ms": None}
    frame = cv2.imread(str(path))
    if frame is None:
        row["error"] = "unreadable image"
        return row
    started = time.monotonic()
    try:
        content = client.request(PROMPT, [client.encode(frame)], max_tokens=700,
                                 response_format=_response_format("closeup"))
    except NvidiaVLMError as exc:
        row["error"] = str(exc)
        return row
    finally:
        row["latency_ms"] = round((time.monotonic() - started) * 1000)
    try:
        analysis = validate_analysis(_extract_json(content), min_confidence,
                                     allow_facial_cues=False)
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        # The guardrail rejected it — a limit worth showing, not a crash.
        row["status"] = "rejected"
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row
    row.update(status="ok", finding_present=analysis.finding_present,
               visible_features=list(analysis.visible_features),
               body_region=analysis.body_region,
               possible_conditions=list(analysis.possible_conditions),
               image_quality=analysis.image_quality,
               confidence=round(float(analysis.confidence), 3))
    return row


def _verdict(row: dict) -> str:
    if row["status"] == "ok":
        return "finding" if row["finding_present"] else "no finding"
    return row["status"]


def render_html(rows: list[dict], embedded: dict[str, str]) -> str:
    """Build a self-contained report; readable across a room."""
    generated = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cells = []
    for row in rows:
        thumb = (f'<img src="{embedded[row["image"]]}" alt="">'
                 if row["image"] in embedded else "")
        detail = (", ".join(row["visible_features"]) or "—") if row["status"] == "ok" \
            else html.escape(str(row["error"] or ""))
        conditions = ", ".join(row["possible_conditions"]) or "—"
        cells.append(f"""
      <tr class="{row['status']}">
        <td class="thumb">{thumb}</td>
        <td>{html.escape(row['image'])}</td>
        <td class="verdict">{html.escape(_verdict(row))}</td>
        <td>{html.escape(detail)}</td>
        <td>{html.escape(conditions)}</td>
        <td>{'' if row['confidence'] is None else row['confidence']}</td>
      </tr>""")
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Skin VLM bench</title><style>
 body {{ background:#0f1113; color:#f2f4f7; margin:0; padding:4vh 4vw;
   font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }}
 h1 {{ font-size:clamp(22px,3vw,38px); margin:0 0 .3em; }}
 p.meta {{ color:#7d8590; margin:0 0 2em; }}
 table {{ border-collapse:collapse; width:100%; font-size:clamp(14px,1.4vw,20px); }}
 th,td {{ text-align:left; padding:.7em .8em; border-bottom:1px solid #23272d;
   vertical-align:middle; }}
 th {{ color:#7d8590; font-weight:600; }}
 td.thumb img {{ height:9vh; max-height:110px; border-radius:6px; display:block; }}
 td.verdict {{ font-weight:700; white-space:nowrap; }}
 tr.ok td.verdict {{ color:#7fd0ff; }}
 tr.rejected td.verdict {{ color:#ffcc66; }}
 tr.error td.verdict {{ color:#ff8080; }}
</style></head><body>
<h1>Skin VLM bench &mdash; {MODEL}</h1>
<p class="meta">{len(rows)} images &middot; generated {generated} &middot;
 non-diagnostic screening output; refusals and rejections are expected results.</p>
<table><thead><tr><th></th><th>Image</th><th>Verdict</th>
 <th>Visible features / reason</th><th>Possible conditions</th><th>Conf.</th>
</tr></thead><tbody>{''.join(cells)}
</tbody></table></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", type=Path,
                        help="directory of images (or a single image file)")
    parser.add_argument("--out", type=Path, help="write an HTML report here")
    parser.add_argument("--json", type=Path, dest="json_out",
                        help="write raw rows here")
    parser.add_argument("--min-confidence", type=float, default=0.35,
                        help="validator threshold, matching SkinVision (default 0.35)")
    parser.add_argument("--embed-images", action="store_true",
                        help="inline the images in the HTML report; off by "
                             "default so no media is written into an artifact")
    args = parser.parse_args()

    key = nvidia_api_key()
    if not key:
        print("no NVIDIA API key configured; set it in .env first", file=sys.stderr)
        return 2

    paths = ([args.images] if args.images.is_file() else
             sorted(p for p in args.images.rglob("*") if p.suffix.lower() in SUFFIXES))
    if not paths:
        print(f"no images found under {args.images}", file=sys.stderr)
        return 2

    client = NvidiaVLMClient(key, ENDPOINT, MODEL)
    rows, embedded = [], {}
    for path in paths:
        row = screen(client, path, args.min_confidence)
        rows.append(row)
        print(f"{row['image'][:38]:<40} {_verdict(row):<12}"
              f" {(', '.join(row['visible_features']) or row['error'] or '')[:60]}")
        if args.embed_images and args.out:
            thumb = cv2.imread(str(path))
            if thumb is not None:
                embedded[row["image"]] = ("data:image/jpeg;base64," + base64.b64encode(
                    client.encode(thumb, max_image_dim=320)).decode())

    counts: dict[str, int] = {}
    for row in rows:
        counts[_verdict(row)] = counts.get(_verdict(row), 0) + 1
    print("\n" + "  ".join(f"{name}: {count}" for name, count in sorted(counts.items())))

    if args.out:
        args.out.write_text(render_html(rows, embedded), encoding="utf-8")
        print(f"report: {args.out}")
    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"json:   {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
