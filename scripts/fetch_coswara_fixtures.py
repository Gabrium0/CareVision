"""Fetch and minimize the CC BY 4.0 Coswara cough benchmark fixtures."""
from __future__ import annotations

import hashlib
import io
import json
import urllib.parse
import urllib.request
import wave
from pathlib import Path
from time import sleep

import numpy as np


DATASET = "szzs1693/coswara-data"
API = "https://datasets-server.huggingface.co/rows"
ORIGINAL_REPOSITORY = "https://github.com/iiscleap/Coswara-Data"
ORIGINAL_COMMIT = "4942c97e31de7180a93d17f2e7530a9c543cfd50"
MIRROR_REVISION = "d61663c0c8098226f17b9628f6f24b5b5d2f85d9"
LICENSE = "https://creativecommons.org/licenses/by/4.0/"
OUTPUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "cough"
QUOTAS = {"cough-heavy": 6, "cough-shallow": 6,
          "breathing-deep": 2, "breathing-shallow": 2,
          "counting-fast": 2, "counting-normal": 2, "vowel-a": 4}


def _rows(offset: int) -> list[dict]:
    query = urllib.parse.urlencode({"dataset": DATASET, "config": "audio",
                                   "split": "train", "offset": offset,
                                   "length": 100})
    with urllib.request.urlopen(f"{API}?{query}", timeout=60) as response:
        return json.load(response)["rows"]


def _select() -> list[dict]:
    selected, used = [], set()
    counts = {key: 0 for key in QUOTAS}
    for offset in range(0, 2000, 100):
        for item in _rows(offset):
            row = item["row"]
            kind = row.get("audio_type")
            participant = row.get("participant_id")
            if kind not in QUOTAS or counts[kind] >= QUOTAS[kind]:
                continue
            if int(row.get("quality_score", -1)) < 1 or participant in used:
                continue
            audio = row.get("audio") or []
            if not audio or not audio[0].get("src"):
                continue
            selected.append({"row_idx": item["row_idx"], "kind": kind,
                             "participant": participant, "url": audio[0]["src"],
                             "quality": int(row["quality_score"])})
            used.add(participant)
            counts[kind] += 1
        if counts == QUOTAS:
            return selected
    raise RuntimeError(f"not enough quality fixtures: {counts}")


def _decode_wav(payload: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(payload), "rb") as source:
        channels, width, rate = (source.getnchannels(), source.getsampwidth(),
                                 source.getframerate())
        raw = source.readframes(source.getnframes())
    if width == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported PCM width: {width}")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


def _prepare(samples: np.ndarray, rate: int) -> np.ndarray:
    if rate != 16000:
        target = max(1, round(len(samples) * 16000 / rate))
        samples = np.interp(np.linspace(0, len(samples) - 1, target),
                            np.arange(len(samples)), samples).astype(np.float32)
    frame = 320
    active = [index for index in range(0, len(samples), frame)
              if np.max(np.abs(samples[index:index + frame]), initial=0) >= .01]
    if active:
        start = max(0, active[0] - 1600)
        end = min(len(samples), active[-1] + frame + 1600)
        samples = samples[start:end]
    return np.clip(samples[:4 * 16000], -1, 1)


def _write_wav(path: Path, samples: np.ndarray) -> None:
    pcm = np.round(samples * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(pcm)


def _download(url: str) -> bytes:
    error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001 - bounded acquisition retry
            error = exc
            sleep(attempt + 1)
    raise RuntimeError("fixture download failed after three attempts") from error


def main() -> None:
    """Download selected rows and write privacy-minimized WAVs plus attribution."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected = _select()
    counts, manifest = {}, []
    for item in selected:
        kind = item["kind"]
        counts[kind] = counts.get(kind, 0) + 1
        filename = f"{kind.replace('-', '_')}_{counts[kind]:02}.wav"
        target = OUTPUT / filename
        if not target.exists():
            samples, rate = _decode_wav(_download(item["url"]))
            prepared = _prepare(samples, rate)
            _write_wav(target, prepared)
        manifest.append({
            "file": filename,
            "category": kind,
            "expected": "cough" if kind.startswith("cough") else "non_cough",
            "quality_score": item["quality"],
            "source_identifier": f"audio/train row {item['row_idx']}",
            "participant_hash": hashlib.sha256(
                item["participant"].encode()).hexdigest()[:12],
            "transform": "mono 16 kHz PCM16; threshold silence trim; max 4 seconds; no gain",
        })
        print(filename)
    attribution = {
        "title": "Coswara respiratory audio benchmark subset",
        "original_repository": ORIGINAL_REPOSITORY,
        "original_commit": ORIGINAL_COMMIT,
        "download_mirror": f"https://huggingface.co/datasets/{DATASET}",
        "mirror_revision": MIRROR_REVISION,
        "license": "Creative Commons Attribution 4.0 International",
        "license_url": LICENSE,
        "citation": ("Bhattacharya et al., Coswara: A respiratory sounds and symptoms "
                     "dataset for remote screening of SARS-CoV-2 infection, "
                     "Scientific Data 10, 397 (2023)."),
        "privacy": "No demographics, symptoms, diagnoses, transcripts, or raw identifiers retained.",
        "fixtures": manifest,
    }
    (OUTPUT / "ATTRIBUTION.json").write_text(
        json.dumps(attribution, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
