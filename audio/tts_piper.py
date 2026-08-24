"""Piper neural TTS backend — a local, offline upgrade for the agent's mouth.

Piper runs an ONNX voice model on CPU (already a project dependency through
onnxruntime), so spoken output no longer depends on venue Wi-Fi and sounds far
more natural than the pyttsx3 system voice, which remains the automatic
fallback. Voice models live in ``assets/tts/`` next to the other bundled
model/calibration artifacts.

Fetch a voice ahead of a demo (one small download, ~60 MB):

    python -m audio.tts_piper --download

The runtime never downloads: if no model is present the Speaker chain simply
falls back to pyttsx3 and prints the fetch hint once.
"""
from __future__ import annotations

import importlib.util
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_VOICE = "en_US-amy-medium"
_HF_BASE = ("https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/"
            "{lang}/{locale}/{speaker}/{quality}/{name}")
_VOICE_PARTS = re.compile(
    r"^(?P<locale>[a-z]{2}_[A-Z]{2})-(?P<speaker>[a-z]+)-(?P<quality>"
    r"x_low|low|medium|high)$")
_DOWNLOAD_TIMEOUT = 30.0


def voices_dir() -> Path:
    """Bundled voice directory (created on demand, gitignore-friendly)."""
    root = Path(__file__).resolve().parents[1]
    path = root / "assets" / "tts"
    return path


def available_voices() -> list[Path]:
    """Every complete voice (model + config sidecar) currently bundled."""
    directory = voices_dir()
    if not directory.is_dir():
        return []
    return sorted(
        model for model in directory.glob("*.onnx")
        if model.with_suffix(".onnx.json").is_file())


def resolve_voice(name: str | None = None) -> Path | None:
    """Find a bundled voice by exact name or stem, else any available one."""
    voices = available_voices()
    if name:
        wanted = name if name.endswith(".onnx") else f"{name}.onnx"
        for model in voices:
            if model.name == wanted:
                return model
        return None
    return voices[0] if voices else None


def dependency_available() -> bool:
    """True when the piper package is importable without importing it."""
    try:
        return importlib.util.find_spec("piper") is not None
    except (ImportError, ValueError):
        return False


def _voice_urls(name: str) -> tuple[str, str] | None:
    match = _VOICE_PARTS.match(name)
    if match is None:
        return None
    locale, speaker, quality = match.group("locale", "speaker", "quality")
    lang = locale.split("_")[0]
    base = _HF_BASE.format(lang=lang, locale=locale, speaker=speaker,
                           quality=quality, name=name)
    return base + ".onnx", base + ".onnx.json"


def _fetch(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "graphify-demo"})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT) as response:
        total = 0
        with dest.open("wb") as out:
            while True:
                block = response.read(1 << 16)
                if not block:
                    break
                total += len(block)
                out.write(block)
    print(f"[tts/piper] fetched {dest.name} ({total // (1 << 10)} KiB, "
          f"{time.monotonic() - started:.1f}s)")


def download_voice(name: str = DEFAULT_VOICE) -> Path | None:
    """Download one voice model + config into assets/tts (offline afterwards)."""
    urls = _voice_urls(name)
    if urls is None:
        print(f"[tts/piper] unrecognized voice name '{name}' "
              f"(expected e.g. {DEFAULT_VOICE})")
        return None
    directory = voices_dir()
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / f"{name}.onnx"
    config_path = directory / f"{name}.onnx.json"
    try:
        _fetch(urls[0], model_path)
        _fetch(urls[1], config_path)
    except (urllib.error.URLError, OSError) as exc:
        detail = str(exc).strip()[:200]
        print(f"[tts/piper] download failed: {detail}")
        for leftover in (model_path, config_path):
            leftover.unlink(missing_ok=True)
        return None
    return model_path


class PiperBackend:
    """Thin wrapper: load one voice once, stream synthesized speech chunks."""

    def __init__(self, model_path: Path, volume: float = 1.0,
                 length_scale: float = 1.0):
        self.model_path = model_path
        self.volume = float(volume)
        self.length_scale = float(length_scale)

    def load(self):
        """Load the model (blocking; call on the worker thread). Returns voice."""
        from piper import PiperVoice, SynthesisConfig

        self._config = SynthesisConfig(volume=self.volume,
                                       length_scale=self.length_scale)
        started = time.monotonic()
        voice = PiperVoice.load(self.model_path)
        print(f"[tts/piper] loaded {self.model_path.name} "
              f"in {time.monotonic() - started:.1f}s")
        return voice

    def synthesize(self, voice, text: str):
        """Yield (float samples, sample_rate) chunks for one sentence."""
        for chunk in voice.synthesize(text, syn_config=self._config):
            yield chunk.audio_float_array, chunk.sample_rate


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m audio.tts_piper [--download NAME|--list]``."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--list" in argv:
        found = available_voices()
        if found:
            for model in found:
                print(model.name)
        else:
            print(f"no voices bundled in {voices_dir()}")
        return 0
    if "--download" in argv:
        index = argv.index("--download")
        name = argv[index + 1] if index + 1 < len(argv) else DEFAULT_VOICE
        return 0 if download_voice(name) is not None else 1
    print(__doc__.strip().splitlines()[-1])
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
