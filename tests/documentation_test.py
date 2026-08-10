"""Dependency-free checks for maintained repository documentation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
DOC_FILES = tuple(
    sorted(
        {
            *ROOT.glob("*.md"),
            *(ROOT / "docs").glob("*.md"),
            *ROOT.glob("*/AGENTS.md"),
            *(ROOT / "tests" / "fixtures").glob("**/*.md"),
        }
    )
)
LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
REPLAY_PATTERN = re.compile(r"\breplay:([a-z][a-z0-9_]*)\b")


def _local_target(raw_target: str) -> str | None:
    """Return a Markdown link's local path, excluding anchors and URLs."""
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    else:
        target = target.split(maxsplit=1)[0]
    target = unquote(target).split("#", 1)[0]
    if not target:
        return None
    parsed = urlparse(target)
    if parsed.scheme or target.startswith("//"):
        return None
    return target


def test_documentation_contract_files_exist() -> None:
    """Keep the canonical agent and operator entry points available."""
    required = {
        ROOT / "AGENTS.md",
        ROOT / "docs" / "README.md",
        ROOT / "docs" / "OPERATIONS.md",
        ROOT / "docs" / "ARCHITECTURE.md",
        ROOT / "docs" / "EXTENDING.md",
        ROOT / "docs" / "AI_DEVELOPMENT.md",
        ROOT / "config" / "modules.yaml",
        ROOT / "config" / "alerts.yaml",
        ROOT / "config" / "replay_scenarios.json",
        ROOT / ".env.example",
    }
    missing = sorted(path.relative_to(ROOT).as_posix() for path in required if not path.is_file())
    assert not missing, f"Missing documentation contract files: {missing}"


def test_repository_relative_markdown_links_resolve() -> None:
    """Reject broken local links in maintained documentation."""
    broken: list[str] = []
    for document in DOC_FILES:
        text = document.read_text(encoding="utf-8")
        for match in LINK_PATTERN.finditer(text):
            target = _local_target(match.group(1))
            if target is None:
                continue
            resolved = (document.parent / Path(target.replace("\\", "/"))).resolve()
            if not resolved.exists():
                broken.append(
                    f"{document.relative_to(ROOT).as_posix()} -> {target}"
                )
    assert not broken, "Broken local Markdown links:\n" + "\n".join(sorted(broken))


def test_documented_replay_scenarios_exist() -> None:
    """Keep replay examples synchronized with the scenario catalogue."""
    scenarios = json.loads(
        (ROOT / "config" / "replay_scenarios.json").read_text(encoding="utf-8")
    )
    missing: list[str] = []
    for document in DOC_FILES:
        for name in REPLAY_PATTERN.findall(document.read_text(encoding="utf-8")):
            if name not in scenarios:
                missing.append(f"{document.relative_to(ROOT).as_posix()} -> replay:{name}")
    assert not missing, "Unknown documented replay scenarios:\n" + "\n".join(sorted(missing))
