"""Module registry: modules self-register via decorator; the pipeline
instantiates whichever ones the config enables. Adding a detector means
dropping a file in modules/ and listing it in config/modules.yaml.
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import Type

_REGISTRY: dict[str, Type] = {}


def register(name: str):
    """Class decorator: @register("fall_detection")."""
    def deco(cls):
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return deco


def discover(package: str = "modules") -> None:
    """Import every submodule of `modules/` so @register decorators run."""
    pkg = importlib.import_module(package)
    for mod in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        importlib.import_module(mod.name)


def build_enabled(config: dict) -> list:
    """Instantiate modules enabled in config, passing their params."""
    instances = []
    for name, opts in config.get("modules", {}).items():
        opts = opts or {}
        if not opts.get("enabled", True):
            continue
        cls = _REGISTRY.get(name)
        if cls is None:
            print(f"[registry] WARNING: module '{name}' in config but not found; skipping")
            continue
        params = {k: v for k, v in opts.items() if k != "enabled"}
        instances.append(cls(**params))
    return instances


def all_registered() -> dict[str, Type]:
    """Return a copy of the name -> module-class registry."""
    return dict(_REGISTRY)
