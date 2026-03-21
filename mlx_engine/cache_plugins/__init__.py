"""Boundary-detector plugin system for the disk KV cache.

At import time, every module in this package that exposes a
``find_stable_boundary(prompt_tokens, tokenizer) -> Optional[int]``
function is auto-discovered and registered.

``find_boundary`` tries each registered detector in discovery order and
returns the first non-None result, so more specific plugins shadow the
generic ChatML fallback in ``cache_wrapper._find_system_prompt_boundary``.

To add a plugin: drop a .py file here with a ``find_stable_boundary``
function — no other registration needed.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Callable, List, Optional

_detectors: List[Callable] = []

for _, _name, __ in pkgutil.iter_modules([str(Path(__file__).parent)]):
    _mod = importlib.import_module(f"mlx_engine.cache_plugins.{_name}")
    if hasattr(_mod, "find_stable_boundary"):
        _detectors.append(_mod.find_stable_boundary)


def find_boundary(prompt_tokens: list, tokenizer) -> Optional[int]:
    """Return the first non-None result from all registered detectors."""
    for detector in _detectors:
        result = detector(prompt_tokens, tokenizer)
        if result is not None:
            return result
    return None
