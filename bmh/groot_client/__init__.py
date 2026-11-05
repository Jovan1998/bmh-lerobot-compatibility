"""Deprecated alias of :mod:`bmh.policy_client` — kept for one release.

The robot client talks to every BMH policy server (GR00T or SmolVLA runner), so it
moved to ``bmh.policy_client``. ``import bmh.groot_client`` and
``from bmh.groot_client.<module> import …`` keep working: each submodule here is a
shim that puts its ``bmh.policy_client`` counterpart into ``sys.modules`` under the
old name, so both names refer to the very same module object (one copy of every
class, so ``isinstance`` checks and monkeypatches behave the same through either
name). Shims are plain files rather than an import hook, so the real module's
``__spec__`` stays untouched and nothing is added to ``sys.meta_path``.
"""

import importlib
import warnings
from typing import Any

warnings.warn(
    "bmh.groot_client is deprecated; import bmh.policy_client instead.",
    DeprecationWarning,
    stacklevel=2,
)

_SUBMODULES = (
    "control",
    "embodiment_tags",
    "feedback",
    "json_utils",
    "layout",
    "policy_base",
    "server_client",
    "types",
)

__all__ = ["PolicyClient"]


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return importlib.import_module(f"bmh.policy_client.{name}")
    return getattr(importlib.import_module("bmh.policy_client"), name)
