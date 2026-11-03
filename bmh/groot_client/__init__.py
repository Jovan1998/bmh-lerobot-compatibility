"""Deprecated alias of :mod:`bmh.policy_client` — kept for one release.

The robot client talks to every BMH policy server (GR00T or SmolVLA runner), so it
moved to ``bmh.policy_client``. ``import bmh.groot_client`` and
``from bmh.groot_client.<module> import …`` keep working: each submodule is the very
same module object as its ``bmh.policy_client`` counterpart.
"""

import importlib
import importlib.abc
import importlib.util
import sys
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


class _LazySubmodules:
    """Finder that serves ``bmh.groot_client.<name>`` from ``bmh.policy_client``.

    Lazy so ``import bmh.groot_client`` stays as light as ``import bmh.policy_client``
    (``server_client`` pulls in pyzmq / msgpack_numpy).
    """

    @staticmethod
    def find_spec(fullname: str, path: object = None, target: object = None) -> Any:
        prefix = f"{__name__}."
        if not fullname.startswith(prefix) or fullname[len(prefix) :] not in _SUBMODULES:
            return None
        module = importlib.import_module(f"bmh.policy_client.{fullname[len(prefix) :]}")
        sys.modules[fullname] = module
        return importlib.util.spec_from_loader(fullname, _AliasLoader(module))


class _AliasLoader(importlib.abc.Loader):
    """Hands back the already-imported ``bmh.policy_client`` module as is."""

    def __init__(self, module: Any) -> None:
        self._module = module

    def create_module(self, spec: Any) -> Any:
        return self._module

    def exec_module(self, module: Any) -> None:
        return None


sys.meta_path.insert(0, _LazySubmodules())

__all__ = ["PolicyClient"]


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        return importlib.import_module(f"bmh.policy_client.{name}")
    return getattr(importlib.import_module("bmh.policy_client"), name)
