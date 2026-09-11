"""BMH GR00T inference client package.

``PolicyClient`` (ZMQ transport) is imported lazily so that the pure-Python
``bmh.groot_client.layout`` module stays importable without the transport
dependencies (``pyzmq``, ``msgpack_numpy``) — e.g. for unit tests on a laptop.
"""

from typing import Any

__all__ = ["PolicyClient"]


def __getattr__(name: str) -> Any:
    if name == "PolicyClient":
        from .server_client import PolicyClient

        return PolicyClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
