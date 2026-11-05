"""BMH policy client package — the robot side of the BMH policy protocol.

Talks to any BMH policy server (the GR00T or SmolVLA runner, or anything that
speaks the protocol documented in bmh-app `gpu-runners/bmh-runner-common/PROTOCOL.md`).

``PolicyClient`` (ZMQ transport) is imported lazily so that the pure-Python
``bmh.policy_client.layout`` module stays importable without the transport
dependencies (``pyzmq``, ``msgpack_numpy``) — e.g. for unit tests on a laptop.
"""

from typing import Any

__all__ = ["PolicyClient"]


def __getattr__(name: str) -> Any:
    if name == "PolicyClient":
        from .server_client import PolicyClient

        return PolicyClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
