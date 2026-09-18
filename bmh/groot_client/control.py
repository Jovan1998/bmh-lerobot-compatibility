"""Live inference target for the BMH-101 GR00T client.

The controller-app hands ``bmh-groot-client`` its *target* — policy server host /
port / API token plus the language instruction — through a small JSON file
(``~/.cache/bmh-101/inference-control.json``) that it replaces atomically (tmp +
rename) whenever the operator applies a change. The client ``os.stat``s the file
once per control tick and re-reads it only when it changed, so a new prompt or
server is picked up within one tick and applied to the *next* inference request —
no restart, no robot reconnect. Same mechanism as the teleop group lock
(``lerobot.teleoperators.bi_so_network_leader.group_lock.LockFileWatcher``).

Every target carries a monotonically increasing ``seq`` so that re-applying the
same values after a rejected switch is a new request, and so the app can tell
"requested" from "applied". The client answers each decided ``seq`` with one
``BMH_TARGET {json}`` log line (see :func:`format_target_status`) that the app
parses back into its status; the API token is never echoed.

File contract (all keys required except ``api_token``):

.. code-block:: json

    {"seq": 3, "policy_host": "1.2.3.4", "policy_port": 5555,
     "lang_instruction": "pick up the cube", "api_token": "..."}

A target can also be *idle* — "no server, no prompt": the client closes its policy
connection, stops moving and holds position until a real target arrives. That is how
the Physical Agent tab starts the client before any skill is live. Only ``seq`` is
read from an idle document, every other key is ignored:

.. code-block:: json

    {"seq": 4, "idle": true}

Idle is never expressed by deleting the file — a missing file means "keep the
current target" (see :class:`ControlFileWatcher`).

stdlib only (no zmq / cv2 / hardware imports) so it is unit-testable anywhere.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Prefix of the one-line JSON status the client logs per decided target. The
# controller-app greps for it in the process output (see
# bmh-101-os/controller-app/src/lib/inference.controller.ts).
STATUS_PREFIX = "BMH_TARGET "

# Signature reported for a missing control file (never collides with a real stat).
_MISSING_SIGNATURE = (-1, -1, -1)


@dataclass(frozen=True)
class InferenceTarget:
    """One requested inference target: which server to ask, with which instruction.

    Attributes:
        seq: Monotonic request number assigned by the writer (0 for CLI-only runs).
        policy_host: GR00T policy server host.
        policy_port: GR00T policy server port.
        lang_instruction: Language instruction sent with every request.
        api_token: Per-request API token (empty = none). Hidden from ``repr``.
        idle: ``True`` for "no server, no prompt" — host / instruction / token are
            empty and the port is 0. Build one with :meth:`make_idle`.
    """

    seq: int
    policy_host: str
    policy_port: int
    lang_instruction: str
    api_token: str = field(default="", repr=False)
    idle: bool = False

    @classmethod
    def make_idle(cls, seq: int) -> InferenceTarget:
        """The idle target for ``seq``: no server to ask, nothing to do."""
        return cls(seq=seq, policy_host="", policy_port=0, lang_instruction="", idle=True)

    @property
    def endpoint(self) -> tuple[str, int, str]:
        """``(host, port, token)`` — what a ``PolicyClient`` is bound to.

        Equal across prompt-only changes, so the worker can tell "just swap the
        instruction" from "reconnect to another server".
        """
        return (self.policy_host, self.policy_port, self.api_token)


@dataclass(frozen=True)
class Hold:
    """Worker reply for a rejected target: hold position until a new ``seq`` arrives."""

    seq: int
    error: str


@dataclass(frozen=True)
class Idle:
    """Worker reply for an accepted idle target: stay still until a new ``seq`` arrives."""

    seq: int


def parse_target(data: object) -> InferenceTarget:
    """Validate a decoded control-file document.

    Args:
        data: Result of ``json.loads`` on the control file.

    Returns:
        The target, with host / instruction / token whitespace-stripped — or the idle
        target for ``seq`` when the document says ``"idle": true`` (its other keys
        are then ignored).

    Raises:
        ValueError: If a required key is missing or has the wrong type / range.
            Bools are rejected where ints are expected, and ``idle`` must be a bool.
    """
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object, got {type(data).__name__}")
    seq = data.get("seq")
    if type(seq) is not int or seq < 0:
        raise ValueError(f"'seq' must be a non-negative integer, got {seq!r}")
    idle = data.get("idle", False)
    if not isinstance(idle, bool):
        raise ValueError(f"'idle' must be a boolean, got {idle!r}")
    if idle:
        return InferenceTarget.make_idle(seq)
    host = data.get("policy_host")
    if not isinstance(host, str) or not host.strip():
        raise ValueError("'policy_host' must be a non-empty string")
    port = data.get("policy_port")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError(f"'policy_port' must be an integer in [1, 65535], got {port!r}")
    lang = data.get("lang_instruction")
    if not isinstance(lang, str) or not lang.strip():
        raise ValueError("'lang_instruction' must be a non-empty string")
    token = data.get("api_token", "")
    if not isinstance(token, str):
        raise ValueError(f"'api_token' must be a string, got {type(token).__name__}")
    return InferenceTarget(
        seq=seq,
        policy_host=host.strip(),
        policy_port=port,
        lang_instruction=lang.strip(),
        api_token=token.strip(),
    )


class ControlFileWatcher:
    """Change detector for the controller-app's inference control file.

    Costs one ``os.stat`` per :meth:`poll`; the file is only opened and parsed when its
    inode / mtime / size changed (i.e. once per apply). Unlike the lock-file watcher a
    *missing* file carries no meaning ("there is no unset target"): it reports ``None``
    and the caller keeps its current target; a re-created file is picked up again.
    """

    def __init__(self, path: Path | str):
        self._path = Path(path)
        self._signature: tuple[int, int, int] | None = None

    @property
    def path(self) -> Path:
        return self._path

    def read_initial(self) -> InferenceTarget | None:
        """Startup read of the file the app wrote before spawning the client.

        Primes the change detector, so a following :meth:`poll` reports ``None`` unless
        the file changed after this call — the startup read is not a "change".

        Returns:
            The target, or ``None`` if the file does not exist.

        Raises:
            ValueError: If the file exists but is unreadable or invalid — the app is
                the only writer, so this is a contract bug worth failing loudly on.
        """
        signature = self._stat()
        self._signature = signature
        if signature == _MISSING_SIGNATURE:
            return None
        try:
            return self._read()
        except OSError as e:
            raise ValueError(f"cannot read control file {self._path}: {e}") from e

    def poll(self) -> InferenceTarget | None:
        """Return the new target if the file changed since the last call, else ``None``.

        A malformed or vanished file is logged once per file version and ignored
        (``None``), leaving the caller's current target untouched.
        """
        signature = self._stat()
        if signature == self._signature:
            return None
        self._signature = signature
        if signature == _MISSING_SIGNATURE:
            return None
        try:
            return self._read()
        except (OSError, ValueError) as e:
            logger.warning("Ignoring unreadable control file %s: %s", self._path, e)
            return None

    def _stat(self) -> tuple[int, int, int]:
        try:
            st = os.stat(self._path)
        except FileNotFoundError:
            return _MISSING_SIGNATURE
        # st_ino changes on every tmp+rename, so even same-size same-mtime rewrites register.
        return (st.st_ino, st.st_mtime_ns, st.st_size)

    def _read(self) -> InferenceTarget:
        # json.JSONDecodeError is a ValueError, so callers see one failure type.
        return parse_target(json.loads(self._path.read_text()))


def format_target_status(target: InferenceTarget, accepted: bool, error: str | None) -> str:
    """One-line ``BMH_TARGET {json}`` status for the app to parse. Never includes the token.

    Args:
        target: The target that was decided on (accepted or rejected). An idle target
            reports ``"idle": true`` with an empty host / instruction and port 0.
        accepted: Whether the client now runs against ``target``.
        error: Reason for a rejection, ``None`` when accepted.
    """
    return STATUS_PREFIX + json.dumps(
        {
            "seq": target.seq,
            "accepted": accepted,
            "error": error,
            "idle": target.idle,
            "policy_host": target.policy_host,
            "policy_port": target.policy_port,
            "lang_instruction": target.lang_instruction,
        }
    )
