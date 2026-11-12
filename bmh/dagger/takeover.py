"""Take-over of a DAgger session: is the leader where the follower is, and the blend into teleop.

**Alignment.** Before the operator steers, the leader arms should be where the follower
is. The reference is the pose the follower was last *commanded* (the hold pose), not the
pose it measures: a gripper squeezing an object never reaches its goal, and aligning to
the measured position would relax the grip the moment teleoperation starts. `alignment`
reports ``leader - hold`` per joint; `AlignGate` says when that stayed inside the
tolerance long enough to start by itself.

**Blend.** `TakeoverBlend` makes the start continuous by construction: the first teleop
action is shifted by ``hold - leader`` (so it *is* the hold pose) and that shift fades out
with a smoothstep. Started from inside the tolerance the fade takes ``blend_s``; forced
from further away it takes longer, so the follower never closes the gap faster than
``max_speed_deg_s``. Once the fade is over the leader's action passes through untouched.

Units: joints in degrees, grippers in 0..100 (one scale for the speed limit, as in
``bmh/policy_client/motion.py``). stdlib only, so it is unit-testable anywhere.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from bmh.motion.player import smoothstep

GRIPPER_SUFFIX = "gripper.pos"


@dataclass(frozen=True)
class AlignState:
    """Leader vs. hold pose: ``diff[key] = leader - hold``, and whether every joint is in tolerance."""

    diff: dict[str, float]
    aligned: bool
    max_abs: float


def alignment(
    leader: Mapping[str, float],
    hold: Mapping[str, float],
    arm_tol_deg: float,
    gripper_tol: float,
) -> AlignState:
    """Compare the leader's action with the pose the follower holds.

    Covers the keys both have. Keys ending in ``gripper.pos`` are checked against
    ``gripper_tol`` (0..100 scale), every other key against ``arm_tol_deg``. No common key
    means not aligned: there is nothing to judge by.
    """
    diff = {key: float(leader[key]) - float(value) for key, value in hold.items() if key in leader}
    aligned = bool(diff) and all(
        abs(delta) <= (gripper_tol if key.endswith(GRIPPER_SUFFIX) else arm_tol_deg)
        for key, delta in diff.items()
    )
    return AlignState(diff=diff, aligned=aligned, max_abs=max((abs(d) for d in diff.values()), default=0.0))


class AlignGate:
    """True once the leader stayed aligned for ``hold_s`` without a gap."""

    def __init__(self, hold_s: float):
        """Require ``hold_s`` seconds of unbroken alignment."""
        self._hold_s = hold_s
        self._since: float | None = None

    def update(self, aligned: bool, now: float) -> bool:
        """Feed this tick's verdict; ``now`` is any monotonic clock in seconds."""
        if not aligned:
            self._since = None
            return False
        if self._since is None:
            self._since = now
        return now - self._since >= self._hold_s

    def reset(self) -> None:
        """Start counting afresh."""
        self._since = None


class TakeoverBlend:
    """Fades the offset between the hold pose and the leader's pose at the start of a take."""

    def __init__(
        self,
        hold: Mapping[str, float],
        leader_start: Mapping[str, float],
        blend_s: float,
        max_speed_deg_s: float = 45.0,
    ):
        """Plan the fade.

        Args:
            hold: The pose the follower holds (what the first action must equal).
            leader_start: The leader's action at the moment the arms connect.
            blend_s: Shortest fade, in seconds.
            max_speed_deg_s: The fade is stretched so the largest offset closes no faster
                than this on average.
        """
        self._offset = {
            key: float(hold[key]) - float(value) for key, value in leader_start.items() if key in hold
        }
        self._hold = {key: float(hold[key]) for key in self._offset}
        self._leader_start = {key: float(leader_start[key]) for key in self._offset}
        gap = max((abs(o) for o in self._offset.values()), default=0.0)
        self.duration_s = max(blend_s, gap / max_speed_deg_s) if max_speed_deg_s > 0 else blend_s

    def apply(self, leader: Mapping[str, float], t_s: float) -> dict[str, float]:
        """The action to send ``t_s`` seconds into the take for the leader's ``leader``.

        Covers exactly ``leader``'s keys. At ``t_s <= 0`` with the leader still at its
        start pose this is the hold pose, bit for bit; from ``duration_s`` on it is
        ``dict(leader)``, with no arithmetic on the positions.
        """
        s = 1.0 if self.duration_s <= 0.0 else smoothstep(t_s / self.duration_s)
        if s >= 1.0:
            return dict(leader)
        out: dict[str, float] = {}
        for key, value in leader.items():
            if key not in self._offset:
                out[key] = value
                continue
            # hold + (how far the leader moved since the start) - (the share already faded).
            moved = float(value) - self._leader_start[key]
            out[key] = self._hold[key] + moved - s * self._offset[key]
        return out
