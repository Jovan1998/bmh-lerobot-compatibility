"""Joint-group layout for the BMH-101 GR00T client.

Derives the GR00T state/action modality groups from a robot's action-feature names
with the *same rule* the platform backend applies when it builds a training manifest
(``bmh-app/backend/src/common/utils/groot-modality.ts::buildGrootModality``). The two
implementations must stay in sync: a checkpoint trained by the platform advertises
the group keys produced there, and ``validate_modality()`` in
``bmh/scripts/bmh_groot_client.py`` compares them against the keys produced here.

Rule, applied to each joint name in column order:

* strip the ``.pos`` suffix; ``side`` is the text before the first ``_`` (must be
  ``left`` or ``right``), ``motor`` is the rest;
* ``kind`` is ``gripper`` if ``motor == "gripper"``, ``head`` if ``motor`` starts
  with ``head_``, otherwise ``single_arm``;
* the group key is ``f"{side}_{kind}"``. Groups keep first-appearance order and each
  must be one contiguous run of columns.

For ``bi_so_follower`` with ``with_head=true`` on the left arm this yields
``left_single_arm`` (6), ``left_gripper`` (1), ``left_head`` (2),
``right_single_arm`` (6), ``right_gripper`` (1) — the 16-dim BMH-101 layout.

stdlib + numpy only (no hardware imports) so it is unit-testable anywhere.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_POS_SUFFIX = ".pos"
_SIDES = ("left", "right")


@dataclass(frozen=True)
class JointGroup:
    """One GR00T modality group: a contiguous run of joint columns.

    Attributes:
        key: Group key as used in the modality config, e.g. ``"left_head"``.
        names: Joint names in column order, e.g.
            ``["left_head_pan.pos", "left_head_tilt.pos"]``.
    """

    key: str
    names: list[str] = field(default_factory=list)


def group_key_for(name: str) -> str:
    """Return the GR00T group key for one joint column name.

    Args:
        name: Joint name as it appears in ``robot.action_features`` /
            ``features["observation.state"].names``, e.g. ``"left_wrist_yaw.pos"``.

    Returns:
        ``"<side>_<kind>"`` with ``kind`` in ``single_arm`` | ``gripper`` | ``head``.

    Raises:
        ValueError: If the name lacks the ``.pos`` suffix or a ``left_``/``right_`` prefix.
    """
    if not name.endswith(_POS_SUFFIX):
        raise ValueError(f"joint name {name!r} has no {_POS_SUFFIX!r} suffix")
    bare = name[: -len(_POS_SUFFIX)]
    side, sep, motor = bare.partition("_")
    if not sep or not motor or side not in _SIDES:
        raise ValueError(f"joint name {name!r} must start with 'left_' or 'right_'")
    if motor == "gripper":
        kind = "gripper"
    elif motor.startswith("head_"):
        kind = "head"
    else:
        kind = "single_arm"
    return f"{side}_{kind}"


def group_joint_names(names: Sequence[str]) -> list[JointGroup]:
    """Split ordered joint names into contiguous GR00T groups.

    Same rule as the backend's ``buildGrootModality`` (see module docstring).

    Args:
        names: Joint column names in vector order.

    Returns:
        Groups in first-appearance order; each carries its joint names in column order.

    Raises:
        ValueError: If ``names`` is empty, a name is malformed, or a group key recurs
            after another group started (non-contiguous layout).
    """
    if not names:
        raise ValueError("no joint names to group")
    groups: list[JointGroup] = []
    seen: set[str] = set()
    for i, name in enumerate(names):
        key = group_key_for(name)
        if groups and groups[-1].key == key:
            groups[-1].names.append(name)
            continue
        if key in seen:
            raise ValueError(f"joint group {key!r} is not contiguous (recurs at column {i}: {name!r})")
        seen.add(key)
        groups.append(JointGroup(key=key, names=[name]))
    return groups


def pack_state(obs: Mapping[str, Any], groups: Sequence[JointGroup]) -> dict[str, np.ndarray]:
    """Pack a robot observation into per-group float32 state vectors.

    Args:
        obs: ``robot.get_observation()`` output (joint name → scalar position).
        groups: Output of :func:`group_joint_names`.

    Returns:
        ``{group.key: float32 array of shape (len(group.names),)}``.
    """
    return {g.key: np.array([obs[name] for name in g.names], dtype=np.float32) for g in groups}


def unpack_action(chunk: Mapping[str, Any], groups: Sequence[JointGroup], t: int) -> dict[str, float]:
    """Inverse of :func:`pack_state` for timestep ``t`` of an action chunk.

    Args:
        chunk: Policy output, ``{group.key: array of shape (B=1, T, dim)}``.
        groups: Output of :func:`group_joint_names`.
        t: Timestep within the chunk.

    Returns:
        ``{joint name: value}`` accepted by ``robot.send_action()``.

    Raises:
        ValueError: If a group's per-step vector width differs from its joint count.
    """
    action: dict[str, float] = {}
    for g in groups:
        values = np.asarray(chunk[g.key])[0][t]
        if values.shape != (len(g.names),):
            raise ValueError(
                f"action chunk {g.key!r} has {values.shape} values at t={t}, expected ({len(g.names)},)"
            )
        for name, value in zip(g.names, values, strict=True):
            action[name] = float(value)
    return action
