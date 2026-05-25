#!/usr/bin/env python
"""
BMH-101 GR00T N inference client.

Runs the closed-loop control on the follower Pi:

    bi_so_follower.get_observation()
        ↓
    BiSoBimanualAdapter.obs_to_policy_inputs(...)   # cameras + 12-joint state + language
        ↓
    PolicyClient.get_action(...)  →  action chunk  (ZMQ to remote GR00T server)
        ↓
    BiSoBimanualAdapter.decode_action_chunk(...)
        ↓
    bi_so_follower.send_action(...)

Mirrors the spirit of Isaac-GR00T `gr00t/eval/real_robot/SO100/eval_so100.py` but
adapted for the BMH-101 bimanual robot (bi_so_follower in the BMH LeRobot fork).
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import Any

import draccus
import numpy as np

# Importing the robot configs ensures draccus CLI registration is populated.
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    make_robot_from_config,
    so_follower,
)
from lerobot.utils.utils import init_logging

from bmh.groot_client import PolicyClient

logger = logging.getLogger(__name__)


# Joint order matches `bi_so_follower.get_observation()` keys, dropping the
# `left_` / `right_` arm prefix. This is the per-arm SO-100/101 joint order
# (matches eval_so100.py `robot_state_keys`).
SO_JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

# Camera keys exposed by `bi_so_follower.get_observation()` for the BMH-101
# camera wiring used at recording time (front + left_wrist attached to the
# left arm; right_wrist attached to the right arm). The bi_so_follower wrapper
# prefixes each per-arm camera with `left_` / `right_`, which is why the
# `left_left_wrist` / `right_right_wrist` names look doubled — that's the
# literal observation key, not a typo.
BIMANUAL_CAMERA_KEYS = ["left_front", "left_left_wrist", "right_right_wrist"]


def recursive_add_extra_dim(obs: dict) -> dict:
    """Recursively prepend a size-1 leading dim. Called twice to inject (B=1, T=1)."""
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            obs[key] = val[np.newaxis, ...]
        elif isinstance(val, dict):
            obs[key] = recursive_add_extra_dim(val)
        else:
            obs[key] = [val]
    return obs


class BiSoBimanualAdapter:
    """
    Bimanual sibling of eval_so100.py's `So100Adapter`. Packs raw `bi_so_follower`
    observations into the GR00T VLA input format and decodes returned action chunks
    back into a 12-key `{left_*, right_*}` dict that `bi_so_follower.send_action()`
    accepts.

    The state-modality key convention is `left_arm` (5,) / `left_gripper` (1,) /
    `right_arm` (5,) / `right_gripper` (1,), mirroring how eval_so100.py splits
    a single arm into `single_arm` + `gripper`. The action chunk returned by
    the server is expected to use the same modality keys; `validate_modality()`
    confirms this at startup against `PolicyClient.get_modality_config()` and
    raises a clear error if the trained model uses a different layout.
    """

    STATE_KEYS = ("left_arm", "left_gripper", "right_arm", "right_gripper")

    def __init__(self, policy_client: PolicyClient):
        self.policy = policy_client

    def obs_to_policy_inputs(self, obs: dict[str, Any]) -> dict:
        model_obs: dict[str, Any] = {}

        model_obs["video"] = {k: obs[k] for k in BIMANUAL_CAMERA_KEYS}

        left_state = np.array([obs[f"left_{k}"] for k in SO_JOINT_NAMES], dtype=np.float32)
        right_state = np.array([obs[f"right_{k}"] for k in SO_JOINT_NAMES], dtype=np.float32)
        model_obs["state"] = {
            "left_arm": left_state[:5],
            "left_gripper": left_state[5:6],
            "right_arm": right_state[:5],
            "right_gripper": right_state[5:6],
        }

        model_obs["language"] = {"annotation.human.task_description": obs["lang"]}

        model_obs = recursive_add_extra_dim(model_obs)
        model_obs = recursive_add_extra_dim(model_obs)
        return model_obs

    def decode_action_chunk(self, chunk: dict, t: int) -> dict[str, float]:
        left_arm = chunk["left_arm"][0][t]      # (5,)
        left_gripper = chunk["left_gripper"][0][t]  # (1,)
        right_arm = chunk["right_arm"][0][t]    # (5,)
        right_gripper = chunk["right_gripper"][0][t]  # (1,)

        left = np.concatenate([left_arm, left_gripper], axis=0)    # (6,)
        right = np.concatenate([right_arm, right_gripper], axis=0)  # (6,)

        action: dict[str, float] = {}
        for i, joint in enumerate(SO_JOINT_NAMES):
            action[f"left_{joint}"] = float(left[i])
            action[f"right_{joint}"] = float(right[i])
        return action

    def get_action(self, obs: dict) -> list[dict[str, float]]:
        model_input = self.obs_to_policy_inputs(obs)
        action_chunk, _info = self.policy.get_action(model_input)

        any_key = next(iter(action_chunk.keys()))
        horizon = action_chunk[any_key].shape[1]
        return [self.decode_action_chunk(action_chunk, t) for t in range(horizon)]

    def validate_modality(self, modality_cfg: dict) -> None:
        """Compare server's advertised state modality keys against `STATE_KEYS`.

        Raises with a clear diff if the trained model uses a different layout —
        this catches the common "model trained with different state grouping"
        mismatch before we ship garbage action commands to the motors.
        """
        if not modality_cfg:
            logger.warning(
                "Server returned an empty modality config; skipping modality validation."
            )
            return

        state_cfg = modality_cfg.get("state")
        if state_cfg is None:
            logger.warning(
                "Server modality config has no 'state' entry; skipping modality validation. "
                "Server keys: %s",
                list(modality_cfg.keys()),
            )
            return

        server_state_keys = tuple(getattr(state_cfg, "modality_keys", ()) or ())
        if server_state_keys != self.STATE_KEYS:
            raise SystemExit(
                "GR00T model state modality keys do not match this bimanual adapter.\n"
                f"  This client expects: {self.STATE_KEYS}\n"
                f"  Server reports:      {server_state_keys}\n"
                "Retrain the model with these keys, or extend `BiSoBimanualAdapter` "
                "to remap them."
            )
        logger.info("Modality check passed (state keys: %s).", self.STATE_KEYS)


@dataclass
class BmhInferenceConfig:
    """CLI configuration for `bmh-groot-client`."""

    robot: RobotConfig
    policy_host: str = "localhost"
    policy_port: int = 5555
    lang_instruction: str = ""
    action_horizon: int = 8
    fps: int = 30
    timeout_ms: int = 15000


@draccus.wrap()
def main(cfg: BmhInferenceConfig) -> None:
    init_logging()
    logger.info(pformat(asdict(cfg)))

    if not cfg.lang_instruction.strip():
        raise SystemExit("--lang_instruction must not be empty.")

    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    logger.info("Robot connected: %s", robot.name)

    client = PolicyClient(
        host=cfg.policy_host,
        port=cfg.policy_port,
        timeout_ms=cfg.timeout_ms,
    )
    if not client.ping():
        robot.disconnect()
        raise SystemExit(
            f"Cannot reach GR00T policy server at {cfg.policy_host}:{cfg.policy_port}"
        )
    logger.info("Connected to GR00T server at %s:%s", cfg.policy_host, cfg.policy_port)

    adapter = BiSoBimanualAdapter(client)
    try:
        modality_cfg = client.get_modality_config()
        adapter.validate_modality(modality_cfg)
    except SystemExit:
        robot.disconnect()
        raise
    except Exception as e:
        # get_modality_config is best-effort — log and continue. If keys are wrong,
        # the first send_action call will fail loudly anyway.
        logger.warning("Could not fetch modality config from server: %s", e)

    logger.info('Running inference with instruction: "%s"', cfg.lang_instruction)
    period = 1.0 / cfg.fps

    try:
        while True:
            obs = robot.get_observation()
            obs["lang"] = cfg.lang_instruction

            actions = adapter.get_action(obs)
            for i, action_dict in enumerate(actions[: cfg.action_horizon]):
                tic = time.perf_counter()
                robot.send_action(action_dict)
                logger.info("action[%d] sent", i)
                sleep = period - (time.perf_counter() - tic)
                if sleep > 0:
                    time.sleep(sleep)
    except KeyboardInterrupt:
        logger.info("Shutting down inference loop…")
    finally:
        try:
            robot.disconnect()
        except Exception as e:
            logger.error("Error disconnecting robot: %s", e)


if __name__ == "__main__":
    main()
