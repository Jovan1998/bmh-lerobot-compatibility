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

import cv2
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

# Maps GR00T video modality key (as trained — bare `front` / `left_wrist` /
# `right_wrist`) to the corresponding key in `bi_so_follower.get_observation()`,
# which prefixes each per-arm camera with `left_` / `right_`. That's why the
# `left_left_wrist` / `right_right_wrist` source names look doubled — that's
# the literal observation key, not a typo.
BIMANUAL_CAMERA_KEYS = {
    "front": "left_front",
    "left_wrist": "left_left_wrist",
    "right_wrist": "right_right_wrist",
}


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

    The state-modality key convention is `left_single_arm` (5,) /
    `left_gripper` (1,) / `right_single_arm` (5,) / `right_gripper` (1,),
    mirroring how eval_so100.py splits a single arm into `single_arm` +
    `gripper`, with `left_` / `right_` prefixes for the bimanual layout.
    The action chunk returned by the server is expected to use the same
    modality keys; `validate_modality()` confirms this at startup against
    `PolicyClient.get_modality_config()` and raises a clear error if the
    trained model uses a different layout.
    """

    STATE_KEYS = ("left_single_arm", "left_gripper", "right_single_arm", "right_gripper")

    def __init__(self, policy_client: PolicyClient, jpeg_quality: int):
        self.policy = policy_client
        self.jpeg_quality = jpeg_quality

    def obs_to_policy_inputs(self, obs: dict[str, Any]) -> dict:
        model_obs: dict[str, Any] = {}

        model_obs["video"] = {
            server_key: obs[obs_key]
            for server_key, obs_key in BIMANUAL_CAMERA_KEYS.items()
        }

        left_state = np.array([obs[f"left_{k}"] for k in SO_JOINT_NAMES], dtype=np.float32)
        right_state = np.array([obs[f"right_{k}"] for k in SO_JOINT_NAMES], dtype=np.float32)
        model_obs["state"] = {
            "left_single_arm": left_state[:5],
            "left_gripper": left_state[5:6],
            "right_single_arm": right_state[:5],
            "right_gripper": right_state[5:6],
        }

        model_obs["language"] = {"annotation.human.task_description": obs["lang"]}

        model_obs = recursive_add_extra_dim(model_obs)
        model_obs = recursive_add_extra_dim(model_obs)

        # JPEG-encode each camera frame in place. The runner-side proxy decodes
        # bytes back to (1, 1, H, W, 3) uint8 RGB; if it ever receives a raw
        # array instead it passes it through, so this is a safe one-sided rollout.
        for cam_key, arr in model_obs["video"].items():
            frame = arr[0, 0]  # (H, W, 3) RGB uint8
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(
                ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            )
            if not ok:
                raise RuntimeError(f"JPEG encode failed for camera {cam_key!r}")
            model_obs["video"][cam_key] = buf.tobytes()
        total = sum(len(v) for v in model_obs["video"].values())
        logger.info(
            "video sent: %d cams, %.1f KiB total (Q=%d)",
            len(model_obs["video"]), total / 1024, self.jpeg_quality,
        )
        return model_obs

    def decode_action_chunk(self, chunk: dict, t: int) -> dict[str, float]:
        left_arm = chunk["left_single_arm"][0][t]      # (5,)
        left_gripper = chunk["left_gripper"][0][t]  # (1,)
        right_arm = chunk["right_single_arm"][0][t]    # (5,)
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
    jpeg_quality: int = 85
    api_token: str = ""


@draccus.wrap()
def main(cfg: BmhInferenceConfig) -> None:
    init_logging()
    logger.info(pformat(asdict(cfg)))

    if not cfg.lang_instruction.strip():
        raise SystemExit("--lang_instruction must not be empty.")
    if not 1 <= cfg.jpeg_quality <= 100:
        raise SystemExit("--jpeg_quality must be in [1, 100].")

    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    logger.info("Robot connected: %s", robot.name)

    client = PolicyClient(
        host=cfg.policy_host,
        port=cfg.policy_port,
        timeout_ms=cfg.timeout_ms,
        api_token=cfg.api_token or None,
    )
    if not client.ping():
        robot.disconnect()
        raise SystemExit(
            f"Cannot reach GR00T policy server at {cfg.policy_host}:{cfg.policy_port}"
        )
    logger.info("Connected to GR00T server at %s:%s", cfg.policy_host, cfg.policy_port)

    adapter = BiSoBimanualAdapter(client, jpeg_quality=cfg.jpeg_quality)
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
