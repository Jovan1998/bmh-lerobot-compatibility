#!/usr/bin/env python
"""
BMH-101 GR00T N inference client.

Runs the closed-loop control on the follower Pi:

    bi_so_follower.get_observation()
        ↓
    BiSoBimanualAdapter.obs_to_policy_inputs(...)   # cameras + grouped joint state + language
        ↓
    PolicyClient.get_action(...)  →  action chunk  (ZMQ to remote GR00T server)
        ↓
    BiSoBimanualAdapter.decode_action_chunk(...)
        ↓
    bi_so_follower.send_action(...)

Mirrors the spirit of Isaac-GR00T `gr00t/eval/real_robot/SO100/eval_so100.py` but
adapted for the BMH-101 bimanual robot (bi_so_follower in the BMH LeRobot fork).

The joint layout is not hardcoded: the state/action groups are derived from the
robot's `action_features` with the same rule the platform uses at training time
(see bmh/groot_client/layout.py). For the 7-DoF + head BMH-101 that is
`left_single_arm` / `left_gripper` / `left_head` / `right_single_arm` /
`right_gripper` (16 dims); the head servos ride the left arm's bus and are driven
like any other `.pos` key.
"""

import logging
import queue
import threading
import time
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import Any

import cv2
import draccus
import numpy as np

from bmh.groot_client import PolicyClient
from bmh.groot_client.layout import JointGroup, group_joint_names, pack_state, unpack_action

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

logger = logging.getLogger(__name__)


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


def _blend_actions(old: dict[str, float], new: dict[str, float], alpha: float) -> dict[str, float]:
    """Convex blend of two joint-command dicts: ``(1 - alpha) * old + alpha * new``.

    Used to crossfade the seam between a retiring action chunk and a freshly
    arrived one. ``alpha`` near 0 keeps the old command, near 1 takes the new.
    Iterates over `new`'s keys; both dicts carry the same SO-joint key set.
    """
    return {k: (1.0 - alpha) * old[k] + alpha * new[k] for k in new}


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
    back into a `{left_*, right_*}` `.pos` dict that `bi_so_follower.send_action()`
    accepts (one key per joint in `joint_names`, head servos included).

    The state/action modality groups are derived from `joint_names` (the robot's
    `action_features` keys, in order) by `bmh.groot_client.layout.group_joint_names`
    — the same rule the platform backend uses to build the training modality, so a
    checkpoint trained on a dataset recorded with this robot advertises exactly
    `self.state_keys`. For the 7-DoF + head BMH-101: `left_single_arm` (6,) /
    `left_gripper` (1,) / `left_head` (2,) / `right_single_arm` (6,) /
    `right_gripper` (1,). `validate_modality()` confirms the server's layout at
    startup against `PolicyClient.get_modality_config()` and raises a clear error
    if the trained model uses a different one.
    """

    def __init__(self, policy_client: PolicyClient, jpeg_quality: int, joint_names: list[str]):
        self.policy = policy_client
        self.jpeg_quality = jpeg_quality
        self.groups: list[JointGroup] = group_joint_names(joint_names)
        self.state_keys: tuple[str, ...] = tuple(g.key for g in self.groups)
        logger.info(
            "Joint layout: %d joints in %d groups: %s",
            len(joint_names),
            len(self.groups),
            {g.key: len(g.names) for g in self.groups},
        )

    def obs_to_policy_inputs(self, obs: dict[str, Any]) -> dict:
        model_obs: dict[str, Any] = {}

        model_obs["video"] = {
            server_key: obs[obs_key] for server_key, obs_key in BIMANUAL_CAMERA_KEYS.items()
        }

        model_obs["state"] = pack_state(obs, self.groups)

        model_obs["language"] = {"annotation.human.task_description": obs["lang"]}

        model_obs = recursive_add_extra_dim(model_obs)
        model_obs = recursive_add_extra_dim(model_obs)

        # JPEG-encode each camera frame in place. The runner-side proxy decodes
        # bytes back to (1, 1, H, W, 3) uint8 RGB; if it ever receives a raw
        # array instead it passes it through, so this is a safe one-sided rollout.
        for cam_key, arr in model_obs["video"].items():
            frame = arr[0, 0]  # (H, W, 3) RGB uint8
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not ok:
                raise RuntimeError(f"JPEG encode failed for camera {cam_key!r}")
            model_obs["video"][cam_key] = buf.tobytes()
        total = sum(len(v) for v in model_obs["video"].values())
        logger.info(
            "video sent: %d cams, %.1f KiB total (Q=%d)",
            len(model_obs["video"]),
            total / 1024,
            self.jpeg_quality,
        )
        return model_obs

    def decode_action_chunk(self, chunk: dict, t: int) -> dict[str, float]:
        # One `<side>_<motor>.pos` key per joint, head included. bi_so_follower
        # routes them by prefix and so_follower writes every `.pos` key to its
        # bus, so the head is driven without further plumbing.
        return unpack_action(chunk, self.groups, t)

    def get_action(self, obs: dict) -> list[dict[str, float]]:
        model_input = self.obs_to_policy_inputs(obs)
        tic = time.perf_counter()
        action_chunk, _info = self.policy.get_action(model_input)
        rtt_ms = (time.perf_counter() - tic) * 1000.0
        logger.info("policy round-trip: %.1f ms", rtt_ms)

        any_key = next(iter(action_chunk.keys()))
        horizon = action_chunk[any_key].shape[1]
        return [self.decode_action_chunk(action_chunk, t) for t in range(horizon)]

    def validate_modality(self, modality_cfg: dict) -> None:
        """Compare the server's advertised modality keys against this robot's layout.

        State keys are compared order-sensitively against `self.state_keys` (derived
        from the robot's joint names); video keys are compared as a set against
        `BIMANUAL_CAMERA_KEYS`. Raises with a clear diff if the trained model uses a
        different layout — this catches the common "model trained on a different
        robot layout" mismatch before we ship garbage action commands to the motors.
        """
        if not modality_cfg:
            logger.warning("Server returned an empty modality config; skipping modality validation.")
            return

        state_cfg = modality_cfg.get("state")
        if state_cfg is None:
            logger.warning(
                "Server modality config has no 'state' entry; skipping modality validation. Server keys: %s",
                list(modality_cfg.keys()),
            )
            return

        server_state_keys = tuple(_modality_keys(state_cfg))
        if server_state_keys != self.state_keys:
            raise SystemExit(
                "GR00T model state modality keys do not match this robot's joint layout.\n"
                f"  This robot expects: {self.state_keys}\n"
                f"  Server reports:     {server_state_keys}\n"
                "Retrain the model on data recorded with this robot layout (e.g. the "
                "7-DoF + head BMH-101 needs a `left_head` group), or run the client on "
                "the robot the model was trained for."
            )

        video_cfg = modality_cfg.get("video")
        if video_cfg is not None:
            server_video_keys = set(_modality_keys(video_cfg))
            expected_video_keys = set(BIMANUAL_CAMERA_KEYS)
            if server_video_keys != expected_video_keys:
                raise SystemExit(
                    "GR00T model video modality keys do not match this client's cameras.\n"
                    f"  This client expects: {sorted(expected_video_keys)}\n"
                    f"  Server reports:      {sorted(server_video_keys)}\n"
                    "Retrain the model with these camera keys, or extend "
                    "`BIMANUAL_CAMERA_KEYS` to map them."
                )
        logger.info(
            "Modality check passed (state keys: %s, video keys: %s).",
            self.state_keys,
            tuple(BIMANUAL_CAMERA_KEYS),
        )


def _modality_keys(section: Any) -> list[str]:
    """`modality_keys` of one server modality section (ModalityConfig object or dict)."""
    if isinstance(section, dict):
        keys = section.get("modality_keys")
    else:
        keys = getattr(section, "modality_keys", None)
    return [str(k) for k in (keys or [])]


def _inference_worker(
    adapter: BiSoBimanualAdapter,
    obs_queue: "queue.Queue[dict[str, Any] | None]",
    chunk_queue: "queue.Queue[list[dict[str, float]] | None]",
    action_horizon: int,
    stop_event: threading.Event,
) -> None:
    """Owns the PolicyClient (ZMQ REQ is single-thread).

    Pulls one observation at a time from `obs_queue`, runs inference, and
    pushes the truncated action chunk to `chunk_queue`. A `None` observation
    is the shutdown sentinel. On failure, pushes `None` to signal the main
    thread, then continues to honor `stop_event`.
    """
    while not stop_event.is_set():
        try:
            obs = obs_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if obs is None:
            break
        try:
            chunk = adapter.get_action(obs)[:action_horizon]
            chunk_queue.put(chunk)
        except Exception:
            logger.exception("inference worker: get_action failed")
            chunk_queue.put(None)


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
    jpeg_quality: int = 95
    api_token: str = ""
    # Fire the next inference request once this many actions have been consumed
    # from the freshly swapped-in chunk. Kept small so we re-sync to the latest
    # observation often. The in-flight guard means the effective request cadence
    # is max(refetch_after, server-latency-in-frames). Must be in
    # [1, action_horizon).
    refetch_after: int = 3
    # Number of leading actions of a freshly arrived chunk to *crossfade* with the
    # tail of the old chunk instead of switching to them outright. At a chunk swap
    # the new chunk's first kept action and the old chunk's next action describe the
    # same instant in time, but they come from two different policy passes, so they
    # rarely agree exactly — sending the new one cold produces a one-frame jump (the
    # seam jitter). Blending the two over the first `blend_frames` actions with a
    # linear ramp (mostly-old → mostly-new) turns that step into a short wash-in.
    # If the old chunk has fewer than `blend_frames` actions left (or the new chunk
    # is shorter), the blend is clamped to whatever overlap exists. Set to 0 to
    # disable blending and switch hard. Must be in [0, action_horizon).
    blend_frames: int = 3


@draccus.wrap()
def main(cfg: BmhInferenceConfig) -> None:
    init_logging()
    logger.info(pformat(asdict(cfg)))

    if not cfg.lang_instruction.strip():
        raise SystemExit("--lang_instruction must not be empty.")
    if not 1 <= cfg.jpeg_quality <= 100:
        raise SystemExit("--jpeg_quality must be in [1, 100].")
    if not 1 <= cfg.refetch_after < cfg.action_horizon:
        raise SystemExit(
            f"--refetch_after must be in [1, action_horizon={cfg.action_horizon}); got {cfg.refetch_after}."
        )
    if not 0 <= cfg.blend_frames < cfg.action_horizon:
        raise SystemExit(
            f"--blend_frames must be in [0, action_horizon={cfg.action_horizon}); got {cfg.blend_frames}."
        )

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
        raise SystemExit(f"Cannot reach GR00T policy server at {cfg.policy_host}:{cfg.policy_port}")
    logger.info("Connected to GR00T server at %s:%s", cfg.policy_host, cfg.policy_port)

    # bi_so_follower.action_features lists every `.pos` key in bus order with the
    # left_/right_ prefix — for the BMH-101 with `--robot.left_arm_config.with_head=true`
    # that is the same 16 names the training dataset's `features` carry.
    adapter = BiSoBimanualAdapter(
        client,
        jpeg_quality=cfg.jpeg_quality,
        joint_names=list(robot.action_features.keys()),
    )
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

    # Bootstrap: one synchronous inference on the main thread to seed the
    # first chunk. After this, the worker thread is the sole owner of the
    # PolicyClient.
    bootstrap_obs = robot.get_observation()
    bootstrap_obs["lang"] = cfg.lang_instruction
    current_chunk: list[dict[str, float]] = adapter.get_action(bootstrap_obs)[: cfg.action_horizon]
    idx = 0
    inflight = False
    last_action: dict[str, float] | None = None
    chunk_exhausted_at: float | None = None
    # Monotonic per-tick frame counter (one tick == one control period at `fps`).
    # Used to measure how many frames a request spent in flight so we can drop
    # the chunk's stale leading actions on arrival.
    frame_counter = 0
    fire_frame = 0
    # Actions left in the old chunk when the in-flight request fired. The robot
    # can advance through at most this many actions before it runs dry and holds
    # position, so it bounds how many leading actions we may skip on arrival.
    remaining_at_fire = 0
    # Actions consumed from the current chunk since it was swapped in; drives the
    # `refetch_after` re-fire trigger below.
    consumed_since_swap = 0

    obs_queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=1)
    chunk_queue: queue.Queue[list[dict[str, float]] | None] = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    worker = threading.Thread(
        target=_inference_worker,
        args=(adapter, obs_queue, chunk_queue, cfg.action_horizon, stop_event),
        name="bmh-inference-worker",
        daemon=True,
    )
    worker.start()

    try:
        while True:
            tick_start = time.perf_counter()
            frame_counter += 1

            # 1. Try to swap in a freshly arrived chunk, time-aligned to the
            #    robot's current position. The chunk's action[0] is the policy's
            #    response to the observation captured when the request fired,
            #    `elapsed` frames ago — but the robot has kept moving along the
            #    old chunk since then. Dropping the first `elapsed` actions
            #    makes the new chunk pick up from where the robot actually is,
            #    instead of snapping it back to the fire-time pose (the cause of
            #    the back-and-forth jitter).
            #
            #    The skip is bounded by `remaining_at_fire`: if the old chunk ran
            #    out before this one arrived (a full gap), the robot was *holding
            #    position*, not advancing, for those extra frames — so they must
            #    not be skipped. Without this bound a gap would drop almost the
            #    whole new chunk, leaving fewer than `refetch_after` actions, so
            #    the re-fire trigger below could never arm again and the robot
            #    would hold forever.
            try:
                new_chunk = chunk_queue.get_nowait()
            except queue.Empty:
                pass
            else:
                if new_chunk is None:
                    raise RuntimeError("inference worker reported failure; aborting")
                elapsed = frame_counter - fire_frame
                # Drop the `elapsed` frames the robot already advanced along the
                # old chunk (capped at what it could actually consume), so the new
                # chunk picks up from where the robot *is*, not from the fire-time
                # pose (the cause of the back-and-forth jitter).
                #
                # Cap: keep at least `refetch_after` actions in the chunk. The
                # re-fire trigger needs `consumed_since_swap >= refetch_after`, and
                # that counter only advances while there are actions left to send.
                # If a high-latency swap trimmed the chunk below `refetch_after`,
                # the trigger could never arm again and the robot would hold
                # position forever (the bug seen when ping exceeds the action
                # horizon). Capping here guarantees the loop always re-fires.
                drop = min(elapsed, remaining_at_fire)
                drop = min(drop, max(len(new_chunk) - cfg.refetch_after, 0))

                # Crossfade the seam. `current_chunk[idx:]` are the old chunk's
                # next actions and `new_chunk[drop:]` are the new chunk's first
                # kept actions; after the time-align above these describe the same
                # instants but come from different policy passes, so a cold switch
                # leaves a one-frame jump. Blend the first `blend_frames` of them
                # with a mostly-old → mostly-new ramp. The blend length is clamped
                # to the overlap: if the old chunk has fewer actions left than
                # `blend_frames` (or already ran dry — `leftover_old` empty, robot
                # holding position) we interpolate over only the frames available.
                leftover_old = current_chunk[idx:]
                new_kept = new_chunk[drop:]
                blend_len = min(cfg.blend_frames, len(leftover_old), len(new_kept))
                for k in range(blend_len):
                    alpha = (k + 1) / (blend_len + 1)
                    new_kept[k] = _blend_actions(leftover_old[k], new_kept[k], alpha)

                logger.info(
                    "chunk swap: %d actions arrived after %d frames, dropping %d "
                    "stale actions, %d kept, %d blended "
                    "(remaining_at_fire=%d, leftover=%d)",
                    len(new_chunk),
                    elapsed,
                    drop,
                    len(new_kept),
                    blend_len,
                    remaining_at_fire,
                    len(leftover_old),
                )
                current_chunk = new_kept
                idx = 0
                inflight = False
                consumed_since_swap = 0
                chunk_exhausted_at = None

            # 2. Send an action (or hold position if we ran out).
            if idx < len(current_chunk):
                action = current_chunk[idx]
                robot.send_action(action)
                last_action = action
                idx += 1
                consumed_since_swap += 1
                chunk_exhausted_at = None
            else:
                if chunk_exhausted_at is None:
                    chunk_exhausted_at = tick_start
                gap_ms = (tick_start - chunk_exhausted_at) * 1000.0
                if last_action is not None:
                    robot.send_action(last_action)
                logger.warning("late chunk: gap=%.1f ms (holding position)", gap_ms)

            # 3. Fire the next request once we've consumed `refetch_after`
            #    actions from the current chunk and nothing is already in flight.
            #    Record the fire frame and how many actions are still queued so
            #    the swap above can measure round-trip latency in frames and cap
            #    the stale-lead trim at what the robot can actually consume.
            if not inflight and consumed_since_swap >= cfg.refetch_after:
                t_start = time.perf_counter()
                obs = robot.get_observation()
                obs["lang"] = cfg.lang_instruction
                obs_queue.put(obs)
                inflight = True
                fire_frame = frame_counter
                remaining_at_fire = max(len(current_chunk) - idx, 0)
                logger.info(
                    "inference fired at frame=%d (consumed_since_swap=%d), "
                    "remaining_at_fire=%d, fire_time=%.6f",
                    frame_counter,
                    consumed_since_swap,
                    remaining_at_fire,
                    time.perf_counter() - t_start,
                )

            # 4. Sleep to next tick.
            sleep = period - (time.perf_counter() - tick_start)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        logger.info("Shutting down inference loop…")
    finally:
        stop_event.set()
        try:
            obs_queue.put_nowait(None)
        except queue.Full:
            pass
        worker.join(timeout=2.0)
        if worker.is_alive():
            logger.warning("inference worker did not exit within 2 s")
        try:
            robot.disconnect()
        except Exception as e:
            logger.error("Error disconnecting robot: %s", e)


if __name__ == "__main__":
    main()
