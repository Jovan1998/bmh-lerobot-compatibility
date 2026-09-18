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

The *target* — policy host / port / API token plus the language instruction — comes
from the CLI flags or, when `--control_file` points at the controller-app's JSON
file, from that file. The file is polled every tick so the app can switch prompt
or policy server live, without restarting the client (see
bmh/groot_client/control.py). A switch is validated (ping + modality check) before
it is used; a rejected switch makes the robot hold position until a valid one
arrives. The same happens when a request fails mid-run (e.g. the server was stopped
in the BMH App): the client stays alive, holds position and reports the failure, so
the app can apply another target.

With `--control_file` the client may also run *idle* — no server, no prompt, robot
holding position. It starts that way when neither the file nor `--lang_instruction`
names a target (the controller-app's Physical Agent tab does this), and returns to
it whenever the app writes an idle document.
"""

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import cv2
import draccus
import numpy as np

from bmh.groot_client import PolicyClient
from bmh.groot_client.control import (
    ControlFileWatcher,
    Hold,
    Idle,
    InferenceTarget,
    format_target_status,
)
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

    `policy` may be attached after construction (and swapped later by the
    inference worker when the app switches policy servers).
    """

    def __init__(self, policy_client: PolicyClient | None, jpeg_quality: int, joint_names: list[str]):
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


class PolicyConnectError(Exception):
    """A target's policy server cannot be reached, rejects the token, or fails the modality check."""


def _connect_policy(
    target: InferenceTarget,
    adapter: BiSoBimanualAdapter,
    probe_timeout_ms: int,
    timeout_ms: int,
) -> PolicyClient:
    """Open a validated `PolicyClient` for `target`, or raise `PolicyConnectError`.

    Ping and modality fetch run with the short `probe_timeout_ms` so an unreachable
    host is reported within seconds; the returned client is then switched to the
    regular request `timeout_ms`. A failed probe closes its client before raising,
    so nothing leaks. Used both at startup and by the worker for live switches.
    """
    client = PolicyClient(
        host=target.policy_host,
        port=target.policy_port,
        timeout_ms=probe_timeout_ms,
        api_token=target.api_token or None,
    )
    where = f"{target.policy_host}:{target.policy_port}"
    try:
        # A wrong token makes the proxy answer {"error": ...}, which `ping()` re-raises
        # as RuntimeError instead of returning False — treat both as unreachable.
        try:
            reachable = client.ping()
        except Exception as e:
            raise PolicyConnectError(f"Cannot reach GR00T policy server at {where}: {e}") from e
        if not reachable:
            raise PolicyConnectError(f"Cannot reach GR00T policy server at {where}")
        logger.info("Connected to GR00T server at %s", where)

        try:
            modality_cfg = client.get_modality_config()
        except Exception as e:
            # get_modality_config is best-effort — log and continue. If keys are wrong,
            # the first send_action call will fail loudly anyway.
            logger.warning("Could not fetch modality config from server: %s", e)
        else:
            try:
                adapter.validate_modality(modality_cfg)
            except SystemExit as e:
                # validate_modality raises SystemExit for a clean CLI error. Inside the
                # worker thread that would end the thread silently, so convert it.
                raise PolicyConnectError(str(e)) from None
    except PolicyConnectError:
        client.close()
        raise
    client.set_timeout_ms(timeout_ms)
    return client


def _inference_worker(
    adapter: BiSoBimanualAdapter,
    obs_queue: "queue.Queue[tuple[InferenceTarget, dict[str, Any]] | None]",
    chunk_queue: "queue.Queue[list[dict[str, float]] | Hold | Idle | None]",
    action_horizon: int,
    stop_event: threading.Event,
    active: InferenceTarget,
    connect: Callable[[InferenceTarget], PolicyClient],
) -> None:
    """Owns the PolicyClient (ZMQ REQ is single-thread) and applies target switches.

    Pulls one `(target, observation)` pair at a time from `obs_queue`. `target` is
    the newest target the main loop knows; `active` is the one the current client
    was validated for — or an idle target, in which case `adapter.policy` is `None`.
    A new `target.seq` is decided exactly once — a rejected seq is never re-probed
    on later requests carrying the same seq:

    - idle target → the client is closed and detached, the reply is an `Idle`;
    - same endpoint (prompt-only change) → adopted as is;
    - other endpoint, no client (coming from idle) or a client whose last request
      failed → `connect()` it (ping + modality check). On success the old client is
      closed and replaced; on `PolicyConnectError` the old client stays but is
      *not* used — the reply is a `Hold` and the robot holds position until a new
      seq arrives.

    The instruction sent with each request is always `active.lang_instruction`, so
    a rejected target's prompt never reaches any server. Every decision is logged
    as a `BMH_TARGET {json}` line for the controller-app.

    A failing request (server stopped, network gone) does not end the client: the
    active seq is re-reported as `accepted: false`, the reply is a `Hold`, and the
    next seq re-probes its server even when the endpoint did not change.

    A `None` item is the shutdown sentinel. A `None` *reply* means the worker itself
    crashed; the main thread aborts on it.
    """
    try:
        _serve_requests(adapter, obs_queue, chunk_queue, action_horizon, stop_event, active, connect)
    except Exception:
        logger.exception("inference worker crashed")
        chunk_queue.put(None)


def _serve_requests(
    adapter: BiSoBimanualAdapter,
    obs_queue: "queue.Queue[tuple[InferenceTarget, dict[str, Any]] | None]",
    chunk_queue: "queue.Queue[list[dict[str, float]] | Hold | Idle | None]",
    action_horizon: int,
    stop_event: threading.Event,
    active: InferenceTarget,
    connect: Callable[[InferenceTarget], PolicyClient],
) -> None:
    """Request loop of `_inference_worker` (see there for the decision rules)."""
    decided_seq = active.seq
    rejected: str | None = None
    # Set after a failed request: the client may be talking to a dead server, so the
    # next target is probed even if it names the same endpoint.
    needs_probe = False
    while not stop_event.is_set():
        try:
            item = obs_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if item is None:
            break
        target, obs = item

        if target.seq != decided_seq:
            decided_seq = target.seq
            rejected = None
            if target.idle:
                old_client = adapter.policy
                adapter.policy = None
                if old_client is not None:
                    old_client.close()
                active = target
                needs_probe = False
            elif adapter.policy is None or needs_probe or target.endpoint != active.endpoint:
                try:
                    new_client = connect(target)
                except PolicyConnectError as e:
                    rejected = str(e)
                    logger.error(
                        "target #%d rejected (%s:%d): %s",
                        target.seq,
                        target.policy_host,
                        target.policy_port,
                        e,
                    )
                except Exception as e:
                    # Unexpected — still hold rather than let the thread die silently.
                    rejected = f"{type(e).__name__}: {e}"
                    logger.exception("target #%d: unexpected error while connecting", target.seq)
                else:
                    old_client = adapter.policy
                    adapter.policy = new_client
                    if old_client is not None:
                        old_client.close()
                    active = target
                    needs_probe = False
            else:
                active = target
            if rejected is None and target.idle:
                logger.info("target #%d applied: idle", target.seq)
            elif rejected is None:
                logger.info(
                    'target #%d applied: %s:%d "%s"',
                    target.seq,
                    target.policy_host,
                    target.policy_port,
                    target.lang_instruction,
                )
            logger.info(format_target_status(target, rejected is None, rejected))

        if rejected is not None:
            chunk_queue.put(Hold(seq=target.seq, error=rejected))
            continue
        if active.idle:
            chunk_queue.put(Idle(seq=active.seq))
            continue

        obs["lang"] = active.lang_instruction
        try:
            chunk = adapter.get_action(obs)[:action_horizon]
        except Exception as e:
            # Typically the server was stopped or became unreachable. Hold instead of
            # exiting, and tell the app under the *same* seq so that re-applying the
            # target is a new request.
            logger.exception("inference worker: get_action failed")
            rejected = f"policy request failed: {type(e).__name__}: {e}"
            needs_probe = True
            logger.info(format_target_status(active, False, rejected))
            chunk_queue.put(Hold(seq=active.seq, error=rejected))
        else:
            chunk_queue.put(chunk)


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
    # Path of the controller-app's JSON control file (see bmh/groot_client/control.py).
    # When set, the file is read at startup — overriding --policy_host / --policy_port /
    # --lang_instruction / --api_token — and polled every tick so the app can switch
    # prompt or policy server live, without restarting the client.
    control_file: str | None = None
    # Socket timeout for the ping + modality fetch that validate a (new) policy
    # server, so an unreachable host is reported within seconds instead of
    # `timeout_ms`. Regular requests use `timeout_ms`.
    probe_timeout_ms: int = 5000
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


def _initial_target(
    watcher: ControlFileWatcher | None,
    policy_host: str,
    policy_port: int,
    lang_instruction: str,
    api_token: str,
) -> InferenceTarget:
    """Pick the target the client starts with.

    The controller-app's control file wins when it exists (the app writes it before
    spawning us); else the CLI flags, if they carry an instruction; else the client
    starts idle — but only with a control file, because nothing could ever move an
    idle client without one.

    Raises:
        SystemExit: On an invalid control file, or when there is neither an
            instruction nor a control file.
    """
    if watcher is not None:
        try:
            target = watcher.read_initial()
        except ValueError as e:
            raise SystemExit(f"--control_file {watcher.path}: {e}") from e
        if target is not None:
            logger.info("Initial target #%d read from %s", target.seq, watcher.path)
            return target
        logger.warning("Control file %s does not exist yet; starting from CLI flags.", watcher.path)
    if lang_instruction.strip():
        return InferenceTarget(
            seq=0,
            policy_host=policy_host,
            policy_port=policy_port,
            lang_instruction=lang_instruction.strip(),
            api_token=api_token,
        )
    if watcher is not None:
        return InferenceTarget.make_idle(0)
    raise SystemExit("--lang_instruction must not be empty (starting idle needs --control_file).")


def _run_control_loop(
    robot: Robot,
    adapter: BiSoBimanualAdapter,
    watcher: ControlFileWatcher | None,
    target: InferenceTarget,
    cfg: BmhInferenceConfig,
    connect: Callable[[InferenceTarget], PolicyClient],
) -> None:
    """Closed control loop, from the initial connect to the worker / client teardown.

    The caller owns `robot` and must disconnect it in a `finally`; everything here —
    the startup connect, the bootstrap inference, the loop — may raise.

    Args:
        robot: Connected robot.
        adapter: Adapter without a policy client; one is attached here.
        watcher: Control-file watcher, `None` for a CLI-only run.
        target: Startup target (see `_initial_target`). Idle skips the connect and
            the bootstrap inference: the robot holds position, and nothing is
            requested until the control file names a real target.
        cfg: CLI configuration.
        connect: Opens a validated client for a target, or raises `PolicyConnectError`.

    Raises:
        SystemExit: If the startup target's server is rejected.
        RuntimeError: If the inference worker crashed.
    """
    period = 1.0 / cfg.fps
    current_chunk: list[dict[str, float]] = []
    idx = 0
    inflight = False
    last_action: dict[str, float] | None = None
    chunk_exhausted_at: float | None = None
    # When the `late chunk` warning was last logged for the current gap (it repeats
    # every tick otherwise).
    late_logged_at: float | None = None
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
    # Set when the control file delivered a new target: fire the next request as
    # soon as nothing is in flight instead of waiting for `refetch_after`, so a
    # prompt / server switch takes effect after a single round-trip.
    fire_now = False
    # True while idle, and after the worker rejected a target or lost its server:
    # the robot holds its last pose and nothing is fired until a new seq arrives.
    holding = target.idle

    obs_queue: queue.Queue[tuple[InferenceTarget, dict[str, Any]] | None] = queue.Queue(maxsize=1)
    chunk_queue: queue.Queue[list[dict[str, float]] | Hold | Idle | None] = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    worker: threading.Thread | None = None

    try:
        if target.idle:
            logger.info("No target yet — idle, holding position until the controller-app applies one.")
        else:
            try:
                adapter.policy = connect(target)
            except PolicyConnectError as e:
                raise SystemExit(str(e)) from e
        logger.info(format_target_status(target, True, None))

        if not target.idle:
            logger.info('Running inference with instruction: "%s"', target.lang_instruction)
            # Bootstrap: one synchronous inference on the main thread to seed the
            # first chunk. After this, the worker thread is the sole owner of the
            # PolicyClient.
            bootstrap_obs = robot.get_observation()
            bootstrap_obs["lang"] = target.lang_instruction
            current_chunk = adapter.get_action(bootstrap_obs)[: cfg.action_horizon]

        worker = threading.Thread(
            target=_inference_worker,
            args=(adapter, obs_queue, chunk_queue, cfg.action_horizon, stop_event, target, connect),
            name="bmh-inference-worker",
            daemon=True,
        )
        worker.start()

        while True:
            tick_start = time.perf_counter()
            frame_counter += 1

            # 0. Pick up a new target from the controller-app (one os.stat per
            #    tick; the file is only parsed when it changed). It travels with
            #    the next request; the worker validates and applies it.
            if watcher is not None:
                new_target = watcher.poll()
                if new_target is not None and new_target.seq != target.seq:
                    target = new_target
                    fire_now = True
                    if target.idle:
                        # Stop at once instead of playing out the rest of the chunk;
                        # the worker is told below so it can drop its client.
                        current_chunk = []
                        idx = 0
                        consumed_since_swap = 0
                        chunk_exhausted_at = None
                        holding = True
                        logger.info("target #%d requested: idle — holding position", target.seq)
                    else:
                        logger.info(
                            'target #%d requested: %s:%d "%s"',
                            target.seq,
                            target.policy_host,
                            target.policy_port,
                            target.lang_instruction,
                        )

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
                if isinstance(new_chunk, (Hold, Idle)) or target.idle:
                    # Rejected target, lost server, or idle: drop what is left of the
                    # previous target's chunk and hold the last commanded pose.
                    # `consumed_since_swap` stays 0, so the `refetch_after` trigger
                    # never re-fires; only a new seq from the control file
                    # (`fire_now`) does. A chunk that was still in flight when the
                    # target went idle lands here too and is discarded.
                    if isinstance(new_chunk, Hold):
                        logger.error(
                            "target #%d on hold: %s — holding position until a valid target is applied",
                            new_chunk.seq,
                            new_chunk.error,
                        )
                    current_chunk = []
                    idx = 0
                    inflight = False
                    consumed_since_swap = 0
                    chunk_exhausted_at = None
                    holding = True
                else:
                    holding = False
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

            # 2. Send an action (or hold position if we ran out / were told to).
            if idx < len(current_chunk):
                action = current_chunk[idx]
                robot.send_action(action)
                last_action = action
                idx += 1
                consumed_since_swap += 1
                chunk_exhausted_at = None
            elif holding:
                # Idle or on hold: keep the last pose, quietly (logged once above).
                if last_action is not None:
                    robot.send_action(last_action)
            else:
                if chunk_exhausted_at is None:
                    chunk_exhausted_at = tick_start
                    late_logged_at = None
                gap_ms = (tick_start - chunk_exhausted_at) * 1000.0
                if last_action is not None:
                    robot.send_action(last_action)
                # First tick of a gap, then once per second — not once per tick.
                if late_logged_at is None or tick_start - late_logged_at >= 1.0:
                    logger.warning("late chunk: gap=%.1f ms (holding position)", gap_ms)
                    late_logged_at = tick_start

            # 3. Fire the next request once we've consumed `refetch_after`
            #    actions from the current chunk and nothing is already in flight.
            #    A new target (`fire_now`) fires as soon as the line is free.
            #    Record the fire frame and how many actions are still queued so
            #    the swap above can measure round-trip latency in frames and cap
            #    the stale-lead trim at what the robot can actually consume. The
            #    instruction is attached by the worker from the target it accepted.
            #    An idle target is handed over once, without an observation (the
            #    worker only drops its client); after that nothing fires while idle.
            if not inflight and (fire_now or consumed_since_swap >= cfg.refetch_after):
                t_start = time.perf_counter()
                obs = {} if target.idle else robot.get_observation()
                obs_queue.put((target, obs))
                inflight = True
                fire_now = False
                fire_frame = frame_counter
                remaining_at_fire = max(len(current_chunk) - idx, 0)
                if not target.idle:
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
        if worker is not None:
            try:
                obs_queue.put_nowait(None)
            except queue.Full:
                pass
            worker.join(timeout=2.0)
        if worker is not None and worker.is_alive():
            logger.warning("inference worker did not exit within 2 s")
        elif adapter.policy is not None:
            # Only when the worker is gone — ZMQ sockets are not thread-safe.
            adapter.policy.close()


@draccus.wrap()
def main(cfg: BmhInferenceConfig) -> None:
    init_logging()
    # Never echo the API token — the controller-app tails this output into its log.
    logger.info(pformat({**asdict(cfg), "api_token": "***" if cfg.api_token else ""}))

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

    # Initial target: the controller-app's control file when given, else the CLI
    # flags, else idle. A real target is validated by `_connect_policy` (ping +
    # modality) in `_run_control_loop`, before the robot moves.
    watcher = ControlFileWatcher(Path(cfg.control_file).expanduser()) if cfg.control_file else None
    target = _initial_target(watcher, cfg.policy_host, cfg.policy_port, cfg.lang_instruction, cfg.api_token)

    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    logger.info("Robot connected: %s", robot.name)

    try:
        # bi_so_follower.action_features lists every `.pos` key in bus order with the
        # left_/right_ prefix — for the BMH-101 with `--robot.left_arm_config.with_head=true`
        # that is the same 16 names the training dataset's `features` carry. The policy
        # client is attached in `_run_control_loop`, once the server passed the modality check.
        adapter = BiSoBimanualAdapter(
            None,
            jpeg_quality=cfg.jpeg_quality,
            joint_names=list(robot.action_features.keys()),
        )

        def connect(t: InferenceTarget) -> PolicyClient:
            return _connect_policy(t, adapter, cfg.probe_timeout_ms, cfg.timeout_ms)

        _run_control_loop(robot, adapter, watcher, target, cfg, connect)
    finally:
        try:
            robot.disconnect()
        except Exception as e:
            logger.error("Error disconnecting robot: %s", e)


if __name__ == "__main__":
    main()
