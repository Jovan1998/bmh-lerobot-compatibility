"""The policy client's chunk loop as a class a DAgger session can suspend and resume.

A port of steps 1 to 3 of ``_run_control_loop`` in ``bmh/scripts/bmh_policy_client.py``
(swap in the arrived chunk, pick the next action, fire the next request), with the same
variables, the same worker thread (`_inference_worker`, imported unchanged) and the same
two ``Queue(maxsize=1)``. The policy client itself is not touched and keeps its own copy;
``tests/bmh/test_dagger_policy_runner.py`` pins this one to the same behaviour.

What differs, because a DAgger session is not always running the policy:

- `tick` picks the action but does not send it; the session sends it and then calls
  `fire`, so the observation of a request is taken after the tick's action went out, as in
  the policy client.
- `suspend` / `resume` bracket the phases in which the policy does not drive (paused,
  rewinding, take-over). A reply to a request fired before the last `suspend` / `resume`
  was computed for a robot that is somewhere else by now, and is dropped on arrival.
- There is no synchronous bootstrap inference: after `resume` the robot holds until the
  first chunk arrives.
- A new target is probed as soon as the line is free, also while suspended, so the app
  sees ``ready`` or the error before the operator presses Run. That costs one inference
  whose chunk is dropped.
- The policy's ``feedback_*`` values are split off and ignored (v1).
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
from collections.abc import Callable
from typing import Any, Literal

from bmh.policy_client import PolicyClient
from bmh.policy_client.control import Hold, Idle, InferenceTarget, LoopSettings
from bmh.policy_client.layout import split_feedback
from bmh.policy_client.motion import LookaheadSmoother, MotionShaper
from bmh.scripts.bmh_policy_client import BiSoBimanualAdapter, _blend_actions, _inference_worker

logger = logging.getLogger(__name__)

PolicyState = Literal["none", "connecting", "ready", "error"]


class PolicyRunner:
    """Plays a policy server's action chunks, one action per `tick`, while resumed."""

    def __init__(
        self,
        adapter: BiSoBimanualAdapter,
        connect: Callable[[InferenceTarget], PolicyClient],
        settings: LoopSettings,
        fps: int,
    ):
        """Start the inference worker, idle and without a client.

        Args:
            adapter: Adapter without a policy client; the worker attaches one per target.
            connect: Opens a validated client for a target, or raises `PolicyConnectError`.
            settings: Validated loop settings to start with.
            fps: Recorded control rate; with ``settings.speed_pct`` it gives the tick period.
        """
        self._adapter = adapter
        self._settings = settings
        self._fps = fps

        self.current_chunk: list[dict[str, float]] = []
        self.idx = 0
        self.inflight = False
        # What `tick` last returned (shaped), and what the smoothing was asked for. Both
        # `None` after `resume` / `suspend`: a late-chunk tick then has nothing to approach.
        self.last_action: dict[str, float] | None = None
        self.last_command: dict[str, float] | None = None
        self.lookahead = LookaheadSmoother(settings.lookahead_frames)
        self.shaper = MotionShaper(settings.smoothing_pct)
        self.frame_counter = 0
        self.fire_frame = 0
        self.remaining_at_fire = 0
        self.consumed_since_swap = 0
        self.fire_now = False

        self._target: InferenceTarget | None = None
        self._seq = 0
        self._fire_seq = -1
        self._state: PolicyState = "none"
        self._error: str | None = None
        # True after the worker rejected the target or lost its server: nothing plays and
        # nothing fires until a new seq (`set_target`, or `resume`, which re-issues it).
        self._holding = False
        self._active = False
        # Bumped by `suspend` and `resume`; a reply fired under an older value is stale.
        self._epoch = 0
        self._fire_epoch = 0
        self._short_chunk_warned_seq: int | None = None
        self._chunk_exhausted_at: float | None = None
        self._late_logged_at: float | None = None

        self._obs_queue: queue.Queue[tuple[InferenceTarget, dict[str, Any]] | None] = queue.Queue(maxsize=1)
        self._chunk_queue: queue.Queue[list[dict[str, float]] | Hold | Idle | None] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._worker = threading.Thread(
            target=_inference_worker,
            args=(
                adapter,
                self._obs_queue,
                self._chunk_queue,
                self._stop_event,
                InferenceTarget.make_idle(0),
                connect,
            ),
            name="bmh-dagger-inference-worker",
            daemon=True,
        )
        self._worker.start()

    # ------------------------------------------------------------------ state

    @property
    def policy_state(self) -> PolicyState:
        """State of the policy connection.

        ``none`` without a target, ``connecting`` until the worker answered the current
        one, then ``ready`` (a chunk came back) or ``error`` (rejected, or a request failed).
        """
        return self._state

    @property
    def policy_error(self) -> str | None:
        """Why the state is ``error``."""
        return self._error

    @property
    def active(self) -> bool:
        """Whether the policy drives (between `resume` and `suspend`)."""
        return self._active

    @property
    def settings(self) -> LoopSettings:
        """The loop settings in effect."""
        return self._settings

    def period_s(self) -> float:
        """Seconds per tick while the policy drives (follows the live ``speed_pct``)."""
        return self._settings.period_s(self._fps)

    def set_target(self, host: str, port: int, lang_instruction: str, api_token: str) -> InferenceTarget:
        """Use this server and instruction from the next request on.

        The worker probes a new endpoint (ping + modality check) and answers with a
        ``BMH_TARGET`` line, exactly as for the policy client; an instruction-only change
        is adopted without a reconnect. The request goes out as soon as the line is free.
        """
        self._seq += 1
        self._target = InferenceTarget(
            seq=self._seq,
            policy_host=host,
            policy_port=port,
            lang_instruction=lang_instruction,
            api_token=api_token,
        )
        self._state = "connecting"
        self._error = None
        self.fire_now = True
        logger.info(
            'target #%d requested: %s:%d "%s"',
            self._target.seq,
            self._target.policy_host,
            self._target.policy_port,
            self._target.lang_instruction,
        )
        return self._target

    def apply_settings(self, settings: LoopSettings) -> None:
        """Live loop settings (already validated): they apply from the next tick / swap."""
        self._settings = settings
        self.lookahead.set_frames(settings.lookahead_frames)
        self.shaper.set_smoothing(settings.smoothing_pct)

    def resume(self, pose: dict[str, float]) -> None:
        """Let the policy drive from ``pose`` (where the robot is held).

        Nothing is played until a chunk requested from here arrives. A target that had
        failed is requested again under a new seq, so pressing Run retries it.

        Raises:
            RuntimeError: Without a target.
        """
        if self._target is None:
            raise RuntimeError("no policy target")
        if self._state == "error":
            t = self._target
            self.set_target(t.policy_host, t.policy_port, t.lang_instruction, t.api_token)
        self._clear()
        self.shaper.reset(pose)
        self._active = True
        self.fire_now = True

    def suspend(self) -> None:
        """Stop playing at once. A reply still in flight is dropped when it arrives."""
        self._clear()
        self.shaper.reset()
        self._active = False

    def _clear(self) -> None:
        self._epoch += 1
        self.current_chunk = []
        self.idx = 0
        self.consumed_since_swap = 0
        self.remaining_at_fire = 0
        self.last_action = None
        self.last_command = None
        self.lookahead.clear()
        self._chunk_exhausted_at = None

    # ------------------------------------------------------------------- tick

    def tick(self, tick_start: float) -> dict[str, float] | None:
        """One control tick: swap in an arrived chunk, then pick this tick's action.

        Returns:
            The action to send, or ``None`` when there is none (suspended, on hold, or
            waiting for the first chunk) and the caller keeps holding.

        Raises:
            RuntimeError: If the inference worker crashed.
        """
        self.frame_counter += 1
        self._swap()
        if not self._active:
            return None
        return self._next_action(tick_start)

    def _swap(self) -> None:
        """Step 1: take a reply off the queue; time-align and crossfade a chunk."""
        try:
            new_chunk = self._chunk_queue.get_nowait()
        except queue.Empty:
            return
        if new_chunk is None:
            raise RuntimeError("inference worker reported failure; aborting")
        settings = self._settings
        self.inflight = False
        current = self._target is not None and self._fire_seq == self._target.seq
        if isinstance(new_chunk, (Hold, Idle)):
            # Rejected target or lost server: drop what is left of the chunk and hold.
            # Only a new seq fires again. (`Idle` cannot happen: no idle target is sent.)
            if isinstance(new_chunk, Hold) and self._target is not None and new_chunk.seq == self._target.seq:
                logger.error(
                    "target #%d on hold: %s — holding position until a valid target is applied",
                    new_chunk.seq,
                    new_chunk.error,
                )
                self._state = "error"
                self._error = new_chunk.error
                self._holding = True
                self.current_chunk = []
                self.idx = 0
                self.consumed_since_swap = 0
                self._chunk_exhausted_at = None
                self.shaper.reset(self.last_action)
                self.last_command = None
            return
        if current:
            self._state = "ready"
            self._error = None
            self._holding = False
        if not self._active or self._fire_epoch != self._epoch:
            # A probe while suspended, or an answer to where the robot was before the
            # last suspend / resume: not for the pose the robot is in now.
            return

        # Cut to the horizon on arrival, so a live change also applies to a request
        # that was already in flight.
        if len(new_chunk) < settings.action_horizon and self._short_chunk_warned_seq != settings.seq:
            logger.warning(
                "policy returns %d-step chunks, shorter than action_horizon=%d%s",
                len(new_chunk),
                settings.action_horizon,
                " — the loop refetches at chunk end" if len(new_chunk) <= settings.refetch_after else "",
            )
            self._short_chunk_warned_seq = settings.seq
        new_chunk = new_chunk[: settings.action_horizon]
        elapsed = self.frame_counter - self.fire_frame
        # Drop the frames the robot already advanced along the old chunk (capped at what
        # it could consume), keeping at least `refetch_after` actions so the re-fire
        # trigger can arm again.
        drop = min(elapsed, self.remaining_at_fire)
        drop = min(drop, max(len(new_chunk) - settings.refetch_after, 0))

        # Crossfade the seam over the first `blend_frames`, clamped to the overlap.
        leftover_old = self.current_chunk[self.idx :]
        new_kept = new_chunk[drop:]
        blend_len = min(settings.blend_frames, len(leftover_old), len(new_kept))
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
            self.remaining_at_fire,
            len(leftover_old),
        )
        self.current_chunk = new_kept
        self.idx = 0
        self.consumed_since_swap = 0
        self._chunk_exhausted_at = None

    def _next_action(self, tick_start: float) -> dict[str, float] | None:
        """Step 2: the next chunk action through look-ahead and smoothing, or the late-chunk hold."""
        dt = self.period_s()
        if self.idx < len(self.current_chunk):
            command, _feedback = split_feedback(self.current_chunk[self.idx])
            upcoming = [
                split_feedback(step)[0]
                for step in self.current_chunk[self.idx + 1 : self.idx + 1 + self.lookahead.frames]
            ]
            smoothed = self.lookahead.step(command, upcoming)
            action = self.shaper.step(smoothed, dt)
            self.last_action = action
            self.last_command = smoothed
            self.idx += 1
            self.consumed_since_swap += 1
            self._chunk_exhausted_at = None
            return action
        self.lookahead.clear()
        if self._holding:
            return None
        if self._chunk_exhausted_at is None:
            self._chunk_exhausted_at = tick_start
            self._late_logged_at = None
        if self.last_action is None:
            # Just resumed: the first chunk is on its way; the caller holds.
            return None
        # Waiting for the next chunk: the policy's last command still stands, so a
        # smoothed arm keeps settling onto it instead of stopping dead.
        if self.last_command is not None:
            self.last_action = self.shaper.step(self.last_command, dt)
        if self._late_logged_at is None or tick_start - self._late_logged_at >= 1.0:
            logger.warning(
                "late chunk: gap=%.1f ms (no new actions; holding the last command)",
                (tick_start - self._chunk_exhausted_at) * 1000.0,
            )
            self._late_logged_at = tick_start
        return self.last_action

    def fire(self, get_observation: Callable[[], dict[str, Any]]) -> bool:
        """Step 3: request the next chunk when it is due. Call after the tick's action was sent.

        Due means: a new target or a `resume` is pending (``fire_now``), or, while
        resumed, ``refetch_after`` actions of the current chunk were played. The stall
        guard fires at the chunk's end at the latest
        (``min(refetch_after, len(current_chunk))``).

        Returns:
            Whether a request went out (and ``get_observation`` was called).
        """
        if self.inflight or self._target is None:
            return False
        refetch_at = min(self._settings.refetch_after, len(self.current_chunk))
        due = self.fire_now or (
            self._active and len(self.current_chunk) > 0 and self.consumed_since_swap >= refetch_at
        )
        if not due:
            return False
        self._obs_queue.put((self._target, get_observation()))
        self.inflight = True
        self.fire_now = False
        self.fire_frame = self.frame_counter
        self._fire_seq = self._target.seq
        self._fire_epoch = self._epoch
        self.remaining_at_fire = max(len(self.current_chunk) - self.idx, 0)
        logger.info(
            "inference fired at frame=%d (consumed_since_swap=%d), remaining_at_fire=%d",
            self.frame_counter,
            self.consumed_since_swap,
            self.remaining_at_fire,
        )
        return True

    # ------------------------------------------------------------------ close

    def close(self) -> None:
        """Stop the worker (2 s) and close the policy client it owned."""
        self._stop_event.set()
        with contextlib.suppress(queue.Full):
            self._obs_queue.put_nowait(None)
        self._worker.join(timeout=2.0)
        if self._worker.is_alive():
            logger.warning("inference worker did not exit within 2 s")
        elif self._adapter.policy is not None:
            # Only when the worker is gone — ZMQ sockets are not thread-safe.
            self._adapter.policy.close()
            self._adapter.policy = None
