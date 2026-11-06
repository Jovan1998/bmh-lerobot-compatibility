# !/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time

from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_so_leader import SOLeaderTeleopConfig

logger = logging.getLogger(__name__)

# BMH: attempts after a missed reply when `configure()` turns the torque off.
_TORQUE_RETRIES = 5

_HOMING_POSITION_DIAGRAM = r"""
       ╭─────┬────────────────────┬──────╮ ◉╲═════╗   ← moveable claw
       │     │      forearm       │  ▤▤  │╤══╲════╝
       ╰┬───┬┴────────────────────┴──────┴┴═══════╝
        │   │                                  ███
        │   │                                  ███  ← handle
        │   │                                  ███
        │   │   upper arm
        │   │
        │   │
        │   │
   ╭────┴───┴────╮
   │   base ◉    │
   ╰──┬───────┬──╯
  ╭───┴───────┴───╮
  │▓▓▓ C-clamp ▓▓▓│
  ╰───────────────╯
 ═══════════════════════════════════════════════════
                table edge
"""


class SOLeader(Teleoperator):
    """Generic SO leader base for SO-100/101/10X teleoperators."""

    config_class = SOLeaderTeleopConfig
    name = "so_leader"

    def __init__(self, config: SOLeaderTeleopConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        # BMH-101 arm: 7 DOF (adds wrist_yaw), IDs counted down from the base.
        motors = {
            "shoulder_pan": Motor(7, "sts3215", norm_mode_body),
            "shoulder_lift": Motor(6, "sts3215", norm_mode_body),
            "elbow_flex": Motor(5, "sts3215", norm_mode_body),
            "wrist_flex": Motor(4, "sts3215", norm_mode_body),
            "wrist_yaw": Motor(3, "sts3215", norm_mode_body),
            "wrist_roll": Motor(2, "sts3215", norm_mode_body),
            "gripper": Motor(1, "sts3215", MotorNormMode.RANGE_0_100),
        }
        if config.with_head:
            # BMH-101: head_pan/head_tilt (IDs 8/9) ride the LEFT arm's serial bus
            motors["head_pan"] = Motor(8, "sts3215", norm_mode_body)
            motors["head_tilt"] = Motor(9, "sts3215", norm_mode_body)
        self.bus = SerialMotorsBus(
            port=self.config.port,
            motors=motors,
            calibration=self.calibration,
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return self.action_features

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            if self.calibration:
                # BMH: motor registers drifted from the calibration file (e.g. replaced
                # servo). Auto-apply the file instead of prompting — connect() must work
                # headless (the leader host runs without stdin). Recalibration happens
                # explicitly via `lerobot-calibrate`.
                logger.info(
                    f"Motor calibration registers differ from the file for id {self.id} — writing file values to the motors"
                )
                self.bus.write_calibration(self.calibration)
            else:
                logger.info("No calibration file found")
                self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self._keep_calibration_file(self.bus):
            return

        print(_HOMING_POSITION_DIAGRAM)
        print("Video walkthrough: https://huggingface.co/docs/lerobot/main/en/so101#calibration-video")
        # BMH-101: wrist_roll has a limited range too, so every joint is swept; the zero position is
        # then moved to the center of each recorded range (independent of the pose held at the prompt).
        calibration = self.bus.record_calibration("position", full_turn_motors=[])
        self.calibration = self.bus.center_calibration_on_ranges(calibration)
        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        # BMH: retry a missed reply instead of aborting the leader host's startup
        # (same count as SOFollower.configure).
        self.bus.disable_torque(num_retry=_TORQUE_RETRIES)
        self.bus.configure_motors()
        self.bus.set_operating_mode("position")

    def enable_torque(self) -> None:
        self.bus.enable_torque()

    def disable_torque(self) -> None:
        self.bus.disable_torque()

    def setup_motors(self) -> None:
        self.bus.setup_motors()

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position", num_retry=self.config.num_read_retries)
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, float]) -> None:
        goals = {k.removesuffix(".pos"): v for k, v in feedback.items() if k.endswith(".pos")}
        if goals:
            self.bus.sync_write("Goal_Position", goals)

    @check_if_not_connected
    def disconnect(self) -> None:
        self.bus.disconnect()
        logger.info(f"{self} disconnected.")


SO100Leader = SOLeader
SO101Leader = SOLeader
