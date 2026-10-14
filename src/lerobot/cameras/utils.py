#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import cast

from lerobot.utils.import_utils import make_device_from_device_class

from .camera import Camera
from .configs import CameraConfig, Cv2Rotation

logger = logging.getLogger(__name__)


def make_cameras_from_configs(camera_configs: dict[str, CameraConfig]) -> dict[str, Camera]:
    cameras: dict[str, Camera] = {}

    for key, cfg in camera_configs.items():
        # TODO(Steven): Consider just using the make_device_from_device_class for all types
        if cfg.type == "opencv":
            from .opencv import OpenCVCamera

            cameras[key] = OpenCVCamera(cfg)

        elif cfg.type == "intelrealsense":
            from .realsense.camera_realsense import RealSenseCamera

            cameras[key] = RealSenseCamera(cfg)

        elif cfg.type == "reachy2_camera":
            from .reachy2_camera.reachy2_camera import Reachy2Camera

            cameras[key] = Reachy2Camera(cfg)

        elif cfg.type == "zmq":
            from .zmq.camera_zmq import ZMQCamera

            cameras[key] = ZMQCamera(cfg)

        else:
            try:
                cameras[key] = cast(Camera, make_device_from_device_class(cfg))
            except Exception as e:
                raise ValueError(f"Error creating camera {key} with config {cfg}: {e}") from e

    return cameras


def connect_cameras(cameras: Mapping[str, Camera]) -> None:
    """Connect every camera, overlapping their open and warmup phases.

    BMH: `Camera.connect()` blocks for the camera's full warmup window (plus the time to open
    the device), so connecting the BMH-101's three USB cameras one after another cost ~16 s on
    a Raspberry Pi 5. The devices are independent, so each one is connected from its own
    thread and the wall time becomes that of the slowest camera instead of the sum.

    If any camera fails, the others still finish (threads are joined), the cameras that did
    connect are disconnected again so no device handle is left dangling, and the first error is
    re-raised — the caller sees exactly the exception the sequential loop would have raised.
    """
    if not cameras:
        return
    if len(cameras) == 1:
        next(iter(cameras.values())).connect()
        return

    errors: dict[str, BaseException] = {}

    def _connect(key: str, cam: Camera) -> None:
        try:
            cam.connect()
        except BaseException as e:  # re-raised below, after every thread has finished
            errors[key] = e

    with ThreadPoolExecutor(max_workers=len(cameras), thread_name_prefix="camera-connect") as pool:
        for key, cam in cameras.items():
            pool.submit(_connect, key, cam)

    if not errors:
        return

    for key, cam in cameras.items():
        if key in errors or not cam.is_connected:
            continue
        try:
            cam.disconnect()
        except Exception as e:  # best effort — the original error is what matters
            logger.warning(f"Failed to disconnect camera '{key}' after another camera failed to connect: {e}")

    first_key = next(iter(errors))
    if len(errors) > 1:
        logger.error(
            f"{len(errors)} cameras failed to connect: {', '.join(errors)} — raising the first error"
        )
    raise errors[first_key]


def get_cv2_rotation(rotation: Cv2Rotation) -> int | None:
    import cv2  # type: ignore  # TODO: add type stubs for OpenCV

    if rotation == Cv2Rotation.ROTATE_90:
        return int(cv2.ROTATE_90_CLOCKWISE)
    elif rotation == Cv2Rotation.ROTATE_180:
        return int(cv2.ROTATE_180)
    elif rotation == Cv2Rotation.ROTATE_270:
        return int(cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return None
