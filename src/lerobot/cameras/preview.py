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

"""
BMH: rate-limited JPEG preview writer, tapped from a camera's background read thread.

Lets an external process (the BMH controller-app) show a live view of a camera that
`lerobot-record` already holds open exclusively, without opening the device a second time.
"""

import contextlib
import logging
import os
import time
from typing import Any

import cv2  # type: ignore  # TODO: add type stubs for OpenCV
from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

logger = logging.getLogger(__name__)


class FramePreviewWriter:
    """Rate-limited JPEG snapshot writer.

    Called from a camera's read thread with the raw BGR frame on every capture. At most
    every ``1 / fps`` seconds the frame is (optionally rotated), downscaled to ``width``
    pixels wide, JPEG-encoded and written atomically (``path + ".tmp"`` then ``os.replace``)
    so readers never see a partial file. Frames arriving inside the interval cost a single
    ``perf_counter`` comparison.

    Any failure is logged once and otherwise swallowed: a preview problem must never
    interrupt recording.
    """

    def __init__(
        self,
        path: str,
        fps: float,
        width: int,
        quality: int,
        rotation: int | None = None,
    ) -> None:
        """
        Args:
            path: Destination JPEG path (ideally on tmpfs, e.g. ``/dev/shm/...``).
            fps: Maximum number of snapshots written per second.
            width: Target width in pixels; frames wider than this are downscaled (aspect kept).
            quality: JPEG quality 1–100.
            rotation: Optional ``cv2.ROTATE_*`` constant applied before encoding so the preview
                matches the recorded orientation.
        """
        self._path = path
        self._tmp_path = path + ".tmp"
        self._min_interval = 1.0 / fps
        self._width = width
        self._quality = quality
        self._rotation = rotation
        self._last = 0.0  # perf_counter of the last attempt; 0 → write on first frame
        self._dir_ready = False
        self._warned = False

    def maybe_write(self, frame_bgr: NDArray[Any]) -> None:
        """Writes a snapshot of ``frame_bgr`` if the rate limit allows it; never raises."""
        now = time.perf_counter()
        if now - self._last < self._min_interval:
            return
        # Bumped before doing any work so a failing encode/write does not retry at full camera rate.
        self._last = now

        try:
            frame = frame_bgr
            if self._rotation is not None:
                frame = cv2.rotate(frame, self._rotation)

            h, w = frame.shape[:2]
            if w > self._width:
                new_h = max(1, round(h * self._width / w))
                frame = cv2.resize(frame, (self._width, new_h), interpolation=cv2.INTER_AREA)

            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality])
            if not ok:
                return

            if not self._dir_ready:
                parent = os.path.dirname(self._path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                self._dir_ready = True

            with open(self._tmp_path, "wb") as f:
                f.write(buf.tobytes())
            os.replace(self._tmp_path, self._path)
        except Exception as e:
            if not self._warned:
                self._warned = True
                logger.warning(f"Preview write to {self._path} failed (further failures are silent): {e}")

    def close(self) -> None:
        """Best-effort removal of the preview file and its tmp sibling."""
        self._dir_ready = False
        for p in (self._tmp_path, self._path):
            with contextlib.suppress(OSError):
                os.remove(p)
