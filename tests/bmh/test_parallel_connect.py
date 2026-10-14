"""BMH: cameras and the two bimanual arms connect side by side (startup-time fix)."""

import threading
import time

import pytest

from lerobot.cameras.utils import connect_cameras
from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig
from lerobot.robots.so_follower import SOFollowerConfig


class _FakeDevice:
    """Minimal connect/disconnect/is_connected stand-in; records thread + timing."""

    def __init__(self, name: str, delay_s: float = 0.0, fail: Exception | None = None):
        self.name = name
        self.delay_s = delay_s
        self.fail = fail
        self.is_connected = False
        self.disconnected = False
        self.thread: str | None = None
        self.started_at: float | None = None
        self.calibrate_arg: bool | None = None

    def connect(self, calibrate: bool = True) -> None:
        self.thread = threading.current_thread().name
        self.started_at = time.perf_counter()
        self.calibrate_arg = calibrate
        time.sleep(self.delay_s)
        if self.fail is not None:
            raise self.fail
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False
        self.disconnected = True


# --- connect_cameras -------------------------------------------------------------------


def test_connect_cameras_overlaps_warmups():
    cams = {k: _FakeDevice(k, delay_s=0.3) for k in ("front", "left_wrist", "right_wrist")}
    t = time.perf_counter()
    connect_cameras(cams)
    elapsed = time.perf_counter() - t

    assert all(c.is_connected for c in cams.values())
    assert elapsed < 0.6, f"cameras connected sequentially ({elapsed:.2f}s for 3 × 0.3s)"
    assert len({c.thread for c in cams.values()}) == 3


def test_connect_cameras_single_camera_runs_inline():
    cam = _FakeDevice("front")
    connect_cameras({"front": cam})
    assert cam.is_connected
    assert cam.thread == threading.main_thread().name


def test_connect_cameras_empty_is_noop():
    connect_cameras({})


def test_connect_cameras_failure_disconnects_the_others_and_reraises_first():
    boom = ConnectionError("no /dev/video4")
    cams = {
        "front": _FakeDevice("front", delay_s=0.1),
        "left_wrist": _FakeDevice("left_wrist", fail=boom),
        "right_wrist": _FakeDevice("right_wrist", delay_s=0.2),
    }
    with pytest.raises(ConnectionError) as exc:
        connect_cameras(cams)
    assert exc.value is boom
    # the slow ones were still joined, then released again
    assert cams["front"].disconnected and cams["right_wrist"].disconnected
    assert not any(c.is_connected for c in cams.values())
    assert not cams["left_wrist"].disconnected  # never connected, so nothing to release


# --- BiSOFollower.connect ----------------------------------------------------------------


def _bi_follower_with_fake_arms(left: _FakeDevice, right: _FakeDevice) -> BiSOFollower:
    robot = BiSOFollower(
        BiSOFollowerConfig(
            left_arm_config=SOFollowerConfig(port="/dev/null"),
            right_arm_config=SOFollowerConfig(port="/dev/null"),
        )
    )
    robot.left_arm = left  # type: ignore[assignment]
    robot.right_arm = right  # type: ignore[assignment]
    return robot


def test_bi_so_follower_connects_arms_in_parallel():
    left, right = _FakeDevice("left", delay_s=0.3), _FakeDevice("right", delay_s=0.3)
    robot = _bi_follower_with_fake_arms(left, right)

    t = time.perf_counter()
    robot.connect(calibrate=False)
    elapsed = time.perf_counter() - t

    assert robot.is_connected
    assert elapsed < 0.55, f"arms connected sequentially ({elapsed:.2f}s for 2 × 0.3s)"
    assert left.thread != right.thread
    assert left.calibrate_arg is False and right.calibrate_arg is False


def test_bi_so_follower_failure_releases_other_arm_and_reraises():
    boom = ConnectionError("Could not connect on port '/dev/ttyACM1'")
    left, right = _FakeDevice("left", fail=boom), _FakeDevice("right", delay_s=0.1)
    robot = _bi_follower_with_fake_arms(left, right)

    with pytest.raises(ConnectionError) as exc:
        robot.connect()
    assert exc.value is boom
    assert right.disconnected and not right.is_connected
    assert not robot.is_connected


def test_bi_so_follower_left_error_wins_when_both_fail():
    left = _FakeDevice("left", delay_s=0.1, fail=ConnectionError("left"))
    right = _FakeDevice("right", fail=ConnectionError("right"))
    robot = _bi_follower_with_fake_arms(left, right)
    with pytest.raises(ConnectionError, match="left"):
        robot.connect()
