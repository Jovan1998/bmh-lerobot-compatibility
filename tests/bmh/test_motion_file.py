"""Unit tests for the fixed-recording file format (``bmh/motion/file.py``) — stdlib only.

Round-trip, every validation rule of ``parse_motion`` / ``load_motion``, the BMH-101 key
order, and ``is_under_motion_root`` (``..`` and the root itself are not "under" it).
"""

import json

import pytest

from bmh.motion.file import (
    MOTION_DIR_NAME,
    MOTION_KEYS,
    MotionFile,
    is_under_motion_root,
    load_motion,
    motion_root,
    parse_motion,
    save_motion,
)

DOC = {
    "version": 1,
    "fps": 60,
    "keys": ["a.pos", "b.pos"],
    "frames": [[0.0, 1.5], [2, -3.25]],
    "recorded_at": 1.0,
}


def test_motion_keys_are_the_16_bimanual_keys_in_bus_order():
    assert len(MOTION_KEYS) == 16 and len(set(MOTION_KEYS)) == 16
    assert MOTION_KEYS[:9] == (
        "left_shoulder_pan.pos",
        "left_shoulder_lift.pos",
        "left_elbow_flex.pos",
        "left_wrist_flex.pos",
        "left_wrist_yaw.pos",
        "left_wrist_roll.pos",
        "left_gripper.pos",
        "left_head_pan.pos",
        "left_head_tilt.pos",
    )
    assert MOTION_KEYS[9] == "right_shoulder_pan.pos" and MOTION_KEYS[-1] == "right_gripper.pos"


def test_parse_motion_accepts_valid_and_ignores_unknown_keys():
    motion = parse_motion({**DOC, "note": "x"})
    assert motion == MotionFile(
        fps=60, keys=("a.pos", "b.pos"), frames=((0.0, 1.5), (2.0, -3.25)), recorded_at=1.0
    )
    assert motion.duration_s == pytest.approx(1 / 60)


def test_save_then_load_round_trips(tmp_path):
    motion = MotionFile(
        fps=60, keys=MOTION_KEYS, frames=((1.234,) * 16, (-5.5,) * 16), recorded_at=1762600000.5
    )
    path = tmp_path / "new" / "folder" / "motion.json"  # the folder is created
    save_motion(path, motion)
    assert load_motion(path) == motion
    assert json.loads(path.read_text())["version"] == 1
    assert [p.name for p in path.parent.iterdir()] == ["motion.json"]  # no tmp file left


def test_save_replaces_an_existing_file(tmp_path):
    path = tmp_path / "motion.json"
    save_motion(path, parse_motion(DOC))
    other = parse_motion({**DOC, "fps": 30})
    save_motion(path, other)
    assert load_motion(path) == other


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"version": 2}, "version"),
        ({"version": True}, "version"),
        ({"version": "1"}, "version"),
        ({"fps": 0}, "fps"),
        ({"fps": -60}, "fps"),
        ({"fps": 60.0}, "fps"),
        ({"fps": True}, "fps"),
        ({"keys": []}, "keys"),
        ({"keys": "a.pos"}, "keys"),
        ({"keys": ["a.pos", ""]}, "keys"),
        ({"keys": ["a.pos", 3]}, "keys"),
        ({"keys": ["a.pos", "a.pos"]}, "repeat"),
        ({"frames": []}, "frames"),
        ({"frames": "x"}, "frames"),
        ({"frames": [[0.0]]}, "frame 0"),
        ({"frames": [[0.0, 1.0], [1.0, 2.0, 3.0]]}, "frame 1"),
        ({"frames": [[0.0, True]]}, "frame 0"),
        ({"frames": [[0.0, "1"]]}, "frame 0"),
        ({"frames": [[0.0, None]]}, "frame 0"),
        ({"frames": [[0.0, float("nan")]]}, "frame 0"),
        ({"frames": [[float("inf"), 0.0]]}, "frame 0"),
        ({"recorded_at": None}, "recorded_at"),
        ({"recorded_at": True}, "recorded_at"),
        ({"recorded_at": float("nan")}, "recorded_at"),
    ],
)
def test_parse_motion_rejects(overrides, message):
    with pytest.raises(ValueError, match=message):
        parse_motion({**DOC, **overrides})


def test_parse_motion_rejects_missing_keys_and_non_objects():
    for key in DOC:
        with pytest.raises(ValueError):
            parse_motion({k: v for k, v in DOC.items() if k != key})
    with pytest.raises(ValueError, match="object"):
        parse_motion([DOC])


def test_load_motion_names_the_file(tmp_path):
    bad = tmp_path / "motion.json"
    bad.write_text("{not json")
    with pytest.raises(ValueError, match="motion.json"):
        load_motion(bad)
    bad.write_text(json.dumps({**DOC, "fps": 0}))
    with pytest.raises(ValueError, match="motion.json.*fps"):
        load_motion(bad)
    with pytest.raises(OSError):
        load_motion(tmp_path / "missing.json")


def test_motion_root_is_in_the_app_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert motion_root() == tmp_path / ".cache" / "bmh-101" / MOTION_DIR_NAME


def test_is_under_motion_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    root = motion_root()
    assert is_under_motion_root(root / "wave" / "motion.json")
    assert is_under_motion_root("~/.cache/bmh-101/fixed-recordings/wave/motion.json")
    assert not is_under_motion_root(root)
    assert not is_under_motion_root(root / ".." / "inference-control.json")
    assert not is_under_motion_root(root / "wave" / ".." / ".." / "x.json")
    assert not is_under_motion_root("/etc/passwd")
    assert not is_under_motion_root(str(root) + "-evil/motion.json")  # a sibling with the same prefix
    assert is_under_motion_root(tmp_path / "r" / "m.json", root=tmp_path / "r")


def test_is_under_motion_root_follows_symlinks(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    root = motion_root()
    root.mkdir(parents=True)
    (root / "escape").symlink_to(tmp_path)
    assert not is_under_motion_root(root / "escape" / "secret.json")
