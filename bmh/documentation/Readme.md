# Follower Server — Network Teleoperator

This directory contains the **follower-side** network teleoperation modules.
These are lerobot `Teleoperator` implementations that receive leader arm positions
over ZMQ instead of reading from local USB.

## Modules

### `so_network_leader/`

Drop-in replacement for `SOLeader`. Connects to a ZMQ PUSH socket on the leader Pi
and receives joint positions as JSON.

- `config_so_network_leader.py` — Config: `remote_ip`, `port_zmq`, timeouts
- `so_network_leader.py` — `SONetworkLeader(Teleoperator)` implementation

### `bi_so_network_leader/`

Bimanual wrapper — composes two `SONetworkLeader` instances (left + right arms
on separate ZMQ ports). Drop-in replacement for `BiSOLeader`.

- `config_bi_so_network_leader.py` — Config with `left_arm_config` + `right_arm_config`
- `bi_so_network_leader.py` — `BiSONetworkLeader(Teleoperator)` implementation

## Deployment

These modules are **symlinked** into the lerobot teleoperators directory on the Pi
by `installation/05-install-network-teleop.sh`. This keeps the lerobot repo clean
while making the modules importable as `lerobot.teleoperators.so_network_leader`.

## Usage (on follower Pi)

```bash
# Single arm
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.id=bmh_follower_left \
    --teleop.type=so_network_leader \
    --teleop.remote_ip=192.168.0.200 \
    --teleop.port_zmq=5555

# Bimanual
# --robot.id=bmh_follower makes bi_so_follower look for calibration files
# bmh_follower_left.json and bmh_follower_right.json (written by 03-calibrate.sh)
lerobot-teleoperate \
    --robot.type=bi_so_follower \
    --robot.id=so_follower \
    --robot.left_arm_config.port=/dev/ttyACM0 \
    --robot.right_arm_config.port=/dev/ttyACM1 \
    --teleop.type=bi_so_network_leader \
    --teleop.left_arm_config.remote_ip=192.168.0.200 \
    --teleop.left_arm_config.port_zmq=5555 \
    --teleop.right_arm_config.remote_ip=192.168.0.200 \
    --teleop.right_arm_config.port_zmq=5557


lerobot-record \
    --robot.type=bi_so_follower \
    --robot.id=so_follower \
    --robot.left_arm_config.port=/dev/ttyACM0 \
    --robot.right_arm_config.port=/dev/ttyACM1 \
    --robot.left_arm_config.cameras='{"front": {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPEG", "warmup_s": 5}, "left_wrist": {"type": "opencv", "index_or_path": 2, "width": 320, "height": 240, "fps": 30, "fourcc": "MJPEG", "warmup_s": 5}}' \
    --robot.right_arm_config.cameras='{"right_wrist": {"type": "opencv", "index_or_path": 0, "width": 320, "height": 240, "fps": 30, "fourcc": "MJPEG", "warmup_s": 5}}' \
    --teleop.type=bi_so_network_leader \
    --teleop.left_arm_config.remote_ip=192.168.0.200 \
    --teleop.left_arm_config.port_zmq=5555 \
    --teleop.right_arm_config.remote_ip=192.168.0.200 \
    --teleop.right_arm_config.port_zmq=5557 \
    --dataset.repo_id=my_local_sets/fun_set_2 \
    --dataset.num_episodes=5 \
    --dataset.single_task="Pick up the Controller" \
    --dataset.encoder_threads=2 \
    --dataset.push_to_hub=false \
    --play_sounds=false

```

> **Live preview while recording (BMH extra).** Every `opencv` camera entry also accepts optional
> `preview_path`, `preview_fps` (default 3), `preview_width` (default 320) and `preview_quality`
> (default 60). When `preview_path` is set, the camera's read thread writes a downscaled JPEG to that
> path up to `preview_fps` times per second (atomic replace — put it on tmpfs such as `/dev/shm`) so
> an external process can show a live view without opening the device a second time. The
> controller-app sets `"preview_path": "/dev/shm/bmh-101/preview-front.jpg"` on the front camera.
> Implementation: `src/lerobot/cameras/preview.py`.

## Fast startup (BMH extras)

Two changes cut the follower's startup from ~38 s to ~10 s on a Pi 5 (measured 2026-10-14):

- **Parallel connect.** `SOFollower.connect` opens its cameras side by side
  (`lerobot.cameras.connect_cameras`), and `BiSOFollower.connect` connects the two arms in
  parallel. Each `OpenCVCamera.connect` blocks for its full `warmup_s`, so three cameras at
  `warmup_s: 5` used to cost 16.5 s in sequence; with `warmup_s: 2` and parallel connect it is
  ~2.5 s. On failure the devices that did connect are released again and the first error is
  re-raised. Any script using these robots (`lerobot-record`, `lerobot-teleoperate`,
  `bmh-groot-client`) benefits.
- **`bmh-record --config <file.json>`** (`bmh/scripts/bmh_record.py`). `lerobot-record`'s
  draccus CLI parse builds an argparse tree over every registered robot / teleoperator /
  policy config and costs ~5.5 s per start on the Pi. `bmh-record` reads the same configuration
  from a JSON file (nested dicts, ChoiceRegistry fields carry a `type` key — the layout draccus
  uses for config files) and decodes it in milliseconds, then calls the unmodified `record()`.
  The controller-app writes `~/.cache/bmh-101/recording-config.json` before each session.

```bash
# Equivalent of the lerobot-record command above, as JSON:
cat > /tmp/record.json <<'JSON'
{
  "robot": {"type": "bi_so_follower", "id": "so_follower",
            "left_arm_config": {"port": "/dev/ttyACM0", "with_head": true,
                                "cameras": {"front": {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30, "fourcc": "MJPG", "warmup_s": 2}}},
            "right_arm_config": {"port": "/dev/ttyACM1", "cameras": {}}},
  "teleop": {"type": "bi_so_network_leader",
             "left_arm_config": {"remote_ip": "192.168.0.200", "port_zmq": 5555, "with_head": true},
             "right_arm_config": {"remote_ip": "192.168.0.200", "port_zmq": 5557}},
  "dataset": {"repo_id": "my_local_sets/fun_set_2", "num_episodes": 5, "single_task": "Pick up the Controller",
              "encoder_threads": 2, "push_to_hub": false},
  "play_sounds": false
}
JSON
bmh-record --config /tmp/record.json
```

## Teleop group locks (BMH extra)

`bi_so_network_leader` can **freeze the left arm, the right arm and the head independently**
(any combination) while teleoperating or recording. A frozen group keeps commanding the pose it
had when the lock engaged, no matter what the leader does; on release it eases back to the live
leader pose over `unlock_blend_s` seconds (smoothstep, default 0.8 s) instead of jumping.
Locking a group again mid-blend holds the blended pose.

- Flags: `--teleop.lock_file=<path>` enables it (default `None` = off, upstream behaviour);
  `--teleop.unlock_blend_s=0.8` tunes the release, `--teleop.home_blend_s=2.0` the move to and
  from the Home position (see below).
- State file: a JSON object `{"left": bool, "right": bool, "head": bool}`, replaced atomically
  (tmp + rename) by whoever drives the UI. The controller-app writes
  `~/.cache/bmh-101/teleop-locks.json` and passes that path on both `lerobot-teleoperate` and
  `lerobot-record`. The teleoperator does one `os.stat` per loop tick and only re-reads the file
  when it changed; a missing file means "all unlocked", a malformed one is logged once and ignored.
- Groups: `left` = the 7 `left_*.pos` arm keys incl. gripper, `right` = the 7 `right_*.pos` keys,
  `head` = `left_head_pan.pos` / `left_head_tilt.pos` (the head rides the left stream, hence the
  explicit key sets). Every change is logged as
  `Teleop locks: left=on right=off head=off | head jitter: pan=+0.0 tilt=+0.0 | home: left`.
- **Head jitter:** the state file may also carry `"head_jitter": {"pan": deg, "tilt": deg}`. A
  *frozen* head is then held at its frozen pose plus that offset (degrees, clamped to ±15) and eases
  to every new offset over `unlock_blend_s`; the offset is always relative to the frozen pose, so it
  never accumulates. It is ignored while the head is not frozen, and a file without the key means no
  offset. The controller-app writes a fresh random offset when the inter-episode reset starts
  ("Jitter head" in the recording panel), so a dataset covers small head-pose differences.
- **Home position (lock pose):** the state file may also carry `"lock_pose": {action_key: value}`
  (the robot's saved Home position; the controller-app lists the keys of every group set to
  "Home"). A held group whose keys are listed eases from wherever it is to that pose over
  `home_blend_s` (smoothstep) instead of freezing in place, then holds it; a head jitter is added
  on top of it (Home + offset). When the pose appears, changes or disappears the hold is re-based
  without a jump - a group whose pose entries vanish stays where it is (freeze). Release blends
  from the pose to the leader over `home_blend_s` too: Home is usually far from where the leader
  is held, and `unlock_blend_s` would make that a fast move. Values must be finite numbers; a
  malformed `lock_pose` rejects that file version (logged once). An older fork ignores the key, so
  "Home" then degrades to a plain freeze.
- **Recording:** the lock runs inside `get_action()`, so `lerobot-record` stores the held / blended
  pose as the dataset `action` - the same values the follower was commanded, and consistent with
  `observation.state`. Locks also hold through the inter-episode reset phase.
- Implementation: `src/lerobot/teleoperators/bi_so_network_leader/group_lock.py`
  (`ActionGroupLock`, `LockFileWatcher`); tests in `tests/bmh/test_group_lock.py`.

```bash
# Manual toggle from any shell on the follower Pi (the app does this for you):
printf '{"left": true, "right": false, "head": true}' > /tmp/locks.tmp && mv /tmp/locks.tmp ~/.cache/bmh-101/teleop-locks.json
```

## Home position (BMH extra)

The controller-app keeps one saved **Home position** for the follower: all 16 joints (both arms
and the head) as normalized action values (`left_shoulder_pan.pos` ... `left_head_tilt.pos`,
`right_*.pos`; degrees for joints, 0-100 for grippers). Three pieces of the fork serve it:

- **Capture - `bmh-read-pose`** (`bmh/scripts/bmh_read_pose.py`). Opens only the two serial buses
  (no cameras, no `configure()`, torque never changed), reads `Present_Position` of every motor and
  prints one line `POSE {json}` with the 16 keys; errors print `ERROR <message>` and exit 1. If a
  bus's calibration registers differ from the file it rewrites them only while every motor is limp
  (torque off), and refuses otherwise. The app runs it while no session uses the arms, after the
  operator posed the robot by hand.

  ```bash
  bmh-read-pose --left_port /dev/ttyACM1 --right_port /dev/ttyACM0 --id so_follower
  ```

- **Teleop / recording - `lock_pose`** in the lock state file (see "Teleop group locks" above).
- **Inference - `home`** on an idle document of the `bmh-groot-client` control file
  (`~/.cache/bmh-101/inference-control.json`):
  `{"seq": 6, "idle": true, "home": {"left_shoulder_pan.pos": 0.5, ...}}`. The client stiffens the
  motors if they were relaxed, ramps from the pose it last commanded (or the present one) to
  `home` over `--home_ramp_s` (default 2.5 s, smoothstep), then holds it. It reports the seq like
  any idle target (`BMH_TARGET {..., "home": true, "home_reached": false}`) and logs a second line
  with `"home_reached": true` on arrival - always after the first one, even when the arms arrive
  before the inference worker got to the seq (it may still be busy with an older one). Keys the robot has no action for are dropped with a
  warning; `home` together with `"relaxed": true` is rejected; any newer document stops the ramp
  where the robot is. Contract details: `bmh/groot_client/control.py`.
