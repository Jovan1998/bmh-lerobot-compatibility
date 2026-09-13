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

## Teleop group locks (BMH extra)

`bi_so_network_leader` can **freeze the left arm, the right arm and the head independently**
(any combination) while teleoperating or recording. A frozen group keeps commanding the pose it
had when the lock engaged, no matter what the leader does; on release it eases back to the live
leader pose over `unlock_blend_s` seconds (smoothstep, default 0.8 s) instead of jumping.
Locking a group again mid-blend holds the blended pose.

- Flags: `--teleop.lock_file=<path>` enables it (default `None` = off, upstream behaviour);
  `--teleop.unlock_blend_s=0.8` tunes the release.
- State file: a JSON object `{"left": bool, "right": bool, "head": bool}`, replaced atomically
  (tmp + rename) by whoever drives the UI. The controller-app writes
  `~/.cache/bmh-101/teleop-locks.json` and passes that path on both `lerobot-teleoperate` and
  `lerobot-record`. The teleoperator does one `os.stat` per loop tick and only re-reads the file
  when it changed; a missing file means "all unlocked", a malformed one is logged once and ignored.
- Groups: `left` = the 7 `left_*.pos` arm keys incl. gripper, `right` = the 7 `right_*.pos` keys,
  `head` = `left_head_pan.pos` / `left_head_tilt.pos` (the head rides the left stream, hence the
  explicit key sets). Every change is logged as `Teleop locks: left=on right=off head=off`.
- **Recording:** the lock runs inside `get_action()`, so `lerobot-record` stores the held / blended
  pose as the dataset `action` - the same values the follower was commanded, and consistent with
  `observation.state`. Locks also hold through the inter-episode reset phase.
- Implementation: `src/lerobot/teleoperators/bi_so_network_leader/group_lock.py`
  (`ActionGroupLock`, `LockFileWatcher`); tests in `tests/bmh/test_group_lock.py`.

```bash
# Manual toggle from any shell on the follower Pi (the app does this for you):
printf '{"left": true, "right": false, "head": true}' > /tmp/locks.tmp && mv /tmp/locks.tmp ~/.cache/bmh-101/teleop-locks.json
```
