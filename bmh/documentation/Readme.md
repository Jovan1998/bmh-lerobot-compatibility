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
    --teleop.left_arm_config.polling_timeout_ms=5 \
    --teleop.right_arm_config.remote_ip=192.168.0.200 \
    --teleop.right_arm_config.port_zmq=5557 \
    --teleop.right_arm_config.polling_timeout_ms=5 \
    --dataset.repo_id=my_local_sets/fun_set_2 \
    --dataset.fps=30 \
    --dataset.num_episodes=5 \
    --dataset.single_task="Pick up the Controller" \
    --dataset.encoder_threads=2 \
    --dataset.push_to_hub=false \
    --control_fps=60 \
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
  `bmh-policy-client`) benefits.
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
             "left_arm_config": {"remote_ip": "192.168.0.200", "port_zmq": 5555, "with_head": true, "polling_timeout_ms": 5},
             "right_arm_config": {"remote_ip": "192.168.0.200", "port_zmq": 5557, "polling_timeout_ms": 5}},
  "dataset": {"repo_id": "my_local_sets/fun_set_2", "fps": 30, "num_episodes": 5, "single_task": "Pick up the Controller",
              "encoder_threads": 2, "push_to_hub": false},
  "control_fps": 60,
  "play_sounds": false
}
JSON
bmh-record --config /tmp/record.json
```

`control_fps` (BMH extra, CLI `--control_fps=60`) is the rate at which the arms are commanded.
It defaults to `dataset.fps` (upstream behaviour) and must be a multiple of it: with 60 and a
30 fps dataset the loop reads the leader and commands the arms every 16.7 ms, like
`lerobot-teleoperate`, but observes the robot and records a frame only on every second tick, so
the dataset and its videos stay at 30 fps. Set `polling_timeout_ms: 5` on both leader arms with
it (the controller-app does): at 60 Hz the follower sometimes asks just before the next leader
message lands, and the default 15 ms wait per arm could then make a recorded frame late. On a
timeout the tick resends the last leader position.

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
  _frozen_ head is then held at its frozen pose plus that offset (degrees, clamped to ±15) and eases
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
- **Inference - `home`** on an idle document of the `bmh-policy-client` control file
  (`~/.cache/bmh-101/inference-control.json`):
  `{"seq": 6, "idle": true, "home": {"left_shoulder_pan.pos": 0.5, ...}}`. The client stiffens the
  motors if they were relaxed, ramps from the pose it last commanded (or the present one) to
  `home` over `--home_ramp_s` (default 2.5 s, smoothstep), then holds it. It reports the seq like
  any idle target (`BMH_TARGET {..., "home": true, "home_reached": false}`) and logs a second line
  with `"home_reached": true` on arrival - always after the first one, even when the arms arrive
  before the inference worker got to the seq (it may still be busy with an older one). Keys the robot has no action for are dropped with a
  warning; `home` together with `"relaxed": true` is rejected; any newer document stops the ramp
  where the robot is. Contract details: `bmh/policy_client/control.py`.

## Fixed recordings (BMH extra)

A **fixed recording** is a joint trajectory recorded once by teleoperation and played back
exactly — no policy, no cameras. The controller-app's Fixed recordings tab records, trims and
plays them; a robot script plays one between prompts with `playRecording("name")`. Code:
`bmh/motion/` (format + replay plan, stdlib only) and the two scripts below.

**Where they live.** `~/.cache/bmh-101/fixed-recordings/<id>/motion.json` (written once by the
recorder) next to the app's `bmh.meta.json` sidecar (name, trim range). `motion.json`, version 1:

```json
{
  "version": 1,
  "fps": 60,
  "keys": [
    "left_shoulder_pan.pos",
    "...",
    "left_head_tilt.pos",
    "right_shoulder_pan.pos",
    "...",
    "right_gripper.pos"
  ],
  "frames": [[0.0, 12.3, "..."], ["..."]],
  "recorded_at": 1762600000.0
}
```

`keys` are the 16 bimanual action keys (`bmh.motion.file.MOTION_KEYS`); one frame per control tick,
each the action actually sent to the follower (after the group-lock blend). Degrees for joints,
0-100 for grippers, 3 decimals; a 2-minute take is about 1 MB.

**Record - `bmh-record-motion --config <file.json>`** (`bmh/scripts/bmh_record_motion.py`). Same
JSON idea as `bmh-record` (decoded with `draccus.decode`, no CLI parse); `robot` / `teleop` are the
sections the app writes for `bmh-record`, here without cameras, plus `output` and `fps` (default 60):

```json
{
  "robot": {
    "type": "bi_so_follower",
    "id": "so_follower",
    "left_arm_config": { "port": "/dev/ttyACM1", "with_head": true },
    "right_arm_config": { "port": "/dev/ttyACM0" }
  },
  "teleop": {
    "type": "bi_so_network_leader",
    "left_arm_config": { "remote_ip": "192.168.1.2", "with_head": true },
    "right_arm_config": { "remote_ip": "192.168.1.2" },
    "lock_file": "/home/pi/.cache/bmh-101/teleop-locks.json"
  },
  "output": "/home/pi/.cache/bmh-101/fixed-recordings/wave-1a2b/motion.json",
  "fps": 60
}
```

Stdout: `MOTION_READY` (both sides connected), `MOTION_FRAMES <n>` about once per second,
`MOTION_SAVED <n>` after SIGTERM / SIGINT, `ERROR <message>` (exit 1). After saving, the follower's
torque goes off (`robot.disconnect()`) like at the end of any session. A signal before
`MOTION_READY` aborts without saving.

**Replay - `bmh-replay-motion`** (`bmh/scripts/bmh_replay_motion.py`), when nothing else holds the
ports. Opens only the buses (calibration check like `bmh-read-pose`), pins every goal to the present
position before enabling torque, plays, then holds the final pose until a newline / EOF on stdin or
SIGTERM, and turns the torque off. SIGUSR1 during playback stops and holds where the arms are.

```bash
bmh-replay-motion --left_port /dev/ttyACM1 --right_port /dev/ttyACM0 --id so_follower \
    --file ~/.cache/bmh-101/fixed-recordings/wave-1a2b/motion.json --start 12 --end 610 \
    --mode blend-in   # or goto-start; --ramp_s 2.5 --blend_s 1.0 --fps 60
```

Stdout: `REPLAY_PHASE ramping` (goto-start only) / `REPLAY_PHASE playing`, `REPLAY_DONE` (holding
the final pose), `REPLAY_PHASE holding` (after SIGUSR1), `ERROR <message>` (exit 1).

**Start modes** (`bmh/motion/player.py`, the same plan in both players). `goto-start`: smoothstep ramp
from the present pose to the first kept frame over `ramp_s`, then the recording. `blend-in`: the
recording starts at once, shifted by (present - first frame); the shift fades out over `blend_s`, so
the arms start where they are and slide onto the trajectory. Playback is time-indexed (linear
interpolation at `t * fps`), so a 60 Hz recording plays at its speed from a 30 Hz loop too.

**Inference - `replay`** on an idle document of the `bmh-policy-client` control file, for a script's
`playRecording()` while the Physical Agent holds the ports:
`{"seq": 7, "idle": true, "replay": {"path": ".../fixed-recordings/<id>/motion.json", "start": 12, "end": 610, "mode": "blend-in"}}`.
`path` must lie under `~/.cache/bmh-101/fixed-recordings/`, `0 <= start < end`; `replay` excludes
`relaxed: true` and `home`. The client stiffens the motors if they were relaxed, plays the range from
the pose it last commanded (`goto-start` ramps over `--home_ramp_s`, `blend-in` fades over
`--replay_blend_s`, default 1.0 s), then holds the final pose. It reports
`BMH_TARGET {..., "replay": true, "replay_done": false}` and, after the last frame, a second line with
`"replay_done": true` (always after the first). A file it cannot load or a range outside the
recording is answered with `"accepted": false` and the reason; the robot holds. Any newer document
stops the replay where the robot is.

## DAgger recording (BMH extra)

`bmh-dagger --config <file.json>` (`bmh/scripts/bmh_dagger.py`, logic in `bmh/dagger/`) runs a
**DAgger session**: a policy drives the robot, the operator pauses it, optionally rewinds the
robot along the motion it just made, takes over with the leader arms and finishes the task.
Only that correction is recorded, as one episode per take-over.

One process owns the robot for the whole session, because `bmh-policy-client` and `bmh-record`
both need the serial ports and the cameras and take ~10 s to start. It connects follower,
cameras, leader stream and dataset writer once and switches between policy, hold, rewind and
teleoperation inside one loop. `bmh-policy-client` and `lerobot-record` are not changed: the
adapter, the policy connect, the inference worker, look-ahead and smoothing are imported from
the policy client as they are, and its chunk swap / refetch logic is ported into
`bmh/dagger/policy_runner.py` (same variables, same tests).

**Config.** `robot`, `teleop` and `dataset` are exactly the blocks of `bmh-record`, so the
staging dataset has the features of a recording staging dataset and the app's converter merges
it unchanged:

```json
{
  "robot": { "type": "bi_so_follower", "...": "as for bmh-record" },
  "teleop": { "type": "bi_so_network_leader", "...": "as for bmh-record" },
  "dataset": { "repo_id": "local/my-set", "root": "/path/to/recording-staging", "...": "as for bmh-record" },
  "control_fps": 60,
  "loop_file": "/home/pi/.cache/bmh-101/inference-loop.json",
  "jpeg_quality": 80
}
```

| Key | Default | Meaning |
| --- | --- | --- |
| `control_fps` | 60 | Arm command rate while the operator steers; a multiple of `dataset.fps` (a frame every `control_fps / dataset.fps`-th command, as in `bmh-record`). |
| `loop_file` | none | The Physical Agent's loop-settings file. Polled every tick and answered with the same `BMH_LOOP` lines as `bmh-policy-client`. |
| `jpeg_quality` | 80 | Camera JPEG quality of the policy requests. |
| `history_s` | 60 | Seconds of sent actions kept for rewinding (30 frames per second). |
| `align_arm_tol_deg` / `align_gripper_tol` | 4 / 5 | The leader counts as aligned when every joint is this close to the held pose (degrees; 0..100 for grippers). |
| `align_hold_s` | 0.4 | Aligned for this long without a gap connects the arms by itself (with `auto_start`). |
| `takeover_blend_s` | 1.0 | Shortest fade of the leader / follower offset when the arms connect. A larger gap stretches it (45 deg/s at most). |
| `takeover_settle_s` | 3.0 | The follower follows the leader for this long before the take's first frame is written. 0 writes from the connection on. |
| `home_ramp_s` | 2.5 | Duration of the Home ramp. |
| `probe_timeout_ms` / `timeout_ms` | 5000 / 15000 | Policy server timeouts, as for `bmh-policy-client`. |

**Phases.**

| Phase | Robot | Leaves on |
| --- | --- | --- |
| `connecting` | not connected yet | both sides connected: `paused` (prints `DAGGER_READY`) |
| `paused` | holds `history[cursor]` | `run`, `step`, `home`, `align`, `finish` |
| `running` | policy chunks through look-ahead and smoothing | `pause`, `align`; a policy error: `paused` |
| `stepping` | one history frame per tick toward the cursor | arrival: `paused`; `pause`, `run`, `step`, `home`, `align` |
| `homing` | smoothstep ramp to the Home pose | arrival: `paused` (the history restarts there); `pause` |
| `aligning` | holds; reads the leader, reports `leader - hold` | aligned for `align_hold_s` with `auto_start`, or `record`: `settling`; `cancel`: `paused` |
| `settling` | follows the leader (blended in), writes nothing | `takeover_settle_s` later: `recording`; `cancel`: `paused` (holds where the leader took it) |
| `recording` | follows the leader, writes frames | `save` / `discard`, or `dataset.episode_time_s` reached: `saving` |
| `saving` | servos keep their last goal (the loop blocks in `save_episode`) | done: `paused` |
| `finishing` | `dataset.finalize()`, disconnect | exit 0 |

**Commands**, one JSON object per line on stdin, each with an integer `id`:

```json
{"id": 1, "cmd": "policy", "policy_host": "1.2.3.4", "policy_port": 5555, "api_token": "...", "lang_instruction": "put the cube in the bowl"}
{"id": 2, "cmd": "run"}
{"id": 3, "cmd": "pause"}
{"id": 4, "cmd": "step", "cursor": 45}
{"id": 5, "cmd": "home", "pose": {"left_shoulder_pan.pos": 0.0}}
{"id": 6, "cmd": "align", "task": "put the cube in the bowl", "auto_start": true, "assist": false, "assist_torque_pct": 10}
{"id": 7, "cmd": "record"}
{"id": 8, "cmd": "cancel"}
{"id": 9, "cmd": "save"}
{"id": 10, "cmd": "discard"}
{"id": 11, "cmd": "finish"}
{"id": 12, "cmd": "assist", "assist": true, "assist_torque_pct": 15}
```

stdin, not a polled file like the policy client's control file: these are events, and two of
them inside one tick must both arrive, in order. The API token never touches the disk, and
stdin EOF (the app died) ends the session instead of leaving a policy running unsupervised. A
command that is not valid in the current phase is refused (`ok: false` with the reason), never
queued. `cursor` is "frames back from the newest history frame"; it is absolute, and the loop
walks toward it by one frame per tick, so lost or repeated key events cannot make the robot
jump. `run` with a policy that had failed asks for it again. `assist` (only while `aligning`)
switches the leader assist on or off or changes its torque; `align` sets it for the take-over.

**Output** the app parses. The `DAGGER_READY`, `BMH_DAGGER`, `BMH_DAGGER_ACK` and `ERROR` lines are
printed on stdout; `BMH_TARGET` and `BMH_LOOP` are log lines like everything else and go to stderr.
The app reads the final `ERROR` line from stdout only, because a log line of level ERROR starts
the same way:

```text
DAGGER_READY
BMH_DAGGER_ACK {"id": 6, "ok": true}
BMH_DAGGER_ACK {"id": 7, "ok": false, "error": "not aligning"}
BMH_DAGGER {"phase": "aligning", "policy": "ready", "history": 412, "cursor": 0, "episodes": 3, "hold": {"left_shoulder_pan.pos": 12.3}, "align": {"diff": {"left_shoulder_pan.pos": -3.2}, "aligned": false, "leader_age_ms": 14, "assist": "off"}, "take": {"frames": 0}}
BMH_DAGGER {"phase": "settling", "policy": "ready", "history": 413, "cursor": 0, "episodes": 3, "take": {"frames": 0, "record_in_s": 2.35}}
BMH_TARGET {...}   as bmh-policy-client, logged by the reused inference worker
BMH_LOOP {...}     as bmh-policy-client
ERROR <message>    then exit 1
```

`BMH_DAGGER` goes out twice a second, ten times a second while `aligning` and `settling`, and
once on every phase change. `policy` is `none | connecting | ready | error` (the reason of an
error is in the `BMH_TARGET` line). `hold` is present while `paused` and `aligning`, `align` while
`aligning`, `take` from `aligning` to `saving` (with `record_in_s`, the seconds until the first
frame is written, while `settling`). Values are rounded to 2 decimals.

**How a take starts.** The reference for the alignment is the pose the follower was last
*commanded*, not the pose it measures: a gripper squeezing an object never reaches its goal, and
aligning to the measured position would relax the grip the moment teleoperation starts. When
the arms connect, the first teleop action is shifted by `hold - leader`, so it equals the hold
pose exactly, and that shift fades out with a smoothstep; after the fade the leader's action
passes through untouched. The arms connect first and the recording starts `takeover_settle_s`
(3 s) later: the connection has a bounce (the fade, the leader assist letting go, the operator
taking the arms' weight) that is not part of the correction. Those 3 s still go into the rewind
history. The action recorded is the action sent. The arms never connect on a leader stream that
has been silent for more than 0.5 s (`SONetworkLeader.last_message_age_s`; `get_action()` itself
keeps returning the cached pose).

**How a session ends.** `finish`, stdin EOF, SIGTERM / SIGINT and any error take the same path:
an unsaved take is discarded, the dataset is finalized, robot and leader are disconnected. An
error (for example a servo that drops off the bus) then prints `ERROR ...` and exits 1, so the
episodes saved before it survive.

**Leader assist.** While the operator lines the leader arms up with the robot, the leader arms
can pull lightly toward the pose the robot holds, so the operator feels where to go. With
`assist: true` and `assist_torque_pct` (1..20, percent of full torque: 10 is a `Torque_Limit` of
100), every `aligning` tick sends the hold pose of the arm joints and grippers to the leader host,
one ZMQ PUSH socket per arm (`SONetworkLeader.send_feedback`). Arms whose group is locked get a
release instead, and the head is never driven. The moment `aligning` ends (the arms connect,
`cancel`, `finish`, an error) the leader is released, so the operator must be holding the arms
when they connect; the 3 s of `settling` before the recording absorb the drop. The status's `assist` is `on`, `off`, or `unavailable` when the leader stream
has no feedback channel.

It needs the channel on both sides:

- leader Pi: `so_leader_host.py ... --left_feedback_port 5556 --right_feedback_port 5558`
  (optionally `--assist_max_torque_limit`, default 200). The host binds one PULL socket per arm.
- follower: `feedback_port_zmq` in both arm configs of the `bi_so_network_leader` teleop
  (`"left_arm_config": {..., "feedback_port_zmq": 5556}`, `5558` on the right).

Messages, JSON, latest only: `{"goal": {"shoulder_pan.pos": 12.0, ...}, "torque_limit": 100}`
and `{"release": true}`. The follower's PUSH socket conflates; the host's PULL socket does not
and reads to the latest message each tick, because a conflating receiver drops a message it has
not read yet when the sender disconnects, and that is the follower's last `release`. The rules are enforced on the leader
(`bmh/dagger/leader_assist.py`), not by the sender: the limit is capped at
`--assist_max_torque_limit` and written to every joint before any torque comes on; the joints come
on one at a time, 50 ms apart, each pinned to where it is, without stalling the position stream;
the goal walks toward the target at 60 degrees (gripper: 60 %) per second; 0.3 s without a
message turns the torque off by itself; a bus error releases. On release the torque goes off
first and the limits read before the first engage are written back.

**Bench test for the leader assist.** `bmh-center-arm --type leader ... --torque_limit 100`
drives a leader arm weakly: `Torque_Limit` is written to every motor before any torque comes on,
the motors are enabled one at a time 50 ms apart, and the previous limits are written back on
release. Run on both leader arms at once with 100 and 200 (2026-11-12): the leader Pi stayed up,
which is why the assist is capped at 200 (20 %).
