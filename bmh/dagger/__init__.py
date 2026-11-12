"""DAgger recording for the BMH-101 (`bmh-dagger`, see `bmh/scripts/bmh_dagger.py`).

A policy runs on the robot; the operator pauses it, optionally rewinds, takes over with
the leader arms and finishes the task. Only that correction is written to the dataset.

- `protocol`: the stdin command lines and stdout status lines shared with the controller-app.
- `history`: ring buffer of the actions sent, for rewinding.
- `takeover`: leader / follower alignment and the blend into teleoperation.
- `policy_runner`: the policy client's chunk loop as a class that can be suspended.
- `recorder`: the staging dataset, built exactly like `lerobot-record` builds it.
- `leader_assist`: runs on the leader Pi (inside `so_leader_host`): pulls the leader arms
  lightly toward the pose the robot holds while the operator lines them up.
"""
