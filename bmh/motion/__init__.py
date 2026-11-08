"""Fixed recordings for the BMH-101: joint trajectories recorded once, played back exactly.

A fixed recording is recorded by teleoperation and replayed without a policy or cameras.

- :mod:`bmh.motion.file` — the ``motion.json`` format (load / save / validate) and where
  recordings live (``~/.cache/bmh-101/fixed-recordings/<id>/``).
- :mod:`bmh.motion.player` — turns (present pose, recording, start mode) into a
  time-indexed ``pose_at(t)``; shared by ``bmh-replay-motion`` and ``bmh-policy-client``.

Both are stdlib-only (no hardware imports), so they are unit-testable anywhere.
``bmh.policy_client.control`` builds on them (replay targets), never the other way round.
"""
