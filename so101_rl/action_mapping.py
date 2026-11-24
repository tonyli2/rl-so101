"""Policy action -> joint targets for "relative" actions. One implementation for sim and hardware.

The policy outputs a small move per joint in [-1, 1]; the target is ``base + action * scale``.
What the move is added to (the "base") decides how the policy copes with control latency:

- ``"measured"``: the joint readings. What the policies were trained with; fails at 2+ steps of
  latency (the real arm has ~2.3-3.3) because the readings are stale.
- ``"command"``: the previous target. Immune to latency, but winds up on a blocked joint: with
  the gripper closed on the cube the policy keeps saying "close a bit more" and the target
  runs to the joint limit (~18 deg past the fingers).
- ``"resync"`` (deploy with this): the previous target, but per joint the target sent ``lag``
  steps ago is compared with the reading now (with the delay, that is where the joint should
  be). A gap above ``threshold`` means the joint is blocked, so its base snaps to its reading
  and stays there until the reading moves again (latch).

Units: whatever ``q``, ``scale``, ``lo``/``hi`` and thresholds share. The sim uses actuator
radians; a hardware runner converts LeRobot degrees to those radians first (So101World.deg_to_ctrl)
because the policy's observation needs them anyway.

    mapper = ActionMapper.from_config(obs_config, lo, hi)   # obs_config: bc_config.json["obs"]
    mapper.reset(q0)
    target = mapper(action, q_measured)                     # every control step
"""

from __future__ import annotations

from collections import deque

import numpy as np

BASES = ("measured", "command", "resync")
RESYNC_DEFAULTS = {
    "lag": 3,  # control steps; matches the measured real-arm delay (~2.3-3.3 steps)
    # Normal tracking gap |q(t) - target(t-3)| in successful sim episodes stays under ~7.5 deg on
    # the arm joints at any latency 0-3; the gripper's is tighter to catch the squeeze.
    "threshold_deg": [10.0, 10.0, 10.0, 10.0, 10.0, 5.0],
    "moving_deg": 1.0,  # a blocked joint is released once its reading changes by more than this
    "latch": True,
}


class ActionMapper:
    def __init__(self, scale, lo, hi, base: str = "measured", lag: int = 3,
                 threshold_deg=RESYNC_DEFAULTS["threshold_deg"], moving_deg: float = 1.0, latch: bool = True):
        if base not in BASES:
            raise ValueError(f"base must be one of {BASES}, got {base!r}")
        self.scale = np.asarray(scale, dtype=np.float64)
        self.lo, self.hi = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
        self.base = base
        self.lag = int(lag)
        self.threshold = np.deg2rad(np.asarray(threshold_deg, dtype=np.float64))
        self.moving = np.deg2rad(float(moving_deg))
        self.latch = latch
        self._sent: deque = deque(maxlen=max(self.lag, 1))
        self._readings: deque = deque(maxlen=max(self.lag, 1))
        self.reset(np.zeros_like(self.scale))

    @classmethod
    def from_config(cls, obs: dict, lo, hi) -> "ActionMapper":
        """From a run's observation/action config (bc_config.json "obs"). Configs saved before
        the base existed mean "measured", the setting those policies were trained with."""
        if obs.get("action") != "relative":
            raise ValueError("ActionMapper handles relative actions only")
        r = {**RESYNC_DEFAULTS, **(obs.get("resync") or {})}
        return cls(obs["relative_scale"], lo, hi, base=obs.get("relative_base", "measured"), **r)

    def reset(self, q) -> None:
        self.prev = np.asarray(q, dtype=np.float64).copy()
        self._sent.clear()
        self._readings.clear()
        self.blocked = np.zeros(len(self.scale), bool)
        self.resync_count = np.zeros(len(self.scale), int)

    def __call__(self, action, q) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        base = q.copy() if self.base == "measured" else self.prev.copy()
        if self.base == "resync" and len(self._sent) == self.lag:
            off = np.abs(q - self._sent[0]) > self.threshold  # _sent[0]: the target from `lag` steps ago
            if self.latch:
                moving = np.abs(q - self._readings[0]) > self.moving
                self.blocked = (self.blocked & ~moving) | off
                off = self.blocked
            base[off] = q[off]
            self.resync_count += off
        target = np.clip(base + a * self.scale, self.lo, self.hi)
        self._sent.append(target.copy())
        self._readings.append(q.copy())
        self.prev = target
        return target
