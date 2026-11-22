"""Per-joint chunk-boundary diagnostics; derivatives retain physical time units."""

import numpy as np
from recording.schema import JOINTS


def analyze_motion(targets, positions, velocities, chunk_starts, fps=30):
    """Analyze T targets and T+1 measured states; exclude startup and mixed stencils.

    A difference at t is a[t]-a[t-1]. An order-k stencil is a boundary
    stencil when any of its k transitions crosses a query boundary. No
    derivative uses padding, an initial target-to-state jump, or autoreset.
    """
    targets, positions, velocities = [
        np.asarray(x, dtype=float) for x in (targets, positions, velocities)
    ]
    starts = np.asarray(chunk_starts, dtype=bool)
    n = len(targets)
    if (
        targets.shape != (n, 6)
        or positions.shape != (n + 1, 6)
        or velocities.shape != (n + 1, 6)
    ):
        raise ValueError("Expected targets (T,6), positions/velocities (T+1,6)")
    if n < 4 or starts.shape != (n,) or fps <= 0:
        raise ValueError("Need at least four steps, positive fps and T boundary flags")
    if not all(np.isfinite(x).all() for x in (targets, positions, velocities)):
        raise ValueError("Motion must be finite")
    starts = starts.copy()
    starts[0] = False

    def stats(x, total=None):
        energy = float(np.sum(x * x))
        return {
            "count": len(x),
            "rms": float(np.sqrt(np.mean(x * x))) if len(x) else None,
            "p95": float(np.percentile(np.abs(x), 95)) if len(x) else None,
            "max": float(np.max(np.abs(x))) if len(x) else None,
            "energy": energy,
            "energy_share": energy / total if len(x) and total else None,
        }

    result = {
        "steps": n,
        "boundary_steps": np.flatnonzero(starts).tolist(),
        "joints": {},
    }
    for j, name in enumerate(JOINTS):
        out = {}
        for key, values, order, scale in (
            ("target_delta", targets[:, j], 1, 1),
            ("target_acceleration", targets[:, j], 2, fps**2),
            ("target_jerk", targets[:, j], 3, fps**3),
            ("state_acceleration", velocities[1:, j], 1, fps),
        ):
            delta = np.diff(values, n=order) * scale
            boundary = np.array(
                [starts[t - order + 1 : t + 1].any() for t in range(order, n)]
            )
            total = float(np.sum(delta * delta))
            out[key] = {
                "boundary": stats(delta[boundary], total),
                "interior": stats(delta[~boundary], total),
            }
        differences = np.diff(targets[:, j])
        centers = np.arange(1, n - 1)
        eligible = centers >= 4
        for b in np.flatnonzero(starts):
            eligible &= np.abs(centers - b) > 3
        reversal = (differences[:-1] * differences[1:] < 0) & (
            np.minimum(np.abs(differences[:-1]), np.abs(differences[1:])) > 0.1
        )
        count = int(np.sum(reversal & eligible))
        out["interior_reversals"] = {
            "threshold": 0.1,
            "count": count,
            "eligible_centers": int(eligible.sum()),
            "per_second": count * fps / int(eligible.sum()) if eligible.any() else 0.0,
        }
        out["tracking_error"] = stats(targets[:, j] - positions[1:, j])
        state_differences = np.diff(positions[1:, j])
        state_reversal = (state_differences[:-1] * state_differences[1:] < 0) & (
            np.minimum(np.abs(state_differences[:-1]), np.abs(state_differences[1:]))
            > 0.1
        )
        state_count = int(np.sum(state_reversal & eligible))
        out["state_interior_reversals"] = {
            "threshold": 0.1,
            "count": state_count,
            "eligible_centers": int(eligible.sum()),
            "per_second": state_count * fps / int(eligible.sum())
            if eligible.any()
            else 0.0,
        }
        # Physical motion can lag a target jump; compare +/-3-tick windows.
        acceleration = np.diff(velocities[:, j]) * fps
        near = np.zeros(n, dtype=bool)
        for b in np.flatnonzero(starts):
            near[max(1, b - 3) : min(n, b + 4)] = True
        valid = np.arange(n) >= 3  # exclude startup response
        total = float(np.sum(acceleration[valid] ** 2))
        out["state_acceleration_boundary_window"] = {
            "boundary": stats(acceleration[valid & near], total),
            "interior": stats(acceleration[valid & ~near], total),
        }
        result["joints"][name] = out
    return result
