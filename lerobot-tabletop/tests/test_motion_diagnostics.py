"""Distinguish chunk handoff jumps from within-chunk noise without a GPU."""

import json
import unittest

import numpy as np
from learning.motion_metrics import analyze_motion
from recording.schema import JOINTS


class MotionMetricTests(unittest.TestCase):
    def analyze(self, first_joint, chunk_starts=(0, 4, 8), fps=30):
        values = np.asarray(first_joint, dtype=np.float64)
        targets = np.zeros((len(values), 6))
        targets[:, 0] = values
        positions = np.zeros((len(values) + 1, 6))
        velocities = np.zeros_like(positions)
        starts = np.zeros(len(values), dtype=bool)
        starts[list(chunk_starts)] = True
        return analyze_motion(targets, positions, velocities, starts, fps=fps)

    def test_jumps_only_at_handoffs_are_assigned_to_boundaries(self):
        report = self.analyze(np.repeat([0.0, 10.0, 20.0], 4))
        self.assertEqual(report["steps"], 12)
        self.assertEqual(report["boundary_steps"], [4, 8])
        delta = report["joints"][JOINTS[0]]["target_delta"]
        self.assertEqual(delta["boundary"]["count"], 2)
        self.assertEqual(delta["interior"]["count"], 9)
        self.assertAlmostEqual(delta["boundary"]["rms"], 10.0)
        self.assertAlmostEqual(delta["boundary"]["energy"], 200.0)
        self.assertAlmostEqual(delta["boundary"]["energy_share"], 1.0)
        self.assertAlmostEqual(delta["interior"]["energy"], 0.0)
        # Keep physical units separate: another joint must not inherit this signal.
        self.assertEqual(
            report["joints"][JOINTS[1]]["target_delta"]["interior"]["energy"],
            0.0,
        )

    def test_alternation_inside_chunks_is_not_a_handoff_jump(self):
        report = self.analyze(np.tile([0.0, 1.0, -1.0, 0.0], 3))
        delta = report["joints"][JOINTS[0]]["target_delta"]
        self.assertAlmostEqual(delta["boundary"]["energy"], 0.0)
        self.assertAlmostEqual(delta["interior"]["energy"], 18.0)
        self.assertAlmostEqual(delta["interior"]["rms"], np.sqrt(2.0))
        self.assertAlmostEqual(delta["interior"]["energy_share"], 1.0)

    def test_constant_speed_is_not_misreported_as_boundary_jitter(self):
        report = self.analyze(np.arange(12, dtype=float))
        joint = report["joints"][JOINTS[0]]
        self.assertEqual(joint["target_delta"]["boundary"]["rms"], 1.0)
        self.assertEqual(joint["target_delta"]["interior"]["rms"], 1.0)
        for kind in ("target_acceleration", "target_jerk"):
            for group in ("boundary", "interior"):
                self.assertEqual(joint[kind][group]["energy"], 0.0)

    def test_acceleration_stencil_marks_both_sides_of_a_handoff(self):
        report = self.analyze(np.repeat([0.0, 10.0, 20.0], 4), fps=30)
        acceleration = report["joints"][JOINTS[0]]["target_acceleration"]
        # At each jump, acceleration has an impulse at b and an opposite one
        # at b+1. Both belong to the handoff, not to the chunk interior.
        self.assertEqual(acceleration["boundary"]["count"], 4)
        self.assertAlmostEqual(acceleration["boundary"]["rms"], 10.0 * 30**2)
        self.assertEqual(acceleration["interior"]["energy"], 0.0)
        jerk = report["joints"][JOINTS[0]]["target_jerk"]
        self.assertEqual(jerk["boundary"]["count"], 6)
        self.assertEqual(jerk["interior"]["energy"], 0.0)

    def test_initial_target_offset_is_not_a_chunk_handoff(self):
        report = self.analyze(np.full(12, 1000.0), chunk_starts=(0,))
        self.assertEqual(report["boundary_steps"], [])
        delta = report["joints"][JOINTS[0]]["target_delta"]
        self.assertEqual(delta["boundary"]["count"], 0)
        self.assertEqual(delta["interior"]["count"], 11)
        self.assertEqual(delta["interior"]["energy"], 0.0)
        # Empty populations must not introduce JSON NaN/Infinity into reports.
        json.dumps(report, allow_nan=False)

    def test_large_interior_reversals_are_counted_outside_boundary_windows(self):
        values = np.tile([0.2, -0.2], 20)
        report = self.analyze(values, chunk_starts=(0, 20))
        reversal = report["joints"][JOINTS[0]]["interior_reversals"]
        eligible = [i for i in range(4, len(values) - 1) if abs(i - 20) > 3]
        self.assertEqual(reversal["threshold"], 0.1)
        self.assertEqual(reversal["eligible_centers"], len(eligible))
        # Every eligible center reverses direction by 0.4 native units.
        self.assertEqual(reversal["count"], len(eligible))
        self.assertGreater(reversal["per_second"], 0.0)
        self.assertTrue(np.isfinite(reversal["per_second"]))

    def test_boundary_staircase_has_no_interior_reversals(self):
        report = self.analyze(np.repeat([0.0, 10.0], 20), chunk_starts=(0, 20))
        reversal = report["joints"][JOINTS[0]]["interior_reversals"]
        self.assertGreater(reversal["eligible_centers"], 0)
        self.assertEqual(reversal["count"], 0)
        self.assertEqual(reversal["per_second"], 0.0)

    def test_tiny_interior_alternation_is_below_reversal_threshold(self):
        report = self.analyze(np.tile([0.01, -0.01], 20), chunk_starts=(0, 20))
        reversal = report["joints"][JOINTS[0]]["interior_reversals"]
        self.assertGreater(reversal["eligible_centers"], 0)
        self.assertEqual(reversal["count"], 0)
        self.assertEqual(reversal["per_second"], 0.0)

    def test_reversal_at_boundary_is_excluded(self):
        values = np.zeros(40)
        values[20] = 10.0
        report = self.analyze(values, chunk_starts=(0, 20))
        reversal = report["joints"][JOINTS[0]]["interior_reversals"]
        self.assertEqual(reversal["count"], 0)

    def test_state_oscillation_is_visible_when_targets_are_constant(self):
        count = 12
        targets = np.zeros((count, 6))
        positions = np.zeros((count + 1, 6))
        velocities = np.zeros_like(positions)
        velocities[:, 0] = np.tile([0.0, 1.0, -1.0], 5)[: count + 1]
        starts = np.zeros(count, dtype=bool)
        starts[[0, 4, 8]] = True
        report = analyze_motion(targets, positions, velocities, starts)
        joint = report["joints"][JOINTS[0]]
        self.assertEqual(joint["target_delta"]["interior"]["energy"], 0.0)
        self.assertGreater(
            sum(
                joint["state_acceleration"][group]["energy"]
                for group in ("boundary", "interior")
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
