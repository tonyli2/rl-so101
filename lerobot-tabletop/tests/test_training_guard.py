"""Reject quarantined data after native CLI/config resolution, before training."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import train_act


class TrainingGuardTests(unittest.TestCase):
    def test_resolved_quarantine_root_is_rejected_before_native_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "QUARANTINED.json").write_text("{}")
            config = SimpleNamespace(dataset=SimpleNamespace(root=str(root)))
            with patch.object(train_act, "train") as native:
                with self.assertRaisesRegex(SystemExit, "quarantined"):
                    train_act.guarded_train.__wrapped__(config)
                native.assert_not_called()

    def test_retained_root_delegates_the_same_resolved_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = SimpleNamespace(dataset=SimpleNamespace(root=temporary))
            with patch.object(train_act, "train") as native:
                train_act.guarded_train.__wrapped__(config)
                native.assert_called_once_with(config)

    def test_historical_backup_is_not_accidentally_used_for_training(self):
        config = SimpleNamespace(
            dataset=SimpleNamespace(root="/tmp/dataset-backups/original")
        )
        with patch.object(train_act, "train") as native:
            with self.assertRaisesRegex(SystemExit, "Historical"):
                train_act.guarded_train.__wrapped__(config)
            native.assert_not_called()
