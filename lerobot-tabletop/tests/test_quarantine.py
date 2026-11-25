"""Guard the dataset migration's atomic commit and source-change protection."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from learning.quarantine import commit, digest


class QuarantineCommitTests(unittest.TestCase):
    def setup_layout(self, root):
        source = root / "recordings/sim-baseline"
        quarantine = root / "recordings/archived-sim-baseline-collision"
        backup = root / "outputs/original-50"
        stage = root / "outputs/staging"
        for path, payload in (
            (source, "original"),
            (stage / source.name, "retained"),
            (stage / quarantine.name, "quarantined"),
        ):
            (path / "meta").mkdir(parents=True)
            (path / "meta/info.json").write_text(payload)
        report = {
            "status": "PREPARED_NOT_COMMITTED",
            "source": str(source),
            "quarantine": str(quarantine),
            "original_backup": str(backup),
            "source_info_sha256": digest(source / "meta/info.json"),
        }
        (stage / "migration.json").write_text(json.dumps(report))
        return source, quarantine, backup, stage

    def test_failed_second_install_restores_original_and_retains_prepared_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, quarantine, backup, stage = self.setup_layout(Path(tmp))
            original_rename = os.rename

            def fail_quarantine_install(src, dst):
                if Path(src) == stage / quarantine.name:
                    raise OSError("injected installation failure")
                return original_rename(src, dst)

            with (
                patch("learning.quarantine.check_append_metadata"),
                patch(
                    "learning.quarantine.os.rename", side_effect=fail_quarantine_install
                ),
            ):
                with self.assertRaisesRegex(OSError, "injected"):
                    commit(stage)
            self.assertEqual((source / "meta/info.json").read_text(), "original")
            self.assertEqual(
                (stage / source.name / "meta/info.json").read_text(), "retained"
            )
            self.assertTrue((stage / quarantine.name).exists())
            self.assertFalse(backup.exists())
            self.assertFalse(quarantine.exists())

    def test_source_change_refuses_commit_before_any_rename(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, quarantine, backup, stage = self.setup_layout(Path(tmp))
            (source / "meta/info.json").write_text("new recording added")
            with (
                patch("learning.quarantine.check_append_metadata"),
                patch("learning.quarantine.os.rename") as rename,
            ):
                with self.assertRaisesRegex(ValueError, "Source changed"):
                    commit(stage)
                rename.assert_not_called()
            self.assertFalse(backup.exists())
            self.assertFalse(quarantine.exists())


if __name__ == "__main__":
    unittest.main()
