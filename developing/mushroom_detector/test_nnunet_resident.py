"""Resident nnU-Net keeps one process for more than one folder."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

_MUSHROOM = Path(__file__).resolve().parent
if str(_MUSHROOM) not in sys.path:
    sys.path.insert(0, str(_MUSHROOM))

from respan_runner.nnunet_resident import (  # noqa: E402
    ResidentNnUNet,
    _worker_command,
    predict_hook,
)


class _Logger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)


class _Module:
    def __init__(self) -> None:
        self.initialized = False

    def initialize_nnUnet(self, settings, logger) -> None:
        del settings, logger
        self.initialized = True

    def run_nnunet_predict(self, *args):
        del args
        raise AssertionError("one-shot predict should not run")


class ResidentProtocolTest(unittest.TestCase):
    def test_two_jobs_share_one_process(self) -> None:
        resident = ResidentNnUNet(_worker_command(fake=True))
        with tempfile.TemporaryDirectory() as tmp:
            first = os.path.join(tmp, "a")
            second = os.path.join(tmp, "b")
            try:
                resident.start()
                pid = resident.pid
                self.assertEqual(resident.predict(first, first), 0)
                self.assertEqual(resident.predict(second, second), 0)
                self.assertEqual(resident.pid, pid)
                self.assertEqual(resident.last_message.get("loads"), 1)
                self.assertEqual(Path(first, "resident_marker.txt").read_text(encoding="utf-8"), "1")
                self.assertEqual(Path(second, "resident_marker.txt").read_text(encoding="utf-8"), "1")
                self.assertIsNone(resident._proc.poll())
            finally:
                resident.close()
        self.assertIsNotNone(resident.pid)
        self.assertIsNone(resident._proc)

    def test_hook_sends_the_folder_to_the_resident(self) -> None:
        class _Resident:
            def __init__(self) -> None:
                self.calls: list[tuple[str, str]] = []

            def predict(self, input_dir: str, output_dir: str, fold: str = "all") -> int:
                self.calls.append((input_dir, output_dir, fold))
                return 0

        module = _Module()
        resident = _Resident()
        logger = _Logger()
        with predict_hook(module, resident):
            code = module.run_nnunet_predict(
                "nnUNetv2_predict.bat",
                r"C:\in",
                r"C:\out",
                "220",
                "3d_fullres",
                object(),
                logger,
            )
        self.assertEqual(code, 0)
        self.assertEqual(resident.calls, [(r"C:\in", r"C:\out", "all")])
        self.assertTrue(module.initialized)
        self.assertTrue(logger.messages)
        self.assertTrue(callable(module.run_nnunet_predict))
        with self.assertRaises(AssertionError):
            module.run_nnunet_predict()


if __name__ == "__main__":
    unittest.main()
