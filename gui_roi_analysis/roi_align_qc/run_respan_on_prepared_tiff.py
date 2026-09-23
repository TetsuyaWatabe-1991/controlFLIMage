"""Run RESPAN on one prepared ZYX TIFF.

Invoke this with the respan_gpu Python. The TIFF and its JSON sidecar must
already exist. Outputs go under --run-parent and do not touch imaging folders.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_MUSHROOM = Path(__file__).resolve().parents[2] / "developing" / "mushroom_detector"
if str(_MUSHROOM) not in sys.path:
    sys.path.insert(0, str(_MUSHROOM))

from respan_runner.run_batch_stacks import run_one_stack  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RESPAN on one prepared TIFF.")
    parser.add_argument("--tiff", type=Path, required=True)
    parser.add_argument("--run-parent", type=Path, required=True)
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    name, ok, message = run_one_stack(
        tiff_path=args.tiff,
        run_parent=args.run_parent,
        rerun=args.rerun,
        skip_analysis=False,
        skip_overlays=True,
    )
    print(f"{name} ok={ok} {message}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
