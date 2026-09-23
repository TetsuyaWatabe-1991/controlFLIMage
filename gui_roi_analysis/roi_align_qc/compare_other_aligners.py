"""Score optical-flow and feature-match aligners against human spine positions.

Usage:
    python compare_other_aligners.py --df-path G:/.../combined_df_respan.pkl
"""

from __future__ import annotations

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from compare_field_aligners import run_session  # noqa: E402
from other_align import OTHER_ALIGNERS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare non-correlation aligners")
    parser.add_argument("--df-path", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    run_session(args.df_path, args.out_dir, OTHER_ALIGNERS, file_stem="other_aligner")


if __name__ == "__main__":
    main()
