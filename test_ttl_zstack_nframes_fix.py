# -*- coding: utf-8 -*-
"""
Manual/hardware verification for the ASI TTL Z-stack nFrames header bug fix.

Loads the exact setting file the user reported the bug with
(z15_10_ttl_Z.txt: State.Acq.nFrames = 1, State.Acq.nSlices = 15,
State.Acq.asiTtlZStack = True), confirms nFrames reads back as 1
immediately after LoadSetting, runs a single grab (nImages temporarily
forced to 1 for a fast test), then re-opens the saved .flim file and
checks that its own header still reports State.Acq.nFrames == 1 (not the
ASI-continuous-Z inflated hardware frame count).

Requires FLIMage2 to be running and connected (live hardware).
"""
import os
import sys
import glob

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from controlflimage_threading import Control_flimage
from FLIMageFileReader2 import FileReader

SETTING_PATH = r"C:\Users\yasudalab\Documents\FLIMage\Init_Files\z15_10_ttl_Z.txt"
TEST_BASENAME = "nframes_fix_test_"


def main():
    FLIMageCont = Control_flimage()

    print("\n=== Loading setting:", SETTING_PATH, "===")
    FLIMageCont.flim.sendCommand(f"LoadSetting, {SETTING_PATH}")

    n_frames_after_load = FLIMageCont.get_val_sendCommand("State.Acq.nFrames")
    n_slices_after_load = FLIMageCont.get_val_sendCommand("State.Acq.nSlices")
    asi_ttl = FLIMageCont.get_val_sendCommand("State.Acq.asiTtlZStack")
    print(f"After LoadSetting: nFrames={n_frames_after_load}, "
          f"nSlices={n_slices_after_load}, asiTtlZStack={asi_ttl}")

    assert int(float(n_frames_after_load)) == 1, (
        f"Expected nFrames=1 right after LoadSetting, got {n_frames_after_load}"
    )

    # Use a dedicated basename/single-image count so this test is fast and
    # never collides with the user's real acquisition files.
    folder = FLIMageCont.get_val_sendCommand("State.Files.pathName")
    FLIMageCont.flim.sendCommand("BeginStateBatch")
    FLIMageCont.flim.sendCommand(f'State.Files.baseName = "{TEST_BASENAME}"')
    FLIMageCont.flim.sendCommand("State.Acq.nImages = 1")
    FLIMageCont.flim.sendCommand("EndStateBatch")

    # Remove any leftover files from a previous run of this test so the
    # new-file detection below is unambiguous (FLIMage reuses the same
    # fileCounter/name when nothing else has incremented it).
    for stale in glob.glob(os.path.join(folder, f"{TEST_BASENAME}*.flim")):
        try:
            os.remove(stale)
        except OSError:
            pass
    existing = sorted(glob.glob(os.path.join(folder, f"{TEST_BASENAME}*.flim")))
    print(f"\n=== Starting grab (single ASI TTL Z-stack: 1 frame x 15 slices) ===")
    FLIMageCont.flim.sendCommand("StartGrab")
    success = FLIMageCont.wait_while_grabbing(sleep_every_sec=0.2, max_waiting_sec=60)
    print("Grab finished, success =", success)

    after = sorted(glob.glob(os.path.join(folder, f"{TEST_BASENAME}*.flim")))
    new_files = [f for f in after if f not in existing]
    assert new_files, f"No new file appeared in {folder} matching {TEST_BASENAME}*.flim"
    saved_path = new_files[-1]
    print("Saved file:", saved_path)

    reader = FileReader()
    reader.read_imageFile(saved_path, True)
    saved_nframes = reader.statedict["State.Acq.nFrames"]
    saved_nslices = reader.statedict["State.Acq.nSlices"]
    print(f"\n=== Saved file header: nFrames={saved_nframes}, nSlices={saved_nslices} ===")

    if int(float(saved_nframes)) == 1:
        print("\nPASS: saved .flim header correctly reports nFrames=1.")
    else:
        print(
            "\nFAIL: saved .flim header reports nFrames="
            f"{saved_nframes} (expected 1). ASI TTL Z-stack nFrames "
            "inflation bug is still present."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
