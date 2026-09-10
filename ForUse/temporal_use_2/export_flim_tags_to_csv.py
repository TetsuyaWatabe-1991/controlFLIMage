"""
Read .flim tag metadata (no image pixels) and export selected tags to CSV.
"""
import csv
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_control_flimage_root = os.path.normpath(os.path.join(_script_dir, "..", ".."))
if _control_flimage_root not in sys.path:
    sys.path.insert(0, _control_flimage_root)

from FLIMageFileReader2 import FileReader

# --- settings ---
FLIM_PATH = (
    r"\\ry-lab-yas15\Users\Yasudalab\Documents\Tetsuya_Imaging"
    r"\20260827\psd_p38fretet_ca1_neuron1_dend3_017.flim"
)
# Add more keys as needed, e.g. "State.Acq.zoom", "State.Motor.motorPosition"
TAG_KEYS = ["State.Acq.power"]
PROCESS_ALL_FLIM_IN_FOLDER = True


def format_tag_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def list_flim_files(flim_path: str, process_all_in_folder: bool) -> list[str]:
    if process_all_in_folder:
        folder = flim_path if os.path.isdir(flim_path) else os.path.dirname(flim_path)
        return sorted(
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if name.lower().endswith(".flim")
        )
    return [flim_path]


def read_tags(file_path: str, tag_keys: list[str]) -> dict[str, str]:
    reader = FileReader()
    reader.read_imageFile(file_path, readImage=False)
    row = {"filename": os.path.basename(file_path)}
    for key in tag_keys:
        row[key] = format_tag_value(reader.statedict.get(key))
    return row


def main() -> None:
    files = list_flim_files(FLIM_PATH, PROCESS_ALL_FLIM_IN_FOLDER)
    if not files:
        raise FileNotFoundError(f"No .flim files found for: {FLIM_PATH}")

    rows = []
    for file_path in files:
        print(f"Reading tags: {os.path.basename(file_path)}")
        rows.append(read_tags(file_path, TAG_KEYS))

    out_dir = os.path.dirname(files[0])
    out_csv = os.path.join(out_dir, "flim_tags.csv")
    fieldnames = ["filename"] + TAG_KEYS
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} row(s): {out_csv}")


if __name__ == "__main__":
    main()
