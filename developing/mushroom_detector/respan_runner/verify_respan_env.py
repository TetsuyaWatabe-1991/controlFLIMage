"""Quick health check for RESPAN conda environments on this machine."""

from __future__ import annotations

import subprocess
import sys

from respan_runner.paths import (
    respan_gpu_python,
    respan_nnunet_python,
    respan_repo_root,
    respan_root,
)


def _run(python_exe, code: str) -> tuple[bool, str]:
    proc = subprocess.run(
        [str(python_exe), "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, output


def main() -> int:
    ok = True
    checks: list[tuple[str, bool, str]] = []

    gpu_py = respan_gpu_python()
    nn_py = respan_nnunet_python()
    gui_script = respan_repo_root() / "Scripts" / "RESPAN_GUI_DIST.py"

    if not gpu_py.is_file():
        print(f"MISSING: {gpu_py}")
        return 1
    if not nn_py.is_file():
        print(f"MISSING: {nn_py}")
        return 1
    if not gui_script.is_file():
        print(f"MISSING: {gui_script}")
        return 1

    gpu_code = (
        "import cupy, tensorflow, csbdeep, skimage, PyQt5; "
        "import cupy as cp; "
        "print('cupy', cupy.__version__, 'devices', cp.cuda.runtime.getDeviceCount())"
    )
    nn_code = (
        "import torch, nnunetv2; "
        "print('torch', torch.__version__); "
        "print('cuda', torch.cuda.is_available()); "
        "print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
    )

    for label, py, code in (
        ("respan_gpu", gpu_py, gpu_code),
        ("respan_nnunet", nn_py, nn_code),
    ):
        passed, output = _run(py, code)
        checks.append((label, passed, output))
        ok = ok and passed

    print("RESPAN environment verification")
    print(f"  clone root: {respan_root()}")
    print(f"  gui script: {gui_script}")
    for label, passed, output in checks:
        status = "OK" if passed else "FAIL"
        print(f"\n[{status}] {label}")
        print(output)

    if ok:
        print("\nAll checks passed.")
        print("Launch GUI with:")
        print(f"  {gpu_py} {gui_script}")
    else:
        print("\nSome checks failed. See messages above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
