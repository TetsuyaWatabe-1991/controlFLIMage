# RESPAN setup (lab)

Official lahammond/RESPAN is **not pip-installable**. Use conda environments plus an external git clone.

## 1. Clone upstream RESPAN

```powershell
cd C:\Users\yasudalab\Documents\Tetsuya_GIT\third_party
git clone https://github.com/lahammond/RESPAN.git RESPAN
```

Or set `RESPAN_ROOT` to an existing clone.

## 2. Conda environments

Follow the [official RESPAN README](https://github.com/lahammond/RESPAN/blob/main/README.md):

- `respan_gpu` — TensorFlow/CuPy GUI and batch driver
- `respan_nnunet` — nnUNet v2 inference subprocess

Install nnUNet inside `respan_nnunet` per upstream instructions (do not commit nnUNet into this repo).

## 3. Pretrained model

Download Model 3 (2P in vivo) via the Google Form linked from the RESPAN README, then:

```powershell
setx RESPAN_MODEL_DIR "D:\path\to\extracted\model_folder"
```

## 4. Environment variables

| Variable | Purpose |
|----------|---------|
| `RESPAN_ROOT` | Path to lahammond/RESPAN clone root |
| `RESPAN_MODEL_DIR` | Pretrained nnU-Net model folder |
| `RESPAN_GPU_PYTHON` | Optional override for respan_gpu python.exe |
| `RESPAN_NNUNET_PYTHON` | Optional override for respan_nnunet python.exe |

Default clone search order when `RESPAN_ROOT` is unset:

1. `Tetsuya_GIT/third_party/RESPAN`
2. `Tetsuya_GIT/ongoing/RESPAN` (legacy)

## 5. Verify

```powershell
cd controlFLIMage\developing\mushroom_detector
python -c "from respan_runner.verify_respan_env import main; raise SystemExit(main())"
```

## 6. Lab wrappers (tracked in Git)

| Script | Purpose |
|--------|---------|
| `run_from_flim.py` | Export FLIM to ZYX TIFF and run RESPAN |
| `run_batch_stacks.py` | Batch TIFF stacks |
| `export_z_overlays.py` | Z-slice overlay PNGs |
| `launch_respan_gui.ps1` | Start upstream GUI |

Integration with spine workflows lives in `respan_mushroom_core.py` and `ongoing/ASIcontroller/respan_*.py`.
