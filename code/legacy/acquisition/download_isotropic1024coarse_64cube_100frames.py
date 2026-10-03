from pathlib import Path
import os
import json
import time
import traceback
import numpy as np
import pandas as pd

from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getCutout

ROOT = Path("legacy_workspace")

DATASET = "isotropic1024coarse"
VAR = "velocity"

OUT = ROOT / "datasets" / DATASET / "velocity_64cube_100frames"
OUT_FRAMES = OUT / "frames_npz"
OUT_META = OUT / "metadata"
OUT_FRAMES.mkdir(parents=True, exist_ok=True)
OUT_META.mkdir(parents=True, exist_ok=True)

MANIFEST_CSV = OUT_META / "manifest_velocity_64cube_100frames.csv"
SUMMARY_JSON = OUT_META / "summary_velocity_64cube_100frames.json"

TOKEN = os.environ.get("JHTDB_TOKEN", "").strip()
if not TOKEN:
    raise RuntimeError("JHTDB_TOKEN is not set in the environment.")

print("===== DOWNLOAD: isotropic1024coarse 64^3 x 100 frames =====")
print("dataset =", DATASET)
print("variable =", VAR)
print("output =", OUT)
print("token = [hidden]")

X_RANGE = [1, 64]
Y_RANGE = [1, 64]
Z_RANGE = [1, 64]
T_START = 1
T_END = 100

EXPECTED_SHAPE = (64, 64, 64, 3)
DX = float(2.0 * np.pi / 1024.0)

metadata = {
    "dataset": DATASET,
    "variable": VAR,
    "x_range": X_RANGE,
    "y_range": Y_RANGE,
    "z_range": Z_RANGE,
    "t_start": T_START,
    "t_end": T_END,
    "n_frames": T_END - T_START + 1,
    "expected_shape_per_frame": list(EXPECTED_SHAPE),
    "grid_points_per_direction": 1024,
    "domain_length": "2*pi",
    "dx": DX,
    "note": "First useful JHTDB velocity subset for coarse-grained Pi_l pre-test."
}

with open(OUT_META / "download_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print("\n===== CONSTRUCT DATASET OBJECT =====")
cube = turb_dataset(
    dataset_title=DATASET,
    output_path=str(OUT),
    auth_token=TOKEN
)
print("dataset object constructed")

xyzt_strides = np.array([1, 1, 1, 1], dtype=np.int32)

def save_manifest_update(row):
    new_row = pd.DataFrame([row])
    if MANIFEST_CSV.exists():
        try:
            old = pd.read_csv(MANIFEST_CSV)
            combined = pd.concat([old, new_row], ignore_index=True)
            combined = combined.drop_duplicates(subset=["frame_index"], keep="last")
            combined = combined.sort_values("frame_index")
        except Exception:
            combined = new_row
    else:
        combined = new_row
    combined.to_csv(MANIFEST_CSV, index=False)

def validate_npz(path):
    if not path.exists():
        return False, "missing"
    try:
        z = np.load(path)
        if "velocity" not in z.files:
            return False, "missing velocity"
        arr = z["velocity"]
        if tuple(arr.shape) != EXPECTED_SHAPE:
            return False, f"bad shape {arr.shape}"
        if arr.dtype != np.float32:
            return False, f"bad dtype {arr.dtype}"
        if int(np.isnan(arr).sum()) != 0:
            return False, "contains NaN"
        if int(np.isinf(arr).sum()) != 0:
            return False, "contains Inf"
        return True, "ok"
    except Exception as e:
        return False, repr(e)

def download_one_frame(t):
    xyzt_axes_ranges_original = np.array([
        X_RANGE,
        Y_RANGE,
        Z_RANGE,
        [t, t],
    ], dtype=np.int32)

    result = getCutout(
        cube,
        VAR,
        xyzt_axes_ranges_original,
        xyzt_strides,
        trace_memory=False,
        verbose=False
    )

    if not hasattr(result, "data_vars"):
        raise RuntimeError("getCutout returned object without data_vars")

    data_vars = list(result.data_vars)
    if len(data_vars) < 1:
        raise RuntimeError("getCutout returned no data variables")

    varname = data_vars[0]
    arr = np.asarray(result[varname].values)

    if tuple(arr.shape) != EXPECTED_SHAPE:
        raise RuntimeError(f"unexpected shape: got {arr.shape}, expected {EXPECTED_SHAPE}")

    arr = arr.astype(np.float32, copy=False)

    nan_count = int(np.isnan(arr).sum())
    inf_count = int(np.isinf(arr).sum())

    if nan_count != 0 or inf_count != 0:
        raise RuntimeError(f"NaN/Inf detected: nan={nan_count}, inf={inf_count}")

    return arr, varname, nan_count, inf_count

t0_all = time.time()

for t in range(T_START, T_END + 1):
    frame_path = OUT_FRAMES / f"velocity_t{t:05d}.npz"

    valid, reason = validate_npz(frame_path)
    if valid:
        print(f"[SKIP VALID] frame={t}")
        save_manifest_update({
            "frame_index": t,
            "status": "ok",
            "file": str(frame_path),
            "varname": "velocity",
            "shape": str(EXPECTED_SHAPE),
            "dtype": "float32",
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "nan_count": 0,
            "inf_count": 0,
            "elapsed_sec": 0.0,
            "bytes": frame_path.stat().st_size,
            "download_mode": "existing_valid",
        })
        continue

    print("\n" + "=" * 80)
    print(f"[DOWNLOAD] frame={t}/{T_END}")

    delays = [10, 20, 40, 80, 160]
    last_error = None

    for attempt in range(1, 6):
        frame_t0 = time.time()
        try:
            print(f"attempt {attempt}/5")
            arr, varname, nan_count, inf_count = download_one_frame(t)

            np.savez_compressed(
                frame_path,
                velocity=arr,
                frame_index=np.int32(t),
                dataset=np.array(DATASET),
                variable=np.array(VAR),
                x_range=np.array(X_RANGE, dtype=np.int32),
                y_range=np.array(Y_RANGE, dtype=np.int32),
                z_range=np.array(Z_RANGE, dtype=np.int32),
                t_range=np.array([t, t], dtype=np.int32),
                dx=np.float64(DX),
            )

            row = {
                "frame_index": t,
                "status": "ok",
                "file": str(frame_path),
                "varname": varname,
                "shape": str(tuple(arr.shape)),
                "dtype": str(arr.dtype),
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "mean": float(np.nanmean(arr)),
                "std": float(np.nanstd(arr)),
                "nan_count": nan_count,
                "inf_count": inf_count,
                "elapsed_sec": float(time.time() - frame_t0),
                "bytes": frame_path.stat().st_size,
                "download_mode": "direct_64cube",
            }

            save_manifest_update(row)

            print("[OK] frame =", t)
            print("shape =", row["shape"])
            print("min/max/mean/std =", row["min"], row["max"], row["mean"], row["std"])
            print("nan/inf =", nan_count, inf_count)
            last_error = None
            break

        except Exception as e:
            last_error = e
            print(f"[FAILED] frame={t}, attempt={attempt}")
            traceback.print_exc()

            if attempt < 5:
                delay = delays[attempt - 1]
                print(f"sleep {delay} seconds before retry...")
                time.sleep(delay)

    if last_error is not None:
        fail_row = {
            "frame_index": t,
            "status": "failed",
            "file": str(frame_path),
            "error": repr(last_error),
            "download_mode": "direct_64cube",
        }
        with open(OUT_META / f"failed_frame_{t:05d}.json", "w", encoding="utf-8") as f:
            json.dump(fail_row, f, indent=2, ensure_ascii=False)
        save_manifest_update(fail_row)
        raise RuntimeError(f"frame {t} failed after 5 retries: {repr(last_error)}")

    time.sleep(1)

elapsed_all = time.time() - t0_all

print("\n===== FINAL CHECK =====")
df = pd.read_csv(MANIFEST_CSV)
ok = df[df["status"] == "ok"].copy()
ok_frames = set(ok["frame_index"].astype(int).tolist())
missing = sorted(set(range(T_START, T_END + 1)) - ok_frames)

print("manifest =", MANIFEST_CSV)
print("rows =", len(df))
print("ok frames =", len(ok_frames))
print("expected frames =", T_END - T_START + 1)
print("missing frames =", missing)

if len(ok_frames) != (T_END - T_START + 1):
    raise RuntimeError(f"incomplete download: ok={len(ok_frames)}, missing={missing}")

total_bytes = 0
for t in range(T_START, T_END + 1):
    p = OUT_FRAMES / f"velocity_t{t:05d}.npz"
    total_bytes += p.stat().st_size

summary = {
    "dataset": DATASET,
    "variable": VAR,
    "output": str(OUT),
    "frames_expected": T_END - T_START + 1,
    "frames_ok": int(len(ok_frames)),
    "missing_frames": missing,
    "expected_shape_per_frame": list(EXPECTED_SHAPE),
    "total_npz_bytes": int(total_bytes),
    "elapsed_total_sec": float(elapsed_all),
    "manifest_csv": str(MANIFEST_CSV),
    "dx": DX,
    "success": True,
}

with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n===== SUMMARY =====")
print(json.dumps(summary, indent=2, ensure_ascii=False))

print("\n===== SUCCESS: isotropic1024coarse 64^3 x 100 frames download complete =====")
