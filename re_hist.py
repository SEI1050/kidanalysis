from pathlib import Path
import subprocess
import sys
import re
import csv
from datetime import datetime

# ======================
# ここだけ基本的に変える
# ======================
DATA_DATE = "20260527"
rebin = 20

# すでにfit済みならskipする
skip_existing = True

# 実行せずに対象ファイルだけ確認したいとき True
dry_run = False

# ======================
# ローカル側
# re_hist.py と Rez_waveform_fit.py があるフォルダ
# ======================
HERE = Path(__file__).resolve().parent
script = HERE / "Rez_waveform_fit.py"

# 出力先:
# KIDANALYSIS/data/20260527/
out_dir = HERE / "data" / DATA_DATE
out_dir.mkdir(parents=True, exist_ok=True)

# ======================
# OneDrive側のデータフォルダ候補
# ======================
onedrive_candidates = [
    Path.home() / "OneDrive - The University of Tokyo" / "東京大学" / "4S" / "kidfit",
    Path.home() / "Library" / "CloudStorage" / "OneDrive-TheUniversityofTokyo" / "東京大学" / "4S" / "kidfit",
]

base_dir = None
for p in onedrive_candidates:
    candidate = p / DATA_DATE
    if candidate.is_dir():
        base_dir = candidate
        break

if base_dir is None:
    print("ERROR: データフォルダが見つかりません。候補は以下です。")
    for p in onedrive_candidates:
        print("  ", p / DATA_DATE)
    sys.exit(1)

if not script.is_file():
    print("ERROR: Rez_waveform_fit.py が見つかりません。")
    print("script =", script)
    sys.exit(1)

# ======================
# 測定フォルダ名のパターン
# 例:
# 5.451GHz_z=7.5mm_x=5.4mm
# 5.451GHz_z=7.5mm_x=3.4mm_second
# ======================
meas_dir_pattern = re.compile(
    r"^(?P<freq>\d+(?:\.\d+)?)GHz_"
    r"z=(?P<z>-?\d+(?:\.\d+)?)mm_"
    r"x=(?P<x>-?\d+(?:\.\d+)?)mm"
    r"(?:_(?P<tag>.+))?$"
)

def parse_meas_dir(path: Path):
    m = meas_dir_pattern.match(path.name)
    if m is None:
        return None

    d = m.groupdict()
    return {
        "freq": float(d["freq"]),
        "z": float(d["z"]),
        "x": float(d["x"]),
        "tag": d["tag"] or "",
    }

def safe_name(s: str):
    return re.sub(r"[^\w.\-=]+", "_", s)

def sort_key(npz_path: Path):
    info = parse_meas_dir(npz_path.parent)
    if info is None:
        return (999, 999, 999, npz_path.as_posix())

    return (
        info["freq"],
        info["z"],
        info["x"],
        info["tag"],
        npz_path.name,
    )

def expected_dst_csv(npz_path: Path, rebin: int):
    meas_name = safe_name(npz_path.parent.name)
    base_name = f"{meas_name}__{npz_path.stem}"
    return out_dir / f"{base_name}_fitres_rebin{rebin}.csv"

def expected_dst_pdf(npz_path: Path, rebin: int):
    meas_name = safe_name(npz_path.parent.name)
    base_name = f"{meas_name}__{npz_path.stem}"
    return out_dir / f"{base_name}_fit_rebin{rebin}.pdf"

# ======================
# npz探索
# ======================
npz_files = []

for d in sorted(base_dir.iterdir()):
    if not d.is_dir():
        continue

    info = parse_meas_dir(d)
    if info is None:
        continue

    # 測定フォルダ内の wf_*.npz を拾う
    npz_files.extend(sorted(d.rglob("wf_*.npz")))

npz_files = sorted(npz_files, key=sort_key)

print("script  =", script)
print("base_dir =", base_dir)
print("out_dir  =", out_dir)
print("found npz =", len(npz_files))

# ======================
# 実行ログ
# ======================
log_path = out_dir / f"wavefit_batch_log_rebin{rebin}.csv"

with open(log_path, "a", newline="", encoding="utf-8") as logf:
    writer = csv.writer(logf)

    if log_path.stat().st_size == 0:
        writer.writerow([
            "time",
            "status",
            "npz",
            "dst_csv",
            "dst_pdf",
            "returncode",
        ])

    for f in npz_files:
        dst_csv = expected_dst_csv(f, rebin)
        dst_pdf = expected_dst_pdf(f, rebin)

        if skip_existing and dst_csv.exists() and dst_pdf.exists():
            print("skip", f)
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                "skip",
                f.as_posix(),
                dst_csv.as_posix(),
                dst_pdf.as_posix(),
                "",
            ])
            continue

        print("fit ", f)

        cmd = [
            sys.executable,
            str(script),
            str(f),
            str(rebin),
            str(out_dir),
        ]

        if dry_run:
            print("  dry_run:", " ".join(cmd))
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                "dry_run",
                f.as_posix(),
                dst_csv.as_posix(),
                dst_pdf.as_posix(),
                "",
            ])
            continue

        result = subprocess.run(cmd, cwd=HERE)

        if result.returncode == 0:
            if dst_csv.exists() and dst_pdf.exists():
                status = "done"
                print("done", f)
                print("  csv:", dst_csv)
                print("  pdf:", dst_pdf)
            else:
                status = "done_but_outputs_not_found"
                print("WARNING: fit finished but output file not found")
                print("  expected csv:", dst_csv)
                print("  expected pdf:", dst_pdf)
        else:
            status = "error"
            print("ERROR", f, "returncode =", result.returncode)

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            status,
            f.as_posix(),
            dst_csv.as_posix(),
            dst_pdf.as_posix(),
            result.returncode,
        ])

print("log saved:", log_path)
print("outputs saved in:", out_dir)