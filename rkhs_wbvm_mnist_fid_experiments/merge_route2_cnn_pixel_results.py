import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

from mnist_fid_experiment import save_table, write_csv


def read_rows(path: Path) -> List[Dict[str, object]]:
    with open(path, newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def normalize_base_row(row: Dict[str, object]) -> Dict[str, object]:
    out = dict(row)
    out.setdefault("inception_fid", out.get("fid", ""))
    out.setdefault("mnist_fid", "nan")
    out.setdefault("mnist_kid_x1000", "nan")
    out.setdefault("mnist_kid_raw_x1000", "nan")
    out.setdefault("loss_statistic", "")
    out.setdefault("model_space", "pixel")
    return out


def normalize_route2_row(row: Dict[str, object]) -> Dict[str, object]:
    out = dict(row)
    out["method"] = "vMMD-WBVM"
    out["fid"] = out.get("mnist_fid", out.get("fid", ""))
    out["inception_fid"] = "nan"
    out.setdefault("selected_tau", "")
    out.setdefault("shortcut_ema_decay", "")
    out.setdefault("validation_metric", "mnist_fid")
    out.setdefault("validation_score", "")
    return out


def format_float(value: object) -> str:
    x = as_float(value)
    if not math.isfinite(x):
        return ""
    return f"{x:.3f}"


def write_markdown(rows: List[Dict[str, object]], outpath: Path) -> None:
    lines = [
        "| Method | FID | MNIST-FID | NFE | Notes |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {fid} | {mnist_fid} | {nfe} | {note} |".format(
                method=row.get("method", ""),
                fid=format_float(row.get("fid", "")),
                mnist_fid=format_float(row.get("mnist_fid", "")),
                nfe=row.get("nfe", ""),
                note=str(row.get("note", "")).replace("|", "/"),
            )
        )
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge(base_dir: Path, route2_dir: Path, outdir: Path) -> None:
    rows = [normalize_base_row(row) for row in read_rows(base_dir / "metrics_summary.csv")]
    rows.extend(normalize_route2_row(row) for row in read_rows(route2_dir / "metrics_summary.csv"))
    rows.sort(key=lambda row: as_float(row.get("fid", "")))
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, outdir / "metrics_summary.csv")
    write_markdown(rows, outdir / "metrics_summary.md")
    save_table(rows, outdir / "mnist_fid_table.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge pixel-CNN route-two MNIST result into corrected combined outputs")
    parser.add_argument("--base-dir", default="outputs_mnist_corrected_standard_combined")
    parser.add_argument("--route2-dir", default="outputs_mnist_route2_cnn_pixel")
    parser.add_argument("--outdir", default="outputs_mnist_corrected_standard_combined_with_route2")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    merge(root / args.base_dir, root / args.route2_dir, root / args.outdir)


if __name__ == "__main__":
    main()
