import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def collect_rows(baseline_dir: Path, route2_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in read_rows(baseline_dir / "metrics_summary.csv"):
        rows.append({"Method": row.get("method", ""), "MNIST-FID": as_float(row.get("mnist_fid", ""))})
    for row in read_rows(route2_dir / "metrics_summary.csv"):
        rows.append({"Method": row.get("method", "vMMD-WBVM"), "MNIST-FID": as_float(row.get("mnist_fid", ""))})
    rows = [row for row in rows if math.isfinite(float(row["MNIST-FID"]))]
    rows.sort(key=lambda row: float(row["MNIST-FID"]))
    return rows


def write_csv(rows: List[Dict[str, object]], outpath: Path) -> None:
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Method", "MNIST-FID"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"Method": row["Method"], "MNIST-FID": f"{float(row['MNIST-FID']):.6f}"})


def write_markdown(rows: List[Dict[str, object]], outpath: Path) -> None:
    lines = ["| Method | MNIST-FID |", "|---|---:|"]
    for row in rows:
        lines.append(f"| {row['Method']} | {float(row['MNIST-FID']):.3f} |")
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_png(rows: List[Dict[str, object]], outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 0.48 * len(rows) + 1.1))
    ax.axis("off")
    cell_text = [[row["Method"], f"{float(row['MNIST-FID']):.3f}"] for row in rows]
    table = ax.table(
        cellText=cell_text,
        colLabels=["Method", "MNIST-FID"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.28)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f3a5f")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f1f5f9")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write final MNIST-FID-only table")
    parser.add_argument("--baseline-dir", default="outputs_mnist_corrected_cnn_mnistfid")
    parser.add_argument("--route2-dir", default="outputs_mnist_route2_cnn_pixel")
    parser.add_argument("--outdir", default="outputs_mnist_cnn_mnistfid_only")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(root / args.baseline_dir, root / args.route2_dir)
    write_csv(rows, outdir / "metrics_summary.csv")
    write_markdown(rows, outdir / "metrics_summary.md")
    write_png(rows, outdir / "mnist_fid_table.png")


if __name__ == "__main__":
    main()
