import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mnist_fid_experiment import save_table, write_csv


def read_rows(path: Path) -> List[Dict[str, object]]:
    with open(path, newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def normalize_route2_row(row: Dict[str, object]) -> Dict[str, object]:
    out = dict(row)
    out["method"] = "vMMD-WBVM"
    out["fid"] = "nan"
    out["inception_fid"] = "nan"
    out.setdefault("model_space", "pixel")
    out.setdefault("nfe", "1")
    return out


def metric_key(row: Dict[str, object]) -> float:
    value = as_float(row.get("mnist_fid", ""))
    if math.isfinite(value):
        return value
    return float("inf")


def fmt(value: object) -> str:
    x = as_float(value)
    if not math.isfinite(x):
        return ""
    return f"{x:.3f}"


def write_markdown(rows: List[Dict[str, object]], outpath: Path) -> None:
    lines = [
        "| Method | Inception-FID | MNIST-FID | NFE | Notes |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {ifid} | {mfid} | {nfe} | {note} |".format(
                method=row.get("method", ""),
                ifid=fmt(row.get("inception_fid", row.get("fid", ""))),
                mfid=fmt(row.get("mnist_fid", "")),
                nfe=row.get("nfe", ""),
                note=str(row.get("note", "")).replace("|", "/"),
            )
        )
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_mnist_only(rows: List[Dict[str, object]], outdir: Path) -> None:
    clean = [
        {"Method": row.get("method", ""), "MNIST-FID": as_float(row.get("mnist_fid", ""))}
        for row in rows
        if math.isfinite(as_float(row.get("mnist_fid", "")))
    ]
    clean.sort(key=lambda row: float(row["MNIST-FID"]))
    with open(outdir / "mnist_fid_only.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Method", "MNIST-FID"])
        writer.writeheader()
        for row in clean:
            writer.writerow({"Method": row["Method"], "MNIST-FID": f"{float(row['MNIST-FID']):.6f}"})
    md = ["| Method | MNIST-FID |", "|---|---:|"]
    for row in clean:
        md.append(f"| {row['Method']} | {float(row['MNIST-FID']):.3f} |")
    (outdir / "mnist_fid_only.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(5.2, 0.48 * len(clean) + 1.1))
    ax.axis("off")
    table = ax.table(
        cellText=[[row["Method"], f"{float(row['MNIST-FID']):.3f}"] for row in clean],
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
    fig.savefig(outdir / "mnist_fid_only.png", dpi=220)
    plt.close(fig)


def merge(base_dir: Path, route2_dir: Path, outdir: Path) -> None:
    rows = read_rows(base_dir / "metrics_summary.csv")
    rows.extend(normalize_route2_row(row) for row in read_rows(route2_dir / "metrics_summary.csv"))
    rows.sort(key=metric_key)
    outdir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, outdir / "metrics_summary.csv")
    write_markdown(rows, outdir / "metrics_summary.md")
    save_table(rows, outdir / "mnist_fid_table.png")
    write_mnist_only(rows, outdir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge v3 pixel-DIT route-two result into the original pixel table")
    parser.add_argument("--base-dir", default="outputs_mnist_v3_standard_pixel")
    parser.add_argument("--route2-dir", default="outputs_mnist_route2_v3_pixel_dit")
    parser.add_argument("--outdir", default="outputs_mnist_v3_standard_pixel_with_route2")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    merge(root / args.base_dir, root / args.route2_dir, root / args.outdir)


if __name__ == "__main__":
    main()
