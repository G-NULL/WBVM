import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from wbvm_rkhs_experiment import DATASETS, write_table1_artifacts


TABLE_FIELDS = [
    "dataset",
    "method",
    "tau_setting",
    "nfe",
    "w2",
    "sliced_w2",
    "off_manifold_rate",
    "off_threshold",
    "sample_seconds_10k",
    "sample_count",
]


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def default_tau(method: str, source: str) -> str:
    if method.endswith("WBVM-all"):
        return "[0.35,0.90]"
    if method.endswith("WBVM-single"):
        return "selected"
    if method == "Flow matching":
        return "[0.05,0.95]"
    return source


def normalize_table_row(row: Dict[str, str], source: str) -> Dict[str, object]:
    method = row["method"]
    if source == "rkhs":
        if method == "WBVM-all":
            method = "RKHS-WBVM-all"
        elif method == "WBVM-single":
            method = "RKHS-WBVM-single"
    elif source == "vector":
        if method == "WBVM-all":
            method = "vMMD-WBVM-all"
        elif method == "WBVM-single":
            method = "vMMD-WBVM-single"
        elif method.startswith("vMMD-"):
            method = method
    tau = row.get("tau_setting") or default_tau(method, source)
    return {
        "dataset": row["dataset"],
        "method": method,
        "tau_setting": tau,
        "nfe": int(float(row["nfe"])),
        "w2": float(row["w2"]),
        "sliced_w2": float(row["sliced_w2"]),
        "off_manifold_rate": float(row["off_manifold_rate"]),
        "off_threshold": float(row.get("off_threshold") or 0.1),
        "sample_seconds_10k": float(row["sample_seconds_10k"]),
        "sample_count": int(float(row["sample_count"])),
    }


def merge_table_rows(rkhs_rows: Iterable[Dict[str, str]], vector_rows: Iterable[Dict[str, str]]) -> List[Dict[str, object]]:
    rows = [normalize_table_row(row, "rkhs") for row in rkhs_rows]
    has_flow = any(row["method"] == "Flow matching" for row in rows)
    for row in vector_rows:
        normalized = normalize_table_row(row, "vector")
        if normalized["method"] == "Flow matching" and has_flow:
            continue
        rows.append(normalized)

    method_order = {
        "RKHS-WBVM-all": 0,
        "RKHS-WBVM-single": 1,
        "vMMD-WBVM-all": 2,
        "vMMD-WBVM-single": 3,
        "Flow matching": 4,
    }
    dataset_order = {name: i for i, name in enumerate(DATASETS)}
    rows.sort(key=lambda r: (dataset_order.get(str(r["dataset"]), 999), method_order.get(str(r["method"]), 999)))
    return rows


def write_metrics_summary(outdir: Path, rows: List[Dict[str, object]]) -> None:
    path = outdir / "metrics_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def copy_prefixed_artifacts(source_dir: Path, outdir: Path, prefix: str) -> None:
    for name in ["run_config.json", "metrics_summary.csv", "table1_metrics.csv", "table1_metrics.md", "table1_metrics.tex"]:
        path = source_dir / name
        if path.exists():
            shutil.copy2(path, outdir / f"{prefix}_{name}")


def merge_sample_panels(rkhs_dir: Path, vector_dir: Path, outpath: Path) -> bool:
    rkhs_image = rkhs_dir / "rkhs_wbvm_toy_6x4.png"
    vector_image = vector_dir / "vector_mmd_wbvm_toy_6x4.png"
    if not rkhs_image.exists() or not vector_image.exists():
        return False
    images = [
        ("Route 1: derivative-kernel RKHS-WBVM", mpimg.imread(rkhs_image)),
        ("Route 2: vector-flux MMD-WBVM", mpimg.imread(vector_image)),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(14, 16))
    for ax, (title, image) in zip(axes, images):
        ax.imshow(image)
        ax.set_title(title, fontsize=14, weight="bold", pad=10)
        ax.axis("off")
    fig.suptitle("WBVM route comparison on Section 6.1 synthetic manifolds", fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return True


def merge_results(rkhs_dir: Path, vector_dir: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    rkhs_rows = read_csv_rows(rkhs_dir / "table1_metrics.csv")
    vector_rows = read_csv_rows(vector_dir / "table1_metrics.csv")
    merged_rows = merge_table_rows(rkhs_rows, vector_rows)
    write_table1_artifacts(outdir, merged_rows)
    write_metrics_summary(outdir, merged_rows)
    copy_prefixed_artifacts(rkhs_dir, outdir, "rkhs")
    copy_prefixed_artifacts(vector_dir, outdir, "vector_mmd")
    merge_sample_panels(rkhs_dir, vector_dir, outdir / "merged_route_sample_panels.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge derivative-kernel RKHS-WBVM and vector-flux MMD-WBVM outputs")
    parser.add_argument("--rkhs-dir", required=True)
    parser.add_argument("--vector-dir", required=True)
    parser.add_argument("--outdir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    merge_results(Path(args.rkhs_dir), Path(args.vector_dir), Path(args.outdir))
    print(f"Merged outputs written to {Path(args.outdir).resolve()}")


if __name__ == "__main__":
    main()
