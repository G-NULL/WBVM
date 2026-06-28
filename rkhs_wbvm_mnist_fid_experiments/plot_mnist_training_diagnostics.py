import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


COLORS = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#0072B2", "#CC79A7", "#8C8C8C"]
METHOD_ORDER = ["Drifting", "WBVM-single", "MeanFlow", "WBVM-all", "vMMD-WBVM", "ShortcutFlow", "Real"]


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linestyle": "-",
        "lines.linewidth": 1.8,
    }
)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def read_json(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite_xy(rows: Iterable[Dict[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ys: List[float] = []
    for row in rows:
        x = as_float(row.get("step"))
        y = as_float(row.get("loss"))
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def smooth(y: np.ndarray) -> np.ndarray:
    if y.size < 5:
        return y
    window = max(3, min(101, int(y.size // 60) * 2 + 1))
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(y, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def selected_wbvm_tau(metrics_rows: List[Dict[str, str]]) -> Optional[float]:
    for row in metrics_rows:
        if row.get("method") == "WBVM-single":
            tau = as_float(row.get("selected_tau"))
            if math.isfinite(tau):
                return tau
    return None


def load_training_runs(outdir: Path, metrics_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    selected_tau = selected_wbvm_tau(metrics_rows)
    runs: List[Dict[str, object]] = []
    for run_dir in sorted((outdir / "training_logs").glob("*")):
        if not run_dir.is_dir():
            continue
        manifest = read_json(run_dir / "manifest.json")
        summary = read_json(run_dir / "summary.json")
        steps = read_csv(run_dir / "steps.csv")
        if not manifest or not steps:
            continue
        method = str(manifest.get("method", run_dir.name))
        extra = manifest.get("extra_hyperparameters", {})
        if not isinstance(extra, dict):
            extra = {}
        candidate_tau = as_float(extra.get("candidate_tau"))
        parent = extra.get("parent_method")
        plot_method = method
        selected = True
        if parent == "WBVM-single":
            selected = selected_tau is None or abs(candidate_tau - selected_tau) < 1e-9
            plot_method = "WBVM-single" if selected else f"WBVM-single tau={candidate_tau:g}"
        x, y = finite_xy(steps)
        if x.size == 0:
            continue
        runs.append(
            {
                "dir": run_dir,
                "method": method,
                "plot_method": plot_method,
                "candidate_tau": candidate_tau,
                "is_selected": selected,
                "parent_method": parent,
                "manifest": manifest,
                "summary": summary,
                "x": x,
                "y": y,
            }
        )
    return runs


def save_fig(fig: plt.Figure, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath.with_suffix(".pdf"))
    fig.savefig(outpath.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_loss_curves(runs: List[Dict[str, object]], outdir: Path) -> None:
    seen = set()
    selected_runs = []
    for method in METHOD_ORDER:
        for run in runs:
            if run["plot_method"] == method and run.get("is_selected", True):
                selected_runs.append(run)
                seen.add(method)
                break
    for run in runs:
        label = str(run["plot_method"])
        if label not in seen and run.get("is_selected", True) and label != "Real":
            selected_runs.append(run)
            seen.add(label)
    if not selected_runs:
        return

    fig, ax = plt.subplots(figsize=(6.75, 3.1))
    for idx, run in enumerate(selected_runs):
        x = run["x"]
        y = run["y"]
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
        ax.plot(x, smooth(y), label=str(run["plot_method"]), color=COLORS[idx % len(COLORS)])
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_title("Training Loss Curves")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    save_fig(fig, outdir / "loss_curves_all_methods")

    fig, ax = plt.subplots(figsize=(6.75, 3.1))
    for idx, run in enumerate(selected_runs):
        x = run["x"]
        y = np.abs(run["y"])
        assert isinstance(x, np.ndarray)
        base = np.nanmedian(y[: max(5, min(25, y.size))])
        if not math.isfinite(float(base)) or base <= 0:
            base = max(float(np.nanmax(y)), 1.0)
        ax.plot(x, smooth(y / base), label=str(run["plot_method"]), color=COLORS[idx % len(COLORS)])
    ax.set_xlabel("Training step")
    ax.set_ylabel("|loss| / initial median")
    ax.set_yscale("log")
    ax.set_title("Normalized Loss Magnitudes")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    save_fig(fig, outdir / "loss_curves_normalized")


def write_fid_only_table(metrics_rows: List[Dict[str, str]], diagnostics_dir: Path) -> None:
    rows = []
    for row in metrics_rows:
        method = row.get("method", "")
        fid = as_float(row.get("mnist_fid"))
        if method and math.isfinite(fid):
            rows.append((method, fid))
    if not rows:
        return
    rows.sort(key=lambda item: item[1])
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    with open(diagnostics_dir / "mnist_fid_only.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "MNIST-FID"])
        writer.writerows((method, f"{fid:.6f}") for method, fid in rows)

    lines = ["| Method | MNIST-FID |", "|---|---:|"]
    for method, fid in rows:
        lines.append(f"| {method} | {fid:.3f} |")
    (diagnostics_dir / "mnist_fid_only.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(5.2, 0.48 * len(rows) + 1.1))
    ax.axis("off")
    table = ax.table(
        cellText=[[method, f"{fid:.3f}"] for method, fid in rows],
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
    fig.savefig(diagnostics_dir / "mnist_fid_only_table.png", dpi=220)
    plt.close(fig)


def plot_wbvm_tau_selection(outdir: Path, diagnostics_dir: Path) -> None:
    rows = read_csv(outdir / "wbvm_single_selection.csv")
    points = []
    for row in rows:
        tau = as_float(row.get("tau"))
        score = as_float(row.get("val_score"))
        if math.isfinite(tau) and math.isfinite(score):
            points.append((tau, score))
    if not points:
        return
    points.sort()
    xs = np.asarray([p[0] for p in points], dtype=float)
    ys = np.asarray([p[1] for p in points], dtype=float)
    best_idx = int(np.argmin(ys))
    fig, ax = plt.subplots(figsize=(3.25, 2.5))
    ax.plot(xs, ys, marker="o", color="#264653")
    ax.scatter([xs[best_idx]], [ys[best_idx]], color="#E76F51", zorder=4, label=f"selected tau={xs[best_idx]:g}")
    ax.set_xlabel("WBVM-single tau")
    ax.set_ylabel("Validation MNIST-FID")
    ax.set_title("WBVM-single Selection")
    ax.legend(loc="best")
    save_fig(fig, diagnostics_dir / "wbvm_single_tau_selection")


def plot_training_time(metrics_rows: List[Dict[str, str]], diagnostics_dir: Path) -> None:
    rows = []
    for row in metrics_rows:
        sec = as_float(row.get("train_seconds"))
        method = row.get("method", "")
        if method and math.isfinite(sec):
            rows.append((method, sec / 60.0))
    if not rows:
        return
    rows.sort(key=lambda item: item[1])
    fig, ax = plt.subplots(figsize=(4.8, 0.45 * len(rows) + 1.3))
    y = np.arange(len(rows))
    ax.barh(y, [value for _, value in rows], color="#2A9D8F", edgecolor="white", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([name for name, _ in rows])
    ax.set_xlabel("Training time (minutes)")
    ax.set_title("Training Cost")
    for yi, (_, value) in zip(y, rows):
        ax.text(value, yi, f" {value:.1f}", va="center", fontsize=8)
    save_fig(fig, diagnostics_dir / "training_time_by_method")


def plot_metric_scatter(metrics_rows: List[Dict[str, str]], diagnostics_dir: Path) -> None:
    points = []
    for row in metrics_rows:
        fid = as_float(row.get("mnist_fid"))
        kid = as_float(row.get("mnist_kid_x1000"))
        method = row.get("method", "")
        if method and math.isfinite(fid) and math.isfinite(kid):
            points.append((method, fid, kid))
    if len(points) < 2:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for idx, (method, fid, kid) in enumerate(points):
        ax.scatter(fid, kid, s=42, color=COLORS[idx % len(COLORS)], edgecolor="white", linewidth=0.7)
        ax.text(fid, kid, f" {method}", va="center", fontsize=8)
    ax.set_xlabel("MNIST-FID")
    ax.set_ylabel("KID-z x1000")
    ax.set_title("Feature-Metric Agreement")
    save_fig(fig, diagnostics_dir / "mnist_fid_vs_kid")


def make_tiled_image(images: np.ndarray, rows: int, cols: int) -> np.ndarray:
    images = images[: rows * cols]
    if images.shape[0] < rows * cols:
        pad = np.zeros((rows * cols - images.shape[0],) + images.shape[1:], dtype=images.dtype)
        images = np.concatenate([images, pad], axis=0)
    h, w = images.shape[-2], images.shape[-1]
    canvas = np.zeros((rows * h, cols * w), dtype=float)
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = img[0]
    return canvas


def plot_sample_panel(outdir: Path, diagnostics_dir: Path) -> None:
    sample_path = outdir / "sample_grid.pt"
    if not sample_path.is_file():
        return
    import torch
    from mnist_fid_experiment import from_model_space

    samples = torch.load(sample_path, map_location="cpu")
    if not isinstance(samples, dict):
        return
    methods = [m for m in METHOD_ORDER if m in samples]
    methods.extend([m for m in samples.keys() if m not in methods])
    cols_per_method = 8
    rows_per_method = 4
    fig, axes = plt.subplots(1, len(methods), figsize=(2.0 * len(methods), 2.65))
    if len(methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        tensor = from_model_space(samples[method][: rows_per_method * cols_per_method]).detach().cpu().clamp(0, 1)
        tile = make_tiled_image(tensor.numpy(), rows_per_method, cols_per_method)
        ax.imshow(tile, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(method, fontsize=10, pad=5)
        ax.axis("off")
    fig.subplots_adjust(wspace=0.05)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(diagnostics_dir / "mnist_samples_panel.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_diagnostics_summary(metrics_rows: List[Dict[str, str]], diagnostics_dir: Path) -> None:
    lines = ["| Method | MNIST-FID | KID-z x1000 | Loss | Train minutes |", "|---|---:|---:|---:|---:|"]
    for row in sorted(metrics_rows, key=lambda r: as_float(r.get("mnist_fid"))):
        method = row.get("method", "")
        fid = as_float(row.get("mnist_fid"))
        kid = as_float(row.get("mnist_kid_x1000"))
        loss = as_float(row.get("loss"))
        minutes = as_float(row.get("train_seconds")) / 60.0
        lines.append(
            f"| {method} | {fid:.3f} | {kid:.3f} | {loss:.5g} | {minutes:.1f} |"
        )
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    (diagnostics_dir / "diagnostics_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot diagnostics for MNIST training logs")
    parser.add_argument("--outdir", default="outputs_mnist_unified_dit_mnistfid")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = SCRIPT_DIR / outdir
    diagnostics_dir = outdir / "diagnostics"
    metrics_rows = read_csv(outdir / "metrics_summary.csv")
    runs = load_training_runs(outdir, metrics_rows)
    plot_loss_curves(runs, diagnostics_dir)
    write_fid_only_table(metrics_rows, diagnostics_dir)
    plot_wbvm_tau_selection(outdir, diagnostics_dir)
    plot_training_time(metrics_rows, diagnostics_dir)
    plot_metric_scatter(metrics_rows, diagnostics_dir)
    plot_sample_panel(outdir, diagnostics_dir)
    write_diagnostics_summary(metrics_rows, diagnostics_dir)
    print(f"Saved diagnostics to {diagnostics_dir}")


if __name__ == "__main__":
    main()
