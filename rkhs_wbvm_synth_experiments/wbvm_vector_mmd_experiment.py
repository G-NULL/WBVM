import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from wbvm_rkhs_experiment import (
    DATASETS,
    MLP,
    ToyData,
    TrainConfig,
    configure_runtime,
    compute_bridge_sigmas,
    endpoint_validation_metric,
    make_toy_data,
    normalize_velocity,
    parse_float_list,
    preset_config,
    rbf_mmd2,
    set_seed,
    table1_metrics,
    take_batch,
    tau_scalar,
    timed_endpoint_samples,
    timed_fm_samples,
    train_fm,
    write_table1_artifacts,
    plot_samples_panel,
)


def vector_flux_kernel_bilinear_means(
    x: torch.Tensor,
    v: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    sigmas: torch.Tensor,
    drop_diagonal: bool = False,
) -> torch.Tensor:
    """Mean of v^T k(x,y) w for the vector-valued kernel K=k I_D.

    This is the route-two objective from the WBVM notes. It treats
    grad(phi) as the vector critic and therefore uses no kernel Hessian.
    """
    sq = torch.cdist(x, y).pow(2)
    dot_vw = v @ w.T
    sigma2 = sigmas.to(x.device, x.dtype).flatten().pow(2).clamp_min(1e-6).view(-1, 1, 1)
    values = torch.exp(-0.5 * sq.unsqueeze(0) / sigma2) * dot_vw.unsqueeze(0)
    if drop_diagonal:
        if x.shape[0] != y.shape[0]:
            raise ValueError("drop_diagonal=True requires square same-size batches")
        n = values.shape[1]
        if n <= 1:
            return values.new_zeros(values.shape[0])
        total = values.sum(dim=(1, 2)) - torch.diagonal(values, dim1=1, dim2=2).sum(dim=1)
        return total / (n * (n - 1))
    return values.mean(dim=(1, 2))


def vector_flux_mmd2(
    x_model: torch.Tensor,
    v_model: torch.Tensor,
    x_data: torch.Tensor,
    v_data: torch.Tensor,
    sigmas: torch.Tensor,
    include_data_data: bool = True,
    statistic: str = "v",
) -> torch.Tensor:
    """Squared vector-flux MMD for K(x,x') = k(x,x') I_D.

    statistic="v" is the direct biased V-statistic. statistic="u" drops
    within-group diagonals as a finite-batch stabilizer.
    """
    if statistic not in {"v", "u"}:
        raise ValueError(f"Unknown vector-flux statistic {statistic!r}")
    drop_diagonal = statistic == "u"
    sigmas = sigmas.to(x_model.device, x_model.dtype).flatten()
    mm = vector_flux_kernel_bilinear_means(
        x_model,
        v_model,
        x_model,
        v_model,
        sigmas,
        drop_diagonal=drop_diagonal,
    )
    md = vector_flux_kernel_bilinear_means(
        x_model,
        v_model,
        x_data,
        v_data,
        sigmas,
        drop_diagonal=False,
    )
    loss_per_scale = mm - 2.0 * md
    if include_data_data:
        dd = vector_flux_kernel_bilinear_means(
            x_data,
            v_data,
            x_data,
            v_data,
            sigmas,
            drop_diagonal=drop_diagonal,
        )
        loss_per_scale = loss_per_scale + dd
    return loss_per_scale.mean()


def train_vector_mmd_wbvm(
    data: ToyData,
    cfg: TrainConfig,
    mode: str,
    center_tau: float,
    train_n: Optional[int] = None,
    kernel_batch: Optional[int] = None,
    steps_override: Optional[int] = None,
    statistic: str = "v",
    verbose: bool = False,
) -> Tuple[nn.Module, Dict[str, float]]:
    device = data.train.device
    train_x = data.train if train_n is None or train_n >= data.train.shape[0] else data.train[:train_n]
    batch = kernel_batch or cfg.batch
    latent_dim = data.D
    kernel_sigmas, kernel_base_sigma = compute_bridge_sigmas(data, cfg, mode, center_tau, train_x=train_x)
    model = MLP(latent_dim, data.D, cfg.width, cfg.depth).to(device)
    model.wbvm_kernel_sigmas = kernel_sigmas.detach()
    model.wbvm_kernel_base_sigma = kernel_base_sigma
    model.wbvm_flux_statistic = statistic
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    ema_loss = None
    ema_model_v_rms = None
    ema_data_v_rms = None
    start = time.time()
    total_steps = steps_override or cfg.steps
    train_tau_low = (
        cfg.all_tau_low
        if mode == "all"
        else (max(cfg.tau_low, center_tau - cfg.local_delta) if mode == "local" else center_tau)
    )
    train_tau_high = (
        cfg.all_tau_high
        if mode == "all"
        else (min(cfg.tau_high, center_tau + cfg.local_delta) if mode == "local" else center_tau)
    )

    for step in range(1, total_steps + 1):
        tau = tau_scalar(mode, center_tau, cfg, device)
        x1 = take_batch(train_x, batch)
        x0 = torch.randn(batch, data.D, device=device)
        u = x0 if cfg.tie_latent_to_base else torch.randn(batch, latent_dim, device=device)
        y = model(u)
        x_model = (1.0 - tau) * x0 + tau * y
        v_model = y - x0
        x_data = (1.0 - tau) * x0 + tau * x1
        v_data = x1 - x0
        if cfg.normalize_velocity:
            v_model_for_loss, v_model_rms = normalize_velocity(v_model, cfg.velocity_eps)
            v_data_for_loss, v_data_rms = normalize_velocity(v_data, cfg.velocity_eps)
        else:
            v_model_for_loss = v_model
            v_data_for_loss = v_data
            v_model_rms = torch.sqrt(v_model.pow(2).sum(dim=1).mean()).detach()
            v_data_rms = torch.sqrt(v_data.pow(2).sum(dim=1).mean()).detach()

        flux = vector_flux_mmd2(
            x_model,
            v_model_for_loss,
            x_data,
            v_data_for_loss,
            sigmas=kernel_sigmas,
            include_data_data=False,
            statistic=statistic,
        )
        loss = flux
        if cfg.lambda_marg > 0:
            loss = loss + cfg.lambda_marg * rbf_mmd2(x_model, x_data, sigmas=kernel_sigmas)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        loss_value = float(loss.detach().cpu())
        ema_loss = loss_value if ema_loss is None else 0.98 * ema_loss + 0.02 * loss_value
        model_v_rms_value = float(v_model_rms.detach().cpu())
        data_v_rms_value = float(v_data_rms.detach().cpu())
        ema_model_v_rms = model_v_rms_value if ema_model_v_rms is None else 0.98 * ema_model_v_rms + 0.02 * model_v_rms_value
        ema_data_v_rms = data_v_rms_value if ema_data_v_rms is None else 0.98 * ema_data_v_rms + 0.02 * data_v_rms_value
        if verbose and (step == 1 or step % max(total_steps // 4, 1) == 0):
            print(
                f"{data.name:10s} vMMD  {mode:6s} tau={center_tau:.2f} "
                f"step={step:5d} loss={ema_loss:.5g}",
                flush=True,
            )

    return model, {
        "train_seconds": time.time() - start,
        "last_loss": float(ema_loss or 0.0),
        "train_steps": total_steps,
        "latent_dim": latent_dim,
        "time_conditioned": False,
        "loss_statistic": f"vector_flux_mmd_{statistic}_theta_terms",
        "common_base_noise": True,
        "tie_latent_to_base": cfg.tie_latent_to_base,
        "normalize_velocity": cfg.normalize_velocity,
        "model_velocity_rms": float(ema_model_v_rms or 0.0),
        "data_velocity_rms": float(ema_data_v_rms or 0.0),
        "train_tau_low": float(train_tau_low),
        "train_tau_high": float(train_tau_high),
        "kernel_base_sigma": kernel_base_sigma,
        "kernel_sigmas": ",".join(f"{float(s.detach().cpu()):.6g}" for s in kernel_sigmas),
    }


@torch.no_grad()
def heldout_vector_flux_residual(
    model: nn.Module,
    data: ToyData,
    cfg: TrainConfig,
    tau: float,
    batch: int = 512,
    split: str = "test",
    statistic: str = "v",
) -> float:
    device = data.train.device
    tau_t = torch.tensor(tau, device=device)
    if split == "train":
        target = data.train
    elif split == "val":
        target = data.val
    elif split == "test":
        target = data.test
    else:
        raise ValueError(f"Unknown heldout split {split}")
    x1 = take_batch(target, batch)
    x0 = torch.randn(batch, data.D, device=device)
    u = x0 if cfg.tie_latent_to_base else torch.randn(batch, data.D, device=device)
    y = model(u)
    x_model = (1.0 - tau_t) * x0 + tau_t * y
    v_model = y - x0
    x_data = (1.0 - tau_t) * x0 + tau_t * x1
    v_data = x1 - x0
    if cfg.normalize_velocity:
        v_model, _ = normalize_velocity(v_model, cfg.velocity_eps)
        v_data, _ = normalize_velocity(v_data, cfg.velocity_eps)
    sigmas = getattr(model, "wbvm_kernel_sigmas", None)
    if sigmas is None:
        sigmas, _ = compute_bridge_sigmas(data, cfg, "single", tau)
    return float(
        vector_flux_mmd2(
            x_model,
            v_model,
            x_data,
            v_data,
            sigmas=sigmas,
            include_data_data=True,
            statistic=statistic,
        )
        .detach()
        .cpu()
    )


@torch.no_grad()
def vector_single_validation_score(
    model: nn.Module,
    data: ToyData,
    cfg: TrainConfig,
    tau: float,
    val_metric: str,
    seed: int,
    statistic: str,
) -> float:
    if val_metric == "flux":
        batch = min(1024, cfg.val_n, data.val.shape[0])
        return heldout_vector_flux_residual(model, data, cfg, tau, batch=batch, split="val", statistic=statistic)
    raw = timed_endpoint_samples(model, data, cfg.val_n)[0]
    return endpoint_validation_metric(raw, data.val_raw, val_metric, seed)


def train_validation_selected_single(
    data: ToyData,
    cfg: TrainConfig,
    tau_candidates: List[float],
    val_metric: str,
    dataset_seed: int,
    statistic: str,
    verbose: bool = False,
) -> Tuple[nn.Module, Dict[str, float], List[Dict[str, object]]]:
    best_model: Optional[nn.Module] = None
    best_info: Dict[str, float] = {}
    best_metric = float("inf")
    best_tau = float("nan")
    rows: List[Dict[str, object]] = []

    for tau in tau_candidates:
        set_seed(dataset_seed + int(round(1000 * tau)))
        model, info = train_vector_mmd_wbvm(
            data,
            cfg,
            mode="single",
            center_tau=tau,
            steps_override=cfg.single_steps,
            statistic=statistic,
            verbose=verbose,
        )
        val_score = vector_single_validation_score(
            model,
            data,
            cfg,
            tau,
            val_metric,
            dataset_seed + int(round(1000 * tau)),
            statistic,
        )
        row = {"dataset": data.name, "tau": tau, "val_metric": val_metric, "val_score": val_score, **info}
        rows.append(row)
        print(f"{data.name:10s} vMMD-single-val tau={tau:.2f} {val_metric}={val_score:.4f}", flush=True)
        if val_score < best_metric:
            best_metric = val_score
            best_tau = tau
            best_model = model
            best_info = info

    if best_model is None:
        raise RuntimeError("No validation-selected model was trained")
    best_info = dict(best_info)
    best_info["selected_tau"] = best_tau
    best_info["validation_metric"] = val_metric
    best_info["validation_score"] = best_metric
    return best_model, best_info, rows


def apply_overrides(cfg: TrainConfig, args: argparse.Namespace) -> None:
    if args.steps is not None:
        cfg.steps = args.steps
    if args.single_steps is not None:
        cfg.single_steps = args.single_steps
    if args.fm_steps is not None:
        cfg.fm_steps = args.fm_steps
    if args.fm_sample_steps is not None:
        cfg.fm_sample_steps = args.fm_sample_steps
    if args.tau_low is not None:
        cfg.tau_low = args.tau_low
        if args.all_tau_low is None:
            cfg.all_tau_low = args.tau_low
    if args.tau_high is not None:
        cfg.tau_high = args.tau_high
        if args.all_tau_high is None:
            cfg.all_tau_high = args.tau_high
    if args.all_tau_low is not None:
        cfg.all_tau_low = args.all_tau_low
    if args.all_tau_high is not None:
        cfg.all_tau_high = args.all_tau_high
    if args.batch is not None:
        cfg.batch = args.batch
    if args.width is not None:
        cfg.width = args.width
    if args.depth is not None:
        cfg.depth = args.depth
    if args.lambda_marg is not None:
        cfg.lambda_marg = args.lambda_marg
    if args.no_velocity_norm:
        cfg.normalize_velocity = False
    if args.velocity_eps is not None:
        cfg.velocity_eps = args.velocity_eps
    if args.tie_latent_to_base:
        cfg.tie_latent_to_base = True
    if args.kernel_scales is not None:
        cfg.kernel_scales = parse_float_list(args.kernel_scales)
    if args.kernel_bandwidth_points is not None:
        cfg.kernel_bandwidth_points = args.kernel_bandwidth_points


def run_main(args: argparse.Namespace) -> None:
    cfg = preset_config(args.preset, args.seed)
    apply_overrides(cfg, args)
    runtime = configure_runtime(args.num_threads)
    set_seed(cfg.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "vector_flux_mmd_wbvm",
                "config": asdict(cfg),
                "device": str(device),
                "runtime": runtime,
                "args": vars(args),
            },
            f,
            indent=2,
        )

    print(f"Running vector-flux MMD WBVM synthetic experiments on {device} with preset={cfg.preset}", flush=True)
    print("This script uses fixed-level vector-flux MMD losses; it does not use a space-time all-time residual.", flush=True)
    print(f"Runtime: {runtime}", flush=True)
    selected_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    tau_candidates = parse_float_list(args.single_taus)
    bad = [d for d in selected_datasets if d not in DATASETS]
    if bad:
        raise ValueError(f"Unknown datasets: {bad}; allowed: {DATASETS}")
    if not tau_candidates:
        raise ValueError("single_taus must contain at least one candidate tau")
    data_map = {
        name: make_toy_data(name, args.train_n, cfg.val_n, cfg.test_n, cfg.seed + 1000 * i, device)
        for i, name in enumerate(selected_datasets)
    }

    metrics: List[Dict[str, object]] = []
    table_rows: List[Dict[str, object]] = []
    single_selection_rows: List[Dict[str, object]] = []
    samples: Dict[Tuple[str, str], np.ndarray] = {}

    for i, name in enumerate(selected_datasets):
        data = data_map[name]
        print(f"\n[{i + 1}/{len(selected_datasets)}] Vector-flux MMD models for {name}", flush=True)
        all_model, all_info = train_vector_mmd_wbvm(
            data,
            cfg,
            mode="all",
            center_tau=0.4,
            statistic=args.loss_statistic,
            verbose=args.verbose,
        )
        single_model, single_info, selection_rows = train_validation_selected_single(
            data,
            cfg,
            tau_candidates=tau_candidates,
            val_metric=args.val_metric,
            dataset_seed=cfg.seed + 1000 * i,
            statistic=args.loss_statistic,
            verbose=args.verbose,
        )
        single_selection_rows.extend(selection_rows)
        fm_model, fm_info = train_fm(data, cfg, verbose=args.verbose)

        raw_all, time_all = timed_endpoint_samples(all_model, data, args.table_samples)
        raw_single, time_single = timed_endpoint_samples(single_model, data, args.table_samples)
        raw_fm, time_fm = timed_fm_samples(fm_model, data, args.table_samples, cfg.fm_sample_steps)
        sample_records = [
            ("WBVM-all", raw_all, time_all, 1, f"[{cfg.all_tau_low:.2f},{cfg.all_tau_high:.2f}]", all_model, all_info),
            ("WBVM-single", raw_single, time_single, 1, f"{single_info['selected_tau']:.2f}", single_model, single_info),
        ]

        for label, raw, sample_time, nfe, tau_setting, model, info in sample_records:
            ev = table1_metrics(name, label, raw, data.test_raw, sample_time, nfe, cfg.seed + i, tau_setting=tau_setting)
            table_rows.append(dict(ev))
            flux_resid = heldout_vector_flux_residual(
                model,
                data,
                cfg,
                0.4,
                batch=min(512, cfg.eval_n),
                statistic=args.loss_statistic,
            )
            metrics.append(
                {
                    "dataset": name,
                    "method": f"vMMD-{label}",
                    "factor": "main",
                    "value": 0.0,
                    **ev,
                    "heldout_vector_flux_tau_0.4": flux_resid,
                    **info,
                }
            )
            print(
                f"{name:10s} {'vMMD-' + label:17s} "
                f"SW2={ev['sliced_w2']:.4f} W2={ev['w2']:.4f} "
                f"out={100 * ev['off_manifold_rate']:.2f}%",
                flush=True,
            )

        ev_fm = table1_metrics(
            name,
            "Flow matching",
            raw_fm,
            data.test_raw,
            time_fm,
            cfg.fm_sample_steps,
            cfg.seed + i,
            tau_setting=f"[{cfg.tau_low:.2f},{cfg.tau_high:.2f}]",
        )
        metrics.append({"dataset": name, "method": "Flow matching", "factor": "main", "value": 0.0, **ev_fm, **fm_info})
        table_rows.append(dict(ev_fm))
        print(
            f"{name:10s} {'Flow matching':17s} "
            f"SW2={ev_fm['sliced_w2']:.4f} W2={ev_fm['w2']:.4f} "
            f"out={100 * ev_fm['off_manifold_rate']:.2f}%",
            flush=True,
        )
        samples[(name, "all")] = raw_all
        samples[(name, "single")] = raw_single
        samples[(name, "fm")] = raw_fm

    plot_samples_panel(outdir / "vector_mmd_wbvm_toy_6x4.png", data_map, samples, selected_datasets)
    write_table1_artifacts(outdir, table_rows)
    if single_selection_rows:
        with open(outdir / "vector_mmd_single_validation_selection.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(single_selection_rows[0].keys()))
            writer.writeheader()
            writer.writerows(single_selection_rows)

    metric_path = outdir / "metrics_summary.csv"
    metric_keys = sorted({k for row in metrics for k in row.keys()})
    with open(metric_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metric_keys)
        writer.writeheader()
        writer.writerows(metrics)

    print(f"\nDone. Outputs written to {outdir.resolve()}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vector-flux MMD WBVM synthetic experiments")
    parser.add_argument("--preset", choices=["smoke", "quick", "standard"], default="quick")
    parser.add_argument("--outdir", default="outputs_vector_mmd")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default=None)
    parser.add_argument("--train-n", type=int, default=20000)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--single-steps", type=int, default=None)
    parser.add_argument("--fm-steps", type=int, default=None)
    parser.add_argument("--fm-sample-steps", type=int, default=None)
    parser.add_argument("--tau-low", type=float, default=None)
    parser.add_argument("--tau-high", type=float, default=None)
    parser.add_argument("--all-tau-low", type=float, default=None)
    parser.add_argument("--all-tau-high", type=float, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--lambda-marg", type=float, default=None)
    parser.add_argument("--no-velocity-norm", action="store_true")
    parser.add_argument("--velocity-eps", type=float, default=None)
    parser.add_argument("--tie-latent-to-base", action="store_true")
    parser.add_argument("--kernel-scales", default=None)
    parser.add_argument("--kernel-bandwidth-points", type=int, default=None)
    parser.add_argument("--loss-statistic", choices=["v", "u"], default="v")
    parser.add_argument("--table-samples", type=int, default=10000)
    parser.add_argument("--single-taus", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--val-metric", choices=["flux", "sliced_w2", "w2", "energy"], default="flux")
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--verbose", action="store_true")
    return parser


if __name__ == "__main__":
    run_main(build_parser().parse_args())
