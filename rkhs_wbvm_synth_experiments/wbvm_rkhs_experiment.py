import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


DATASETS = ["rings2d", "spiral2d", "moons2d", "checker2d", "helix3d", "torus3d"]


def intrinsic_dim(name: str) -> int:
    return {"rings2d": 1, "spiral2d": 1, "moons2d": 2, "checker2d": 2, "helix3d": 1, "torus3d": 2}[name]


def ambient_dim(name: str) -> int:
    return 3 if name in {"helix3d", "torus3d"} else 2


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime(num_threads: Optional[int]) -> Dict[str, object]:
    cpu_count = os.cpu_count() or 1
    threads = num_threads if num_threads is not None else min(20, cpu_count)
    threads = max(1, min(int(threads), cpu_count))
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, min(4, threads)))
    except RuntimeError:
        pass
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    return {
        "cpu_count": cpu_count,
        "torch_num_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32) if torch.cuda.is_available() else False,
    }


def sample_toy_np(name: str, n: int, seed: int, jitter: float = 0.02) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if name == "rings2d":
        radii = np.array([0.45, 0.75, 1.05, 1.35], dtype=np.float32)
        idx = rng.integers(0, len(radii), size=n)
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
        x = np.stack([radii[idx] * np.cos(theta), radii[idx] * np.sin(theta)], axis=1)
        x += rng.normal(0.0, jitter, size=x.shape)
        return x.astype(np.float32)

    if name == "spiral2d":
        t = rng.uniform(0.4, 4.5 * np.pi, size=n)
        r = 0.10 + 0.105 * t
        x = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
        x += rng.normal(0.0, jitter, size=x.shape)
        return x.astype(np.float32)

    if name == "moons2d":
        c = rng.integers(0, 2, size=n)
        theta = rng.uniform(0.0, np.pi, size=n)
        x0 = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        x1 = np.stack([1.0 - np.cos(theta), 1.0 - np.sin(theta) - 0.45], axis=1)
        x = np.where(c[:, None] == 0, x0, x1)
        x += rng.normal(0.0, jitter, size=x.shape)
        return x.astype(np.float32)

    if name == "checker2d":
        cells = 4
        width = 2.0 / cells
        valid = [(i, j) for i in range(cells) for j in range(cells) if (i + j) % 2 == 0]
        chosen = np.array(valid, dtype=np.int64)[rng.integers(0, len(valid), size=n)]
        lo = -1.0 + chosen * width
        x = lo + rng.uniform(0.0, width, size=(n, 2))
        x += rng.normal(0.0, jitter, size=x.shape)
        return x.astype(np.float32)

    if name == "helix3d":
        t = rng.uniform(0.0, 2.0 * np.pi, size=n)
        z = (t - np.pi) / np.pi
        x = np.stack([np.cos(t), np.sin(t), z], axis=1)
        x += rng.normal(0.0, jitter, size=x.shape)
        return x.astype(np.float32)

    if name == "torus3d":
        u = rng.uniform(0.0, 2.0 * np.pi, size=n)
        v = rng.uniform(0.0, 2.0 * np.pi, size=n)
        major, minor = 1.0, 0.38
        x = np.stack(
            [
                (major + minor * np.cos(v)) * np.cos(u),
                (major + minor * np.cos(v)) * np.sin(u),
                minor * np.sin(v),
            ],
            axis=1,
        )
        x += rng.normal(0.0, jitter, size=x.shape)
        return x.astype(np.float32)

    raise ValueError(f"Unknown dataset {name}")


def manifold_distance_np(name: str, x: np.ndarray) -> np.ndarray:
    """Distance to the noiseless support used for the Section 6.1-style toys."""
    if name == "rings2d":
        radii = np.array([0.45, 0.75, 1.05, 1.35], dtype=np.float32)
        r = np.linalg.norm(x[:, :2], axis=1)
        return np.min(np.abs(r[:, None] - radii[None, :]), axis=1)

    if name == "spiral2d":
        t = np.linspace(0.4, 4.5 * np.pi, 4096)
        r = 0.10 + 0.105 * t
        curve = np.stack([r * np.cos(t), r * np.sin(t)], axis=1)
        chunks = []
        for start in range(0, x.shape[0], 2048):
            xx = x[start : start + 2048, :2]
            d2 = ((xx[:, None, :] - curve[None, :, :]) ** 2).sum(axis=2)
            chunks.append(np.sqrt(d2.min(axis=1)))
        return np.concatenate(chunks)

    if name == "moons2d":
        theta = np.linspace(0.0, np.pi, 2048)
        moon0 = np.stack([np.cos(theta), np.sin(theta)], axis=1)
        moon1 = np.stack([1.0 - np.cos(theta), 1.0 - np.sin(theta) - 0.45], axis=1)
        curve = np.concatenate([moon0, moon1], axis=0)
        chunks = []
        for start in range(0, x.shape[0], 2048):
            xx = x[start : start + 2048, :2]
            d2 = ((xx[:, None, :] - curve[None, :, :]) ** 2).sum(axis=2)
            chunks.append(np.sqrt(d2.min(axis=1)))
        return np.concatenate(chunks)

    if name == "checker2d":
        cells = 4
        width = 2.0 / cells
        valid = [(i, j) for i in range(cells) for j in range(cells) if (i + j) % 2 == 0]
        dists = []
        xx = x[:, :2]
        for i, j in valid:
            lo = np.array([-1.0 + i * width, -1.0 + j * width])
            hi = lo + width
            delta = np.maximum(np.maximum(lo - xx, 0.0), xx - hi)
            dists.append(np.linalg.norm(delta, axis=1))
        return np.min(np.stack(dists, axis=1), axis=1)

    if name == "helix3d":
        t = np.linspace(0.0, 2.0 * np.pi, 4096)
        curve = np.stack([np.cos(t), np.sin(t), (t - np.pi) / np.pi], axis=1)
        chunks = []
        for start in range(0, x.shape[0], 2048):
            xx = x[start : start + 2048, :3]
            d2 = ((xx[:, None, :] - curve[None, :, :]) ** 2).sum(axis=2)
            chunks.append(np.sqrt(d2.min(axis=1)))
        return np.concatenate(chunks)

    if name == "torus3d":
        major, minor = 1.0, 0.38
        radial = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
        return np.abs(np.sqrt((radial - major) ** 2 + x[:, 2] ** 2) - minor)

    raise ValueError(f"Unknown dataset {name}")


@dataclass
class ToyData:
    name: str
    train: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    train_raw: np.ndarray
    val_raw: np.ndarray
    test_raw: np.ndarray

    @property
    def D(self) -> int:
        return self.train.shape[1]

    @property
    def d(self) -> int:
        return intrinsic_dim(self.name)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.to(x.device) + self.mean.to(x.device)


def make_toy_data(name: str, n_train: int, n_val: int, n_test: int, seed: int, device: torch.device) -> ToyData:
    train_raw = sample_toy_np(name, n_train, seed + 11)
    val_raw = sample_toy_np(name, n_val, seed + 23)
    test_raw = sample_toy_np(name, n_test, seed + 37)
    mean_np = train_raw.mean(axis=0, keepdims=True)
    std_np = train_raw.std(axis=0, keepdims=True) + 1e-6
    train = torch.tensor((train_raw - mean_np) / std_np, device=device)
    val = torch.tensor((val_raw - mean_np) / std_np, device=device)
    test = torch.tensor((test_raw - mean_np) / std_np, device=device)
    mean = torch.tensor(mean_np.squeeze(0), device=device)
    std = torch.tensor(std_np.squeeze(0), device=device)
    return ToyData(name, train, val, test, mean, std, train_raw, val_raw, test_raw)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int, depth: int):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(last, width))
            layers.append(nn.ReLU())
            last = width
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TimeVelocityMLP(nn.Module):
    def __init__(self, dim: int, width: int, depth: int, t_features: int = 16):
        super().__init__()
        self.t_features = t_features
        self.mlp = MLP(dim + 2 * t_features, dim, width, depth)

    def time_embed(self, tau: torch.Tensor) -> torch.Tensor:
        freqs = torch.arange(1, self.t_features + 1, device=tau.device, dtype=tau.dtype)
        angles = 2.0 * math.pi * tau[:, None] * freqs[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        if tau.ndim == 0:
            tau = tau.expand(x.shape[0])
        if tau.ndim == 1:
            tau = tau.to(x.device, x.dtype)
        return self.mlp(torch.cat([x, self.time_embed(tau)], dim=1))


def take_batch(x: torch.Tensor, batch: int) -> torch.Tensor:
    idx = torch.randint(0, x.shape[0], (batch,), device=x.device)
    return x[idx]


def pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x, y).pow(2)


@torch.no_grad()
def median_bandwidth(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = torch.cat([x.detach(), y.detach()], dim=0)
    return median_bandwidth_from_samples(z)


@torch.no_grad()
def median_bandwidth_from_samples(
    z: torch.Tensor,
    max_points: int = 512,
    sigma_min: float = 0.25,
    sigma_max: float = 10.0,
) -> torch.Tensor:
    z = z.detach()
    if z.shape[0] > max_points:
        z = z[torch.randperm(z.shape[0], device=z.device)[:max_points]]
    dist = torch.pdist(z).pow(2)
    med = torch.median(dist[dist > 1e-12]) if torch.any(dist > 1e-12) else torch.tensor(1.0, device=z.device)
    sigma = torch.sqrt(0.5 * med).clamp(sigma_min, sigma_max)
    return sigma


def derivative_kernel_bilinear(
    x: torch.Tensor, v: torch.Tensor, y: torch.Tensor, w: torch.Tensor, sigma: torch.Tensor
) -> torch.Tensor:
    diff = x[:, None, :] - y[None, :, :]
    sq = diff.pow(2).sum(dim=-1)
    sigma2 = sigma.pow(2).clamp_min(1e-6)
    k = torch.exp(-0.5 * sq / sigma2)
    dot_vw = v @ w.T
    v_diff = (v[:, None, :] * diff).sum(dim=-1)
    w_diff = (w[None, :, :] * diff).sum(dim=-1)
    return k * (dot_vw / sigma2 - (v_diff * w_diff) / sigma2.pow(2))


def derivative_kernel_bilinear_means(
    x: torch.Tensor,
    v: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    sigmas: torch.Tensor,
    drop_diagonal: bool = False,
) -> torch.Tensor:
    diff = x[:, None, :] - y[None, :, :]
    sq = diff.pow(2).sum(dim=-1)
    dot_vw = v @ w.T
    v_diff = (v[:, None, :] * diff).sum(dim=-1)
    w_diff = (w[None, :, :] * diff).sum(dim=-1)
    sigma2 = sigmas.to(x.device, x.dtype).flatten().pow(2).clamp_min(1e-6).view(-1, 1, 1)
    k = torch.exp(-0.5 * sq.unsqueeze(0) / sigma2)
    values = k * (
        dot_vw.unsqueeze(0) / sigma2
        - (v_diff * w_diff).unsqueeze(0) / sigma2.pow(2)
    )
    if drop_diagonal:
        n = values.shape[1]
        if n <= 1:
            return values.new_zeros(values.shape[0])
        total = values.sum(dim=(1, 2)) - torch.diagonal(values, dim1=1, dim2=2).sum(dim=1)
        return total / (n * (n - 1))
    return values.mean(dim=(1, 2))


def rkhs_flux_u_stat(
    x_model: torch.Tensor,
    v_model: torch.Tensor,
    x_data: torch.Tensor,
    v_data: torch.Tensor,
    sigma: Optional[torch.Tensor] = None,
    sigmas: Optional[torch.Tensor] = None,
    include_data_data: bool = True,
) -> torch.Tensor:
    if sigmas is None:
        if sigma is None:
            sigma = median_bandwidth(x_model, x_data)
        sigmas = sigma.reshape(1)
    sigmas = sigmas.to(x_model.device, x_model.dtype).flatten()
    mm = derivative_kernel_bilinear_means(x_model, v_model, x_model, v_model, sigmas, drop_diagonal=True)
    md = derivative_kernel_bilinear_means(x_model, v_model, x_data, v_data, sigmas)
    loss_per_scale = mm - 2.0 * md
    if include_data_data:
        dd = derivative_kernel_bilinear_means(x_data, v_data, x_data, v_data, sigmas, drop_diagonal=True)
        loss_per_scale = loss_per_scale + dd
    return loss_per_scale.mean()


def rkhs_flux_v_stat(
    x_model: torch.Tensor,
    v_model: torch.Tensor,
    x_data: torch.Tensor,
    v_data: torch.Tensor,
    sigma: Optional[torch.Tensor] = None,
    sigmas: Optional[torch.Tensor] = None,
    include_data_data: bool = True,
) -> torch.Tensor:
    """Biased RKHS squared flux discrepancy.

    With include_data_data=False this returns the theta-dependent part only.
    Its gradient equals the gradient of the complete V-statistic because the
    data-data term has no theta dependence.
    """
    if sigmas is None:
        if sigma is None:
            sigma = median_bandwidth(x_model, x_data)
        sigmas = sigma.reshape(1)
    sigmas = sigmas.to(x_model.device, x_model.dtype).flatten()
    mm = derivative_kernel_bilinear_means(x_model, v_model, x_model, v_model, sigmas)
    md = derivative_kernel_bilinear_means(x_model, v_model, x_data, v_data, sigmas)
    loss_per_scale = mm - 2.0 * md
    if include_data_data:
        dd = derivative_kernel_bilinear_means(x_data, v_data, x_data, v_data, sigmas)
        loss_per_scale = loss_per_scale + dd
    return loss_per_scale.mean()


def rbf_mmd2(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: Optional[torch.Tensor] = None,
    sigmas: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if sigmas is None:
        if sigma is None:
            sigma = median_bandwidth(x, y)
        sigmas = sigma.reshape(1)
    d_xx = pairwise_sq_dists(x, x)
    d_yy = pairwise_sq_dists(y, y)
    d_xy = pairwise_sq_dists(x, y)
    vals = []
    for s in sigmas.to(x.device, x.dtype).flatten():
        sigma2 = s.pow(2).clamp_min(1e-6)
        xx = torch.exp(-0.5 * d_xx / sigma2).mean()
        yy = torch.exp(-0.5 * d_yy / sigma2).mean()
        xy = torch.exp(-0.5 * d_xy / sigma2).mean()
        vals.append(xx + yy - 2.0 * xy)
    return torch.stack(vals).mean()


@torch.no_grad()
def sliced_wasserstein_np(x: np.ndarray, y: np.ndarray, n_proj: int = 128, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    dim = x.shape[1]
    dirs = rng.normal(size=(n_proj, dim))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    xp = np.sort(x @ dirs.T, axis=0)
    yp = np.sort(y @ dirs.T, axis=0)
    n = min(xp.shape[0], yp.shape[0])
    return float(np.sqrt(np.mean((xp[:n] - yp[:n]) ** 2)))


@torch.no_grad()
def energy_distance_np(x: np.ndarray, y: np.ndarray, max_points: int = 2048, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n = min(max_points, x.shape[0], y.shape[0])
    xi = x[rng.choice(x.shape[0], size=n, replace=False)] if x.shape[0] > n else x[:n]
    yi = y[rng.choice(y.shape[0], size=n, replace=False)] if y.shape[0] > n else y[:n]
    xt = torch.tensor(xi, dtype=torch.float32)
    yt = torch.tensor(yi, dtype=torch.float32)
    xy = torch.cdist(xt, yt).mean()
    xx = torch.pdist(xt).mean() if xt.shape[0] > 1 else torch.tensor(0.0)
    yy = torch.pdist(yt).mean() if yt.shape[0] > 1 else torch.tensor(0.0)
    return float((2.0 * xy - xx - yy).clamp_min(0.0).item())


@torch.no_grad()
def empirical_w2_np(x: np.ndarray, y: np.ndarray, max_points: int = 512, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n = min(max_points, x.shape[0], y.shape[0])
    xi = x[rng.choice(x.shape[0], size=n, replace=False)]
    yi = y[rng.choice(y.shape[0], size=n, replace=False)]
    cost = ((xi[:, None, :] - yi[None, :, :]) ** 2).sum(axis=2)
    if linear_sum_assignment is None:
        return sinkhorn_w2_from_cost(cost)
    row, col = linear_sum_assignment(cost)
    return float(np.sqrt(cost[row, col].mean()))


def sinkhorn_w2_from_cost(cost_np: np.ndarray, n_iter: int = 80) -> float:
    cost = torch.tensor(cost_np, dtype=torch.float32)
    n, m = cost.shape
    eps = torch.median(cost[cost > 1e-12]).item() * 0.05 if torch.any(cost > 1e-12) else 0.05
    eps = max(float(eps), 1e-3)
    f = torch.zeros(n, dtype=torch.float32)
    g = torch.zeros(m, dtype=torch.float32)
    log_a = -math.log(n)
    log_b = -math.log(m)
    for _ in range(n_iter):
        f = eps * (log_a - torch.logsumexp((g[None, :] - cost) / eps, dim=1))
        g = eps * (log_b - torch.logsumexp((f[:, None] - cost) / eps, dim=0))
    log_plan = (f[:, None] + g[None, :] - cost) / eps
    ot_cost = torch.sum(torch.exp(log_plan) * cost).clamp_min(0.0)
    return float(torch.sqrt(ot_cost).item())


@torch.no_grad()
def generate_endpoint(
    model: nn.Module,
    n: int,
    latent_dim: int,
    device: torch.device,
    batch: int = 4096,
) -> torch.Tensor:
    outs = []
    for start in range(0, n, batch):
        b = min(batch, n - start)
        u = torch.randn(b, latent_dim, device=device)
        outs.append(model(u))
    return torch.cat(outs, dim=0)


@torch.no_grad()
def sample_fm(model: TimeVelocityMLP, n: int, D: int, device: torch.device, steps: int = 20) -> torch.Tensor:
    x = torch.randn(n, D, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        tau = torch.full((n,), (i + 0.5) * dt, device=device)
        x_mid = x + 0.5 * dt * model(x, torch.full((n,), i * dt, device=device))
        x = x + dt * model(x_mid, tau)
    return x


@dataclass
class TrainConfig:
    preset: str
    width: int
    depth: int
    lr: float
    steps: int
    single_steps: int
    fm_steps: int
    fm_sample_steps: int
    batch: int
    eval_n: int
    val_n: int
    test_n: int
    tau_low: float
    tau_high: float
    all_tau_low: float
    all_tau_high: float
    local_delta: float
    lambda_marg: float
    normalize_velocity: bool
    velocity_eps: float
    tie_latent_to_base: bool
    kernel_scales: List[float]
    kernel_bandwidth_points: int
    seed: int


def tau_scalar(mode: str, center_tau: float, cfg: TrainConfig, device: torch.device) -> torch.Tensor:
    if mode == "single":
        return torch.tensor(center_tau, device=device)
    if mode == "local":
        lo = max(cfg.tau_low, center_tau - cfg.local_delta)
        hi = min(cfg.tau_high, center_tau + cfg.local_delta)
        return torch.empty((), device=device).uniform_(lo, hi)
    if mode == "all":
        return torch.empty((), device=device).uniform_(cfg.all_tau_low, cfg.all_tau_high)
    raise ValueError(mode)


def tau_vector(mode: str, center_tau: float, cfg: TrainConfig, n: int, device: torch.device) -> torch.Tensor:
    if mode == "single":
        return torch.full((n,), center_tau, device=device)
    if mode == "local":
        lo = max(cfg.tau_low, center_tau - cfg.local_delta)
        hi = min(cfg.tau_high, center_tau + cfg.local_delta)
        return torch.empty(n, device=device).uniform_(lo, hi)
    if mode == "all":
        return torch.empty(n, device=device).uniform_(cfg.all_tau_low, cfg.all_tau_high)
    raise ValueError(mode)


def normalize_velocity(v: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = torch.sqrt(v.pow(2).sum(dim=1).mean()).detach()
    return v / (scale + eps), scale


@torch.no_grad()
def compute_bridge_sigmas(
    data: ToyData,
    cfg: TrainConfig,
    mode: str,
    center_tau: float,
    train_x: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    device = data.train.device
    source = data.train if train_x is None else train_x
    n = min(max(cfg.kernel_bandwidth_points, 2), max(source.shape[0], 2))
    x1 = take_batch(source, n)
    x0 = torch.randn(n, data.D, device=device)
    tau = tau_vector(mode, center_tau, cfg, n, device).view(-1, 1)
    x_bridge = (1.0 - tau) * x0 + tau * x1
    sigma0 = median_bandwidth_from_samples(x_bridge, max_points=cfg.kernel_bandwidth_points)
    scales = torch.tensor(cfg.kernel_scales, device=device, dtype=x_bridge.dtype)
    sigmas = (sigma0 * scales).detach()
    return sigmas, float(sigma0.detach().cpu())


def train_rkhs_wbvm(
    data: ToyData,
    cfg: TrainConfig,
    mode: str,
    center_tau: float,
    train_n: Optional[int] = None,
    kernel_batch: Optional[int] = None,
    steps_override: Optional[int] = None,
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
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    ema_loss = None
    ema_model_v_rms = None
    ema_data_v_rms = None
    start = time.time()
    total_steps = steps_override or cfg.steps
    train_tau_low = cfg.all_tau_low if mode == "all" else (max(cfg.tau_low, center_tau - cfg.local_delta) if mode == "local" else center_tau)
    train_tau_high = cfg.all_tau_high if mode == "all" else (min(cfg.tau_high, center_tau + cfg.local_delta) if mode == "local" else center_tau)
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
            v_model_for_loss, v_model_rms = v_model, torch.sqrt(v_model.pow(2).sum(dim=1).mean()).detach()
            v_data_for_loss, v_data_rms = v_data, torch.sqrt(v_data.pow(2).sum(dim=1).mean()).detach()
        flux = rkhs_flux_u_stat(
            x_model,
            v_model_for_loss,
            x_data,
            v_data_for_loss,
            sigmas=kernel_sigmas,
            include_data_data=False,
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
            print(f"{data.name:10s} {mode:6s} tau={center_tau:.2f} step={step:5d} loss={ema_loss:.5g}", flush=True)
    return model, {
        "train_seconds": time.time() - start,
        "last_loss": float(ema_loss or 0.0),
        "train_steps": total_steps,
        "latent_dim": latent_dim,
        "time_conditioned": False,
        "loss_statistic": "u_theta_terms",
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
def endpoint_validation_metric(
    raw_samples: np.ndarray,
    val_raw: np.ndarray,
    metric: str,
    seed: int,
) -> float:
    if not np.all(np.isfinite(raw_samples)):
        return float("inf")
    max_abs = float(np.max(np.abs(raw_samples))) if raw_samples.size else 0.0
    if max_abs > 1e5:
        return float("inf")
    n = min(raw_samples.shape[0], val_raw.shape[0])
    if metric == "w2":
        return empirical_w2_np(raw_samples, val_raw, max_points=512, seed=seed)
    if metric == "sliced_w2":
        return sliced_wasserstein_np(raw_samples[:n], val_raw[:n], seed=seed)
    if metric == "energy":
        return energy_distance_np(raw_samples, val_raw, seed=seed)
    raise ValueError(f"Unknown validation metric {metric}")


@torch.no_grad()
def wbvm_single_validation_score(
    model: nn.Module,
    data: ToyData,
    cfg: TrainConfig,
    tau: float,
    val_metric: str,
    seed: int,
) -> float:
    if val_metric == "flux":
        batch = min(1024, cfg.val_n, data.val.shape[0])
        return heldout_flux_residual(model, data, cfg, tau, batch=batch, split="val")
    raw = as_raw_samples(model, data, cfg.val_n)
    return endpoint_validation_metric(raw, data.val_raw, val_metric, seed)


def train_validation_selected_single(
    data: ToyData,
    cfg: TrainConfig,
    tau_candidates: List[float],
    val_metric: str,
    dataset_seed: int,
    verbose: bool = False,
) -> Tuple[nn.Module, Dict[str, float], List[Dict[str, object]]]:
    best_model: Optional[nn.Module] = None
    best_info: Dict[str, float] = {}
    best_metric = float("inf")
    best_tau = float("nan")
    rows: List[Dict[str, object]] = []
    device = data.train.device

    for tau in tau_candidates:
        set_seed(dataset_seed + int(round(1000 * tau)))
        model, info = train_rkhs_wbvm(
            data,
            cfg,
            mode="single",
            center_tau=tau,
            steps_override=cfg.single_steps,
            verbose=verbose,
        )
        val_score = wbvm_single_validation_score(
            model,
            data,
            cfg,
            tau,
            val_metric,
            dataset_seed + int(round(1000 * tau)),
        )
        row = {
            "dataset": data.name,
            "tau": tau,
            "val_metric": val_metric,
            "val_score": val_score,
            **info,
        }
        rows.append(row)
        print(f"{data.name:10s} single-val tau={tau:.2f} {val_metric}={val_score:.4f}", flush=True)

        if val_score < best_metric:
            if best_model is not None:
                del best_model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            best_model = model
            best_metric = val_score
            best_tau = tau
            best_info = dict(info)
        else:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if best_model is None:
        raise RuntimeError(f"No finite validation-selected WBVM-single candidate for {data.name}")

    best_info.update({"selected_tau": best_tau, "validation_metric": best_metric})
    return best_model, best_info, rows


def train_fm(data: ToyData, cfg: TrainConfig, verbose: bool = False) -> Tuple[TimeVelocityMLP, Dict[str, float]]:
    device = data.train.device
    model = TimeVelocityMLP(data.D, cfg.width, cfg.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
    ema_loss = None
    start = time.time()
    for step in range(1, cfg.fm_steps + 1):
        x1 = take_batch(data.train, cfg.batch)
        x0 = torch.randn(cfg.batch, data.D, device=device)
        tau = cfg.tau_low + (cfg.tau_high - cfg.tau_low) * torch.rand(cfg.batch, device=device)
        xt = (1.0 - tau[:, None]) * x0 + tau[:, None] * x1
        target = x1 - x0
        pred = model(xt, tau)
        loss = F.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        loss_value = float(loss.detach().cpu())
        ema_loss = loss_value if ema_loss is None else 0.98 * ema_loss + 0.02 * loss_value
        if verbose and (step == 1 or step % max(cfg.fm_steps // 3, 1) == 0):
            print(f"{data.name:10s} FM     step={step:5d} loss={ema_loss:.5g}", flush=True)
    return model, {"train_seconds": time.time() - start, "last_loss": float(ema_loss or 0.0)}


@torch.no_grad()
def evaluate_generator(
    model: nn.Module,
    data: ToyData,
    cfg: TrainConfig,
    seed: int,
    exact_w2_points: int = 512,
) -> Dict[str, float]:
    gen = generate_endpoint(model, cfg.eval_n, data.D, data.train.device).detach()
    gen_raw = data.inverse(gen).cpu().numpy()
    n = min(cfg.eval_n, data.test_raw.shape[0])
    test_raw = data.test_raw[:n]
    exact_w2 = float("nan") if exact_w2_points <= 0 else empirical_w2_np(gen_raw, data.test_raw, exact_w2_points, seed=seed)
    return {
        "sliced_w2": sliced_wasserstein_np(gen_raw[:n], test_raw, seed=seed),
        "exact_w2_subset": exact_w2,
    }


@torch.no_grad()
def evaluate_fm(model: TimeVelocityMLP, data: ToyData, cfg: TrainConfig, seed: int) -> Dict[str, float]:
    gen = sample_fm(model, cfg.eval_n, data.D, data.train.device, steps=cfg.fm_sample_steps).detach()
    gen_raw = data.inverse(gen).cpu().numpy()
    n = min(cfg.eval_n, data.test_raw.shape[0])
    return {
        "sliced_w2": sliced_wasserstein_np(gen_raw[:n], data.test_raw[:n], seed=seed),
        "exact_w2_subset": empirical_w2_np(gen_raw, data.test_raw, 512, seed=seed),
    }


@torch.no_grad()
def heldout_flux_residual(
    model: nn.Module,
    data: ToyData,
    cfg: TrainConfig,
    tau: float,
    batch: int = 512,
    split: str = "test",
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
    return float(rkhs_flux_u_stat(x_model, v_model, x_data, v_data, sigmas=sigmas, include_data_data=True).detach().cpu())


def as_raw_samples(model: nn.Module, data: ToyData, n: int) -> np.ndarray:
    with torch.no_grad():
        x = generate_endpoint(model, n, data.D, data.train.device)
        return data.inverse(x).cpu().numpy()


def as_raw_fm(model: TimeVelocityMLP, data: ToyData, n: int, steps: int = 20) -> np.ndarray:
    with torch.no_grad():
        x = sample_fm(model, n, data.D, data.train.device, steps=steps)
        return data.inverse(x).cpu().numpy()


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def timed_endpoint_samples(
    model: nn.Module,
    data: ToyData,
    n: int,
) -> Tuple[np.ndarray, float]:
    device = data.train.device
    sync_device(device)
    start = time.perf_counter()
    x = generate_endpoint(model, n, data.D, device)
    sync_device(device)
    elapsed = time.perf_counter() - start
    return data.inverse(x).cpu().numpy(), elapsed


@torch.no_grad()
def timed_fm_samples(model: TimeVelocityMLP, data: ToyData, n: int, steps: int) -> Tuple[np.ndarray, float]:
    device = data.train.device
    sync_device(device)
    start = time.perf_counter()
    x = sample_fm(model, n, data.D, device, steps=steps)
    sync_device(device)
    elapsed = time.perf_counter() - start
    return data.inverse(x).cpu().numpy(), elapsed


def table1_metrics(
    dataset: str,
    method: str,
    raw_samples: np.ndarray,
    test_raw: np.ndarray,
    sample_seconds: float,
    nfe: int,
    seed: int,
    tau_setting: str,
    threshold: float = 0.1,
) -> Dict[str, object]:
    n = min(raw_samples.shape[0], test_raw.shape[0])
    distances = manifold_distance_np(dataset, raw_samples)
    return {
        "dataset": dataset,
        "method": method,
        "tau_setting": tau_setting,
        "nfe": nfe,
        "w2": empirical_w2_np(raw_samples, test_raw, max_points=512, seed=seed),
        "sliced_w2": sliced_wasserstein_np(raw_samples[:n], test_raw[:n], seed=seed),
        "off_manifold_rate": float(np.mean(distances > threshold)),
        "off_threshold": threshold,
        "sample_seconds_10k": sample_seconds * (10000.0 / raw_samples.shape[0]),
        "sample_count": raw_samples.shape[0],
    }


def write_table1_artifacts(outdir: Path, rows: List[Dict[str, object]]) -> None:
    csv_path = outdir / "table1_metrics.csv"
    fieldnames = ["dataset", "method", "tau_setting", "nfe", "w2", "sliced_w2", "off_manifold_rate", "off_threshold", "sample_seconds_10k", "sample_count"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "| Dataset | Method | tau/range | NFE | W2 | Sliced W2 | % out | Time / 10k (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            "| {dataset} | {method} | {tau} | {nfe} | {w2:.4f} | {sliced_w2:.4f} | {out:.4f} | {time:.4f} |".format(
                dataset=row["dataset"],
                method=row["method"],
                tau=row["tau_setting"],
                nfe=int(row["nfe"]),
                w2=float(row["w2"]),
                sliced_w2=float(row["sliced_w2"]),
                out=100.0 * float(row["off_manifold_rate"]),
                time=float(row["sample_seconds_10k"]),
            )
        )
    (outdir / "table1_metrics.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    tex_lines = [
        "\\begin{tabular}{lllrrrr}",
        "\\toprule",
        "Dataset & Method & $\\tau$ & NFE & $W_2$ & \\% out & Time/10k (s) \\\\",
        "\\midrule",
    ]
    for row in rows:
        tex_lines.append(
            "{dataset} & {method} & {tau} & {nfe:d} & {w2:.4f} & {out:.2f} & {time:.4f} \\\\".format(
                dataset=row["dataset"],
                method=row["method"],
                tau=row["tau_setting"].replace("[", "$[").replace("]", "]$"),
                nfe=int(row["nfe"]),
                w2=float(row["w2"]),
                out=100.0 * float(row["off_manifold_rate"]),
                time=float(row["sample_seconds_10k"]),
            )
        )
    tex_lines += ["\\bottomrule", "\\end{tabular}", ""]
    (outdir / "table1_metrics.tex").write_text("\n".join(tex_lines), encoding="utf-8")

    render_table1_png(outdir / "table1_metrics.png", rows)


def render_table1_png(outpath: Path, rows: List[Dict[str, object]]) -> None:
    columns = ["Dataset", "Method", "tau/range", "NFE", "W2", "Sliced W2", "% out", "Time/10k"]
    cell_text = []
    for row in rows:
        cell_text.append(
            [
                str(row["dataset"]),
                str(row["method"]),
                str(row["tau_setting"]),
                str(int(row["nfe"])),
                f"{float(row['w2']):.4f}",
                f"{float(row['sliced_w2']):.4f}",
                f"{100.0 * float(row['off_manifold_rate']):.2f}",
                f"{float(row['sample_seconds_10k']):.4f}",
            ]
        )

    fig_height = max(4.5, 0.42 * (len(rows) + 2))
    fig, ax = plt.subplots(figsize=(13.5, fig_height))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)

    header_color = "#23395d"
    even_color = "#f5f7fb"
    odd_color = "#ffffff"
    method_colors = {
        "WBVM-all": "#e8f2ff",
        "WBVM-single": "#fff4e1",
        "Flow matching": "#eaf7ed",
    }
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d5dd")
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold")
        else:
            method = rows[r - 1]["method"]
            cell.set_facecolor(method_colors.get(method, even_color) if c == 1 else (even_color if r % 2 == 0 else odd_color))
    ax.set_title("Table 1-style Synthetic Metrics (10,000 Samples)", fontsize=15, weight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_samples_panel(
    outpath: Path,
    data_map: Dict[str, ToyData],
    sample_map: Dict[Tuple[str, str], np.ndarray],
    datasets: List[str],
    n_plot: int = 1500,
) -> None:
    cols = ["Ground truth", "WBVM-all", "WBVM-single", "Flow matching"]
    fig = plt.figure(figsize=(14, max(4, 3 * len(datasets))))
    for r, name in enumerate(datasets):
        data = data_map[name]
        arrays = [
            data.test_raw[:n_plot],
            sample_map[(name, "all")][:n_plot],
            sample_map[(name, "single")][:n_plot],
            sample_map[(name, "fm")][:n_plot],
        ]
        for c, arr in enumerate(arrays):
            ax = fig.add_subplot(len(datasets), len(cols), r * len(cols) + c + 1, projection="3d" if data.D == 3 else None)
            if data.D == 2:
                ax.scatter(arr[:, 0], arr[:, 1], s=2, alpha=0.65, linewidths=0)
                ax.set_aspect("equal", adjustable="box")
            else:
                ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], s=2, alpha=0.55, linewidths=0)
                ax.view_init(elev=22, azim=-55)
                try:
                    ax.set_box_aspect((1, 1, 0.8))
                except Exception:
                    pass
            ax.set_xticks([])
            ax.set_yticks([])
            if data.D == 3:
                ax.set_zticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=11)
            if c == 0:
                ax.set_ylabel(name, fontsize=11)
            ax.grid(False)
    fig.suptitle("RKHS-WBVM synthetic bridge benchmark", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def plot_knt(outpath: Path, rows: List[Dict[str, object]], datasets: List[str]) -> None:
    by_dataset: Dict[str, List[Dict[str, object]]] = {name: [] for name in datasets}
    for row in rows:
        by_dataset[str(row["dataset"])].append(row)

    factors = ["n", "K", "tau"]
    fig, axes = plt.subplots(len(datasets), len(factors), figsize=(17, max(4, 3.5 * len(datasets))))
    if len(datasets) == 1:
        axes = axes[None, :]
    colors = {"WBVM-single": "#1f77b4", "WBVM-local": "#ff7f0e"}
    for i, name in enumerate(datasets):
        for j, factor in enumerate(factors):
            ax = axes[i, j]
            sub = [r for r in by_dataset[name] if r["factor"] == factor]
            for method in ["WBVM-single", "WBVM-local"]:
                ms = [r for r in sub if r["method"] == method]
                if not ms:
                    continue
                xs = np.array([float(r["value"]) for r in ms])
                ys = np.array([float(r["mean"]) for r in ms])
                es = np.array([float(r["std"]) for r in ms])
                order = np.argsort(xs)
                ax.errorbar(xs[order], ys[order], yerr=es[order], marker="o", linewidth=1.5, capsize=3, label=method, color=colors[method])
            if factor in {"n", "K"}:
                ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.25, linestyle="--")
            ax.set_ylabel("Sliced W2 (mean +- std)")
            if factor == "n":
                ax.set_title(f"{name} -- vs n @ K, tau=0.40")
            elif factor == "K":
                ax.set_title(f"{name} -- vs K @ n, tau=0.40")
            else:
                ax.set_title(f"{name} -- vs tau @ n, K")
            if i == 0 and j == 2:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def preset_config(preset: str, seed: int) -> TrainConfig:
    if preset == "smoke":
        return TrainConfig(
            preset=preset,
            width=64,
            depth=2,
            lr=2e-3,
            steps=80,
            single_steps=40,
            fm_steps=80,
            fm_sample_steps=20,
            batch=64,
            eval_n=512,
            val_n=512,
            test_n=1024,
            tau_low=0.05,
            tau_high=0.95,
            all_tau_low=0.35,
            all_tau_high=0.90,
            local_delta=0.08,
            lambda_marg=0.0,
            normalize_velocity=True,
            velocity_eps=1e-6,
            tie_latent_to_base=False,
            kernel_scales=[0.5, 1.0, 2.0, 4.0],
            kernel_bandwidth_points=1024,
            seed=seed,
        )
    if preset == "quick":
        return TrainConfig(
            preset=preset,
            width=128,
            depth=3,
            lr=1.5e-3,
            steps=300,
            single_steps=120,
            fm_steps=350,
            fm_sample_steps=20,
            batch=128,
            eval_n=1500,
            val_n=1500,
            test_n=2500,
            tau_low=0.05,
            tau_high=0.95,
            all_tau_low=0.35,
            all_tau_high=0.90,
            local_delta=0.08,
            lambda_marg=0.0,
            normalize_velocity=True,
            velocity_eps=1e-6,
            tie_latent_to_base=False,
            kernel_scales=[0.5, 1.0, 2.0, 4.0],
            kernel_bandwidth_points=2048,
            seed=seed,
        )
    if preset == "standard":
        return TrainConfig(
            preset=preset,
            width=512,
            depth=5,
            lr=8e-4,
            steps=8000,
            single_steps=4000,
            fm_steps=8000,
            fm_sample_steps=20,
            batch=1024,
            eval_n=10000,
            val_n=5000,
            test_n=10000,
            tau_low=0.05,
            tau_high=0.95,
            all_tau_low=0.35,
            all_tau_high=0.90,
            local_delta=0.08,
            lambda_marg=0.0,
            normalize_velocity=True,
            velocity_eps=1e-6,
            tie_latent_to_base=False,
            kernel_scales=[0.5, 1.0, 2.0, 4.0],
            kernel_bandwidth_points=4096,
            seed=seed,
        )
    raise ValueError(f"Unknown preset {preset}")


def aggregate(values: List[float]) -> Tuple[float, float]:
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def parse_float_list(text: str) -> List[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one comma-separated float")
    return values


def run_main(args: argparse.Namespace) -> None:
    cfg = preset_config(args.preset, args.seed)
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

    runtime = configure_runtime(args.num_threads)
    set_seed(cfg.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "device": str(device), "runtime": runtime, "args": vars(args)}, f, indent=2)

    print(f"Running RKHS-WBVM synthetic experiments on {device} with preset={cfg.preset}", flush=True)
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
        print(f"\n[{i + 1}/{len(selected_datasets)}] Qualitative models for {name}", flush=True)
        all_model, all_info = train_rkhs_wbvm(data, cfg, mode="all", center_tau=0.4, verbose=args.verbose)
        single_model, single_info, selection_rows = train_validation_selected_single(
            data,
            cfg,
            tau_candidates=tau_candidates,
            val_metric=args.val_metric,
            dataset_seed=cfg.seed + 1000 * i,
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
            flux_resid = heldout_flux_residual(model, data, cfg, 0.4, batch=min(512, cfg.eval_n))
            metrics.append({"dataset": name, "method": label, "factor": "main", "value": 0.0, **ev, "heldout_flux_tau_0.4": flux_resid, **info})
            print(f"{name:10s} {label:12s} SW2={ev['sliced_w2']:.4f} W2={ev['w2']:.4f} out={100*ev['off_manifold_rate']:.2f}%", flush=True)
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
        print(f"{name:10s} {'Flow matching':12s} SW2={ev_fm['sliced_w2']:.4f} W2={ev_fm['w2']:.4f} out={100*ev_fm['off_manifold_rate']:.2f}%", flush=True)
        samples[(name, "all")] = raw_all
        samples[(name, "single")] = raw_single
        samples[(name, "fm")] = raw_fm

    plot_samples_panel(outdir / "rkhs_wbvm_toy_6x4.png", data_map, samples, selected_datasets)
    write_table1_artifacts(outdir, table_rows)
    if single_selection_rows:
        with open(outdir / "wbvm_single_validation_selection.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(single_selection_rows[0].keys()))
            writer.writeheader()
            writer.writerows(single_selection_rows)

    if not args.skip_knt:
        print("\nRunning n/K/tau sensitivity grid", flush=True)
        main_steps = cfg.steps
        if args.knt_steps is not None:
            cfg.steps = args.knt_steps
        if cfg.preset == "smoke":
            n_values = [512, 1024]
            k_values = [48, 96]
            tau_values = [0.2, 0.5, 0.8]
            seeds = [cfg.seed]
        elif cfg.preset == "quick":
            n_values = [1000, 2500, 5000]
            k_values = [64, 128, 256]
            tau_values = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]
            seeds = [cfg.seed]
        else:
            n_values = [2500, 5000, 10000, 20000]
            k_values = [96, 192, 384, 768]
            tau_values = [0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
            seeds = [cfg.seed, cfg.seed + 17]
        knt_rows: List[Dict[str, object]] = []
        for di, name in enumerate(selected_datasets):
            data = data_map[name]
            base_n = min(args.train_n, max(n_values))
            base_k = min(cfg.batch, max(k_values))
            tasks: List[Tuple[str, float, int, int]] = []
            tasks += [("n", float(n), int(n), base_k) for n in n_values]
            tasks += [("K", float(k), base_n, int(k)) for k in k_values]
            tasks += [("tau", float(t), base_n, base_k) for t in tau_values]
            for factor, value, train_n, kernel_batch in tasks:
                for method, mode in [("WBVM-single", "single"), ("WBVM-local", "local")]:
                    scores = []
                    for s in seeds:
                        set_seed(s + 10000 * di + int(value * 1000))
                        center_tau = float(value) if factor == "tau" else 0.4
                        model, info = train_rkhs_wbvm(
                            data,
                            cfg,
                            mode=mode,
                            center_tau=center_tau,
                            train_n=min(train_n, data.train.shape[0]),
                            kernel_batch=min(kernel_batch, data.train.shape[0]),
                            verbose=False,
                        )
                        ev = evaluate_generator(model, data, cfg, s + di, exact_w2_points=0)
                        scores.append(ev["sliced_w2"])
                    mean, std = aggregate(scores)
                    row = {
                        "dataset": name,
                        "factor": factor,
                        "value": value,
                        "method": method,
                        "mean": mean,
                        "std": std,
                        "train_n": train_n,
                        "kernel_batch_K": kernel_batch,
                    }
                    knt_rows.append(row)
                    print(
                        f"{name:10s} {factor:3s}={value:<7g} {method:11s} SW2={mean:.4f} +- {std:.4f}",
                        flush=True,
                    )
        plot_knt(outdir / "rkhs_wbvm_knt.png", knt_rows, selected_datasets)
        with open(outdir / "knt_results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(knt_rows[0].keys()))
            writer.writeheader()
            writer.writerows(knt_rows)
        cfg.steps = main_steps

    metric_path = outdir / "metrics_summary.csv"
    metric_keys = sorted({k for row in metrics for k in row.keys()})
    with open(metric_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metric_keys)
        writer.writeheader()
        writer.writerows(metrics)

    print(f"\nDone. Outputs written to {outdir.resolve()}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RKHS derivative-kernel WBVM synthetic experiments")
    parser.add_argument("--preset", choices=["smoke", "quick", "standard"], default="quick")
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default=None)
    parser.add_argument("--train-n", type=int, default=20000)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--single-steps", type=int, default=None)
    parser.add_argument("--fm-steps", type=int, default=None)
    parser.add_argument("--fm-sample-steps", type=int, default=None)
    parser.add_argument("--knt-steps", type=int, default=None)
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
    parser.add_argument("--table-samples", type=int, default=10000)
    parser.add_argument("--single-taus", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--val-metric", choices=["flux", "sliced_w2", "w2", "energy"], default="flux")
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--skip-knt", action="store_true")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--verbose", action="store_true")
    return parser


if __name__ == "__main__":
    run_main(build_parser().parse_args())
