import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import trange


IMAGE_SIZE = 32
EVAL_IMAGE_SIZE = 28
IMAGE_SHAPE = (1, IMAGE_SIZE, IMAGE_SIZE)
IMAGE_DIM = IMAGE_SIZE * IMAGE_SIZE
LATENT_SHAPE = (16, 4, 4)
LATENT_DIM = 16 * 4 * 4


@dataclass
class MnistConfig:
    preset: str
    model_space: str
    seed: int
    train_n: int
    val_n: int
    fid_samples: int
    steps: int
    single_steps: int
    batch_size: int
    kernel_batch: int
    lr: float
    weight_decay: float
    grad_clip: float
    hidden: int
    num_workers: int
    all_tau_low: float
    all_tau_high: float
    wbvm_single_taus: List[float]
    kernel_scales: List[float]
    kernel_bandwidth_points: int
    vector_loss_statistic: str
    velocity_eps: float
    meanflow_gap_prob: float
    shortcut_empirical_frac: float
    shortcut_min_steps: int
    shortcut_ema_decay: float
    drifting_step_size: float
    drifting_temperatures: List[float]
    drifting_normalize_distances: bool
    drifting_normalize_drift: bool
    drifting_space: str
    fid_batch_size: int
    fid_backend: str
    selection_metric: str
    extra_mnist_metrics: bool
    kid_subset_size: int
    kid_subsets: int
    classifier_epochs: int
    classifier_lr: float
    codec_epochs: int
    codec_lr: float
    latent_stats_samples: int
    latent_hidden: int
    latent_depth: int
    direct_backbone: str
    pixel_dit_hidden: int
    pixel_dit_depth: int
    pixel_dit_heads: int
    pixel_patch_size: int
    pixel_dit_mlp_ratio: float
    pixel_dit_register_tokens: int
    pixel_dit_use_qk_norm: bool
    pixel_dit_use_style_embed: bool
    pixel_dit_style_tokens: int
    pixel_dit_style_codebook: int
    meanflow_data_proportion: float
    meanflow_logit_mean: float
    meanflow_logit_std: float
    meanflow_norm_p: float
    meanflow_norm_eps: float
    shortcut_bootstrap_every: int
    early_stop_min_steps: int
    early_stop_patience: int
    early_stop_min_delta: float
    early_stop_metric: str
    methods: List[str]


class InfiniteLoader:
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.iterator = iter(loader)

    def next(self) -> torch.Tensor:
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            batch = next(self.iterator)
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        return x


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime(num_threads: Optional[int]) -> Dict[str, object]:
    cpu_count = os.cpu_count() or 1
    threads = int(num_threads or min(20, cpu_count))
    threads = max(1, min(threads, cpu_count))
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
        "tf32": bool(torch.backends.cuda.matmul.allow_tf32) if torch.cuda.is_available() else False,
    }


def to_model_space(x: torch.Tensor) -> torch.Tensor:
    return x.mul(2.0).sub(1.0)


def from_model_space(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(-1.0, 1.0).add(1.0).mul(0.5)


def resize_for_eval(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == (EVAL_IMAGE_SIZE, EVAL_IMAGE_SIZE):
        return x
    return F.interpolate(x, size=(EVAL_IMAGE_SIZE, EVAL_IMAGE_SIZE), mode="bilinear", align_corners=False)


def flatten_img(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


def cycle_batch(loader: InfiniteLoader, device: torch.device) -> torch.Tensor:
    return to_model_space(loader.next().to(device, non_blocking=True))


def cycle_space_batch(
    loader: InfiniteLoader,
    cfg: MnistConfig,
    device: torch.device,
    codec: Optional["LeNetLatentCodec"],
) -> torch.Tensor:
    x_model = cycle_batch(loader, device)
    if cfg.model_space == "pixel":
        return x_model
    if codec is None:
        raise ValueError("model_space=latent requires a trained latent codec")
    return codec.encode_model(x_model)


def decode_space_batch(
    x: torch.Tensor,
    cfg: MnistConfig,
    codec: Optional["LeNetLatentCodec"],
) -> torch.Tensor:
    if cfg.model_space == "pixel":
        return x.clamp(-1.0, 1.0)
    if codec is None:
        raise ValueError("model_space=latent requires a trained latent codec")
    return codec.decode_model(x)


def make_loaders(cfg: MnistConfig, data_dir: Path) -> Tuple[DataLoader, DataLoader, DataLoader]:
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.Resize(IMAGE_SIZE), transforms.ToTensor()])
    train_full = datasets.MNIST(str(data_dir), train=True, transform=transform, download=True)
    test = datasets.MNIST(str(data_dir), train=False, transform=transform, download=True)

    val_n = min(cfg.val_n, len(train_full) // 6)
    train_n = min(cfg.train_n, len(train_full) - val_n)
    generator = torch.Generator().manual_seed(cfg.seed)
    train_part, val_part, _ = random_split(
        train_full,
        [train_n, val_n, len(train_full) - train_n - val_n],
        generator=generator,
    )
    test_n = min(cfg.fid_samples, len(test))
    test_part = Subset(test, list(range(test_n)))

    loader_args = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": True,
    }
    train_loader = DataLoader(train_part, shuffle=True, **loader_args)
    val_loader = DataLoader(val_part, shuffle=False, batch_size=cfg.fid_batch_size, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_part, shuffle=False, batch_size=cfg.fid_batch_size, num_workers=cfg.num_workers)
    return train_loader, val_loader, test_loader


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = max(1, min(8, channels // 8))
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CondConvNet(nn.Module):
    def __init__(self, cond_dim: int, hidden: int, out_tanh: bool):
        super().__init__()
        self.cond_dim = cond_dim
        self.out_tanh = out_tanh
        self.in_conv = nn.Conv2d(1 + cond_dim, hidden, 3, padding=1)
        self.down1 = nn.Conv2d(hidden, hidden * 2, 4, stride=2, padding=1)
        self.down2 = nn.Conv2d(hidden * 2, hidden * 4, 4, stride=2, padding=1)
        self.mid = nn.Sequential(ResBlock(hidden * 4), ResBlock(hidden * 4))
        self.up1 = nn.ConvTranspose2d(hidden * 4, hidden * 2, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(hidden * 2, hidden, 4, stride=2, padding=1)
        self.out = nn.Sequential(
            ResBlock(hidden),
            nn.GroupNorm(max(1, min(8, hidden // 8)), hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.cond_dim:
            if cond is None:
                raise ValueError("Missing conditioning tensor")
            cond_map = cond.to(x.dtype).view(cond.shape[0], self.cond_dim, 1, 1).expand(-1, -1, x.shape[-2], x.shape[-1])
            x = torch.cat([x, cond_map], dim=1)
        h = self.in_conv(x)
        h = F.silu(h)
        h = F.silu(self.down1(h))
        h = F.silu(self.down2(h))
        h = self.mid(h)
        h = F.silu(self.up1(h))
        h = F.silu(self.up2(h))
        y = self.out(h)
        return torch.tanh(y) if self.out_tanh else y


class DirectGenerator(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = CondConvNet(cond_dim=0, hidden=hidden, out_tanh=True)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.net(u)


class VelocityNet(nn.Module):
    def __init__(self, cond_dim: int, hidden: int):
        super().__init__()
        self.net = CondConvNet(cond_dim=cond_dim, hidden=hidden, out_tanh=False)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(x, cond)


class LeNet5FeatureNet(nn.Module):
    """Classic LeNet-5-style MNIST classifier and feature extractor."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(LATENT_DIM, 120)
        self.fc2 = nn.Linear(120, 84)
        self.head = nn.Linear(84, 10)

    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        x = resize_for_eval(x)
        x = F.avg_pool2d(torch.tanh(self.conv1(x)), 2)
        x = F.avg_pool2d(torch.tanh(self.conv2(x)), 2)
        return x

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        latent = self.encode_latent(x).flatten(1)
        hidden = torch.tanh(self.fc1(latent))
        features = torch.tanh(self.fc2(hidden))
        if return_features:
            return features
        return self.head(features)


MnistFeatureNet = LeNet5FeatureNet


class LeNet5Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv2 = nn.ConvTranspose2d(16, 6, 5)
        self.deconv1 = nn.ConvTranspose2d(6, 1, 5)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(latent, scale_factor=2.0, mode="nearest")
        x = torch.tanh(self.deconv2(x))
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return torch.sigmoid(self.deconv1(x))


class LeNetLatentCodec(nn.Module):
    def __init__(self, encoder: LeNet5FeatureNet, decoder: LeNet5Decoder, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.register_buffer("latent_mean", mean.reshape(1, *LATENT_SHAPE))
        self.register_buffer("latent_std", std.reshape(1, *LATENT_SHAPE).clamp_min(1e-4))
        self.encoder.eval()
        self.decoder.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def encode_model(self, x_model: torch.Tensor) -> torch.Tensor:
        latent = self.encoder.encode_latent(from_model_space(x_model))
        return (latent - self.latent_mean) / self.latent_std

    @torch.no_grad()
    def decode_model(self, latent: torch.Tensor) -> torch.Tensor:
        raw = latent * self.latent_std + self.latent_mean
        return to_model_space(self.decoder(raw)).clamp(-1.0, 1.0)


class LatentResidualBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.net = nn.Sequential(nn.Linear(hidden, hidden * 2), nn.SiLU(), nn.Linear(hidden * 2, hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class LatentGenerator(nn.Module):
    def __init__(self, hidden: int, depth: int):
        super().__init__()
        self.input = nn.Linear(LATENT_DIM, hidden)
        self.blocks = nn.Sequential(*[LatentResidualBlock(hidden) for _ in range(depth)])
        self.output = nn.Linear(hidden, LATENT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        h = self.blocks(F.silu(self.input(x.reshape(x.shape[0], -1))))
        return self.output(h).reshape(shape)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    t = t.reshape(-1).float()
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half, 1))
    args = t[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 1024, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])
        self.max_seq_len = seq_len

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]


class SwiGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(hidden_features, out_features, bias=False)
        self.w3 = nn.Linear(in_features, hidden_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DriftAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = False, use_qk_norm: bool = True):
        super().__init__()
        if dim % num_heads:
            raise ValueError("hidden size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q = (q * rope_cos) + (rotate_half(q) * rope_sin)
        k = (k * rope_cos) + (rotate_half(k) * rope_sin)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(batch, tokens, channels))


def modulate_tokens(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class DriftDiTBlock(nn.Module):
    def __init__(self, hidden: int, num_heads: int, mlp_ratio: float = 4.0, use_qk_norm: bool = True):
        super().__init__()
        self.norm1 = RMSNorm(hidden)
        self.attn = DriftAttention(hidden, num_heads=num_heads, use_qk_norm=use_qk_norm)
        self.norm2 = RMSNorm(hidden)
        self.mlp = SwiGLU(hidden, int(hidden * mlp_ratio), hidden)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6 * hidden))

    def forward(self, x: torch.Tensor, cond: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN_modulation(cond).chunk(6, dim=-1)
        attn = self.attn(modulate_tokens(self.norm1(x), shift_a, scale_a), rope_cos, rope_sin)
        x = x + gate_a[:, None, :] * attn
        mlp_input = modulate_tokens(self.norm2(x), shift_m, scale_m)
        return x + gate_m[:, None, :] * self.mlp(mlp_input)


class DriftDiTFinalLayer(nn.Module):
    def __init__(self, hidden: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = RMSNorm(hidden)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        self.linear = nn.Linear(hidden, patch_size * patch_size * out_channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
        return self.linear(modulate_tokens(self.norm(x), shift, scale))


class StyleEmbedder(nn.Module):
    def __init__(self, hidden_size: int, num_tokens: int = 32, codebook_size: int = 64):
        super().__init__()
        self.num_tokens = num_tokens
        self.codebook_size = codebook_size
        self.codebook = nn.Embedding(codebook_size, hidden_size)

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        indices = torch.randint(0, self.codebook_size, (batch_size, self.num_tokens), device=device)
        return self.codebook(indices).sum(dim=1)


class DriftConditioner(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        use_style_embed: bool,
        style_tokens: int,
        style_codebook: int,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.step_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.use_style_embed = use_style_embed
        if use_style_embed:
            self.style_embed = StyleEmbedder(hidden_size, num_tokens=style_tokens, codebook_size=style_codebook)

    def forward(self, t: torch.Tensor, step: torch.Tensor, batch_size: int, device: torch.device, hidden_size: int) -> torch.Tensor:
        cond = self.time_mlp(timestep_embedding(t, hidden_size))
        cond = cond + self.step_mlp(timestep_embedding(step, hidden_size))
        if self.use_style_embed:
            cond = cond + self.style_embed(batch_size, device)
        return cond


class PixelDiT(nn.Module):
    """DriftDiT-Tiny-style backbone adapted to the local direct/velocity interfaces."""

    def __init__(
        self,
        hidden_size: int,
        depth: int,
        num_heads: int,
        patch_size: int,
        dual_head: bool,
        image_size: int = IMAGE_SIZE,
        in_channels: int = 1,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 8,
        use_qk_norm: bool = True,
        use_style_embed: bool = True,
        style_tokens: int = 32,
        style_codebook: int = 64,
    ):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("patch size must divide the spatial input size")
        self.patch_size = patch_size
        self.image_size = image_size
        self.in_channels = in_channels
        self.grid_size = image_size // patch_size
        self.dual_head = dual_head
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_register_tokens = num_register_tokens
        self.use_qk_norm = use_qk_norm
        self.use_style_embed = use_style_embed
        self.style_tokens = style_tokens
        self.style_codebook = style_codebook
        self.patch_embed = nn.Conv2d(in_channels, hidden_size, patch_size, stride=patch_size)
        self.register_tokens = nn.Parameter(torch.randn(1, num_register_tokens, hidden_size) * 0.02)
        self.rope = RotaryPositionEmbedding(
            dim=hidden_size // num_heads,
            max_seq_len=(self.grid_size * self.grid_size) + num_register_tokens + 64,
        )
        self.conditioner = DriftConditioner(
            hidden_size=hidden_size,
            use_style_embed=use_style_embed,
            style_tokens=style_tokens,
            style_codebook=style_codebook,
        )
        self.blocks = nn.ModuleList(
            [
                DriftDiTBlock(
                    hidden=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_qk_norm=use_qk_norm,
                )
                for _ in range(depth)
            ]
        )
        self.u_final = DriftDiTFinalLayer(hidden_size, patch_size, in_channels)
        self.v_final = DriftDiTFinalLayer(hidden_size, patch_size, in_channels) if dual_head else None
        self._init_weights()

    def _init_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

        self.apply(_basic_init)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        final_layers = [self.u_final]
        if self.v_final is not None:
            final_layers.append(self.v_final)
        for final_layer in final_layers:
            nn.init.zeros_(final_layer.adaLN_modulation[-1].weight)
            nn.init.zeros_(final_layer.adaLN_modulation[-1].bias)
            nn.init.normal_(final_layer.linear.weight, std=0.02)
            nn.init.zeros_(final_layer.linear.bias)

    def _unpatchify(self, tokens: torch.Tensor) -> torch.Tensor:
        b, _, _ = tokens.shape
        p = self.patch_size
        x = tokens.reshape(b, self.grid_size, self.grid_size, p, p, self.in_channels)
        x = torch.einsum("bhwpqc->bchpwq", x)
        return x.reshape(b, self.in_channels, self.image_size, self.image_size)

    def forward(self, x: torch.Tensor, t: torch.Tensor, step: torch.Tensor):
        batch = x.shape[0]
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        register = self.register_tokens.expand(batch, -1, -1)
        tokens = torch.cat([register, tokens], dim=1)
        rope_cos, rope_sin = self.rope(tokens.shape[1])
        cond = self.conditioner(t, step, batch, x.device, self.hidden_size)
        for block in self.blocks:
            tokens = block(tokens, cond, rope_cos, rope_sin)
        tokens = tokens[:, self.num_register_tokens :, :]
        u = self._unpatchify(self.u_final(tokens, cond))
        if not self.dual_head:
            return u
        if self.v_final is None:
            raise RuntimeError("dual_head=True requires a v_final layer")
        v = self._unpatchify(self.v_final(tokens, cond))
        return u, v


class DirectDiTGenerator(nn.Module):
    def __init__(self, cfg: MnistConfig):
        super().__init__()
        image_size, channels, patch_size = space_network_shape(cfg)
        self.pixel_output = cfg.model_space == "pixel"
        self.net = PixelDiT(
            hidden_size=cfg.pixel_dit_hidden,
            depth=cfg.pixel_dit_depth,
            num_heads=cfg.pixel_dit_heads,
            patch_size=patch_size,
            dual_head=False,
            image_size=image_size,
            in_channels=channels,
            mlp_ratio=cfg.pixel_dit_mlp_ratio,
            num_register_tokens=cfg.pixel_dit_register_tokens,
            use_qk_norm=cfg.pixel_dit_use_qk_norm,
            use_style_embed=cfg.pixel_dit_use_style_embed,
            style_tokens=cfg.pixel_dit_style_tokens,
            style_codebook=cfg.pixel_dit_style_codebook,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        output = self.net(x, zeros, zeros)
        return torch.tanh(output) if self.pixel_output else output


def make_direct_generator(cfg: MnistConfig) -> nn.Module:
    if cfg.direct_backbone == "dit":
        return DirectDiTGenerator(cfg)
    if cfg.direct_backbone == "cnn":
        if cfg.model_space != "pixel":
            raise ValueError("direct_backbone=cnn is only supported for pixel-space MNIST")
        return DirectGenerator(cfg.hidden)
    raise ValueError(f"Unknown direct_backbone {cfg.direct_backbone!r}")


def space_network_shape(cfg: MnistConfig) -> Tuple[int, int, int]:
    if cfg.model_space == "pixel":
        return IMAGE_SIZE, 1, cfg.pixel_patch_size
    if cfg.model_space == "latent":
        return LATENT_SHAPE[1], LATENT_SHAPE[0], 1
    raise ValueError(f"Unknown model_space {cfg.model_space}")


def space_tensor_shape(cfg: MnistConfig) -> Tuple[int, int, int]:
    return IMAGE_SHAPE if cfg.model_space == "pixel" else LATENT_SHAPE


def make_conditioned_dit(cfg: MnistConfig, dual_head: bool) -> PixelDiT:
    image_size, channels, patch_size = space_network_shape(cfg)
    return PixelDiT(
        hidden_size=cfg.pixel_dit_hidden,
        depth=cfg.pixel_dit_depth,
        num_heads=cfg.pixel_dit_heads,
        patch_size=patch_size,
        dual_head=dual_head,
        image_size=image_size,
        in_channels=channels,
        mlp_ratio=cfg.pixel_dit_mlp_ratio,
        num_register_tokens=cfg.pixel_dit_register_tokens,
        use_qk_norm=cfg.pixel_dit_use_qk_norm,
        use_style_embed=cfg.pixel_dit_use_style_embed,
        style_tokens=cfg.pixel_dit_style_tokens,
        style_codebook=cfg.pixel_dit_style_codebook,
    )


@torch.no_grad()
def median_bandwidth_from_samples(
    z: torch.Tensor,
    max_points: int,
    sigma_min: float = 1.0,
    sigma_max: float = 64.0,
) -> torch.Tensor:
    z = z.detach()
    if z.shape[0] > max_points:
        z = z[torch.randperm(z.shape[0], device=z.device)[:max_points]]
    dist = torch.pdist(z).pow(2)
    valid = dist[dist > 1e-12]
    med = torch.median(valid) if valid.numel() else torch.tensor(float(IMAGE_DIM), device=z.device)
    return torch.sqrt(0.5 * med).clamp(sigma_min, sigma_max)


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
    values = k * (dot_vw.unsqueeze(0) / sigma2 - (v_diff * w_diff).unsqueeze(0) / sigma2.pow(2))
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
    sigmas: torch.Tensor,
    include_data_data: bool = False,
) -> torch.Tensor:
    full_mmd = include_data_data
    mm = derivative_kernel_bilinear_means(x_model, v_model, x_model, v_model, sigmas, drop_diagonal=not full_mmd)
    md = derivative_kernel_bilinear_means(x_model, v_model, x_data, v_data, sigmas)
    loss_per_scale = mm - 2.0 * md
    if include_data_data:
        dd = derivative_kernel_bilinear_means(x_data, v_data, x_data, v_data, sigmas, drop_diagonal=False)
        loss_per_scale = loss_per_scale + dd
    return loss_per_scale.mean()


def vector_flux_kernel_bilinear_means(
    x: torch.Tensor,
    v: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    sigmas: torch.Tensor,
    drop_diagonal: bool = False,
) -> torch.Tensor:
    """Mean of v^T K(x,y) w for K=diag(k_j(x_j,y_j)).

    Each output component has its own one-dimensional RBF kernel, so this is
    not the older scalar RBF kernel multiplied by the identity matrix.
    """
    sigma2 = sigmas.to(x.device, x.dtype).flatten().pow(2).clamp_min(1e-6)
    per_scale = []
    chunk = 256
    for s2 in sigma2:
        total = x.new_zeros(x.shape[0], y.shape[0])
        for start in range(0, x.shape[1], chunk):
            stop = min(start + chunk, x.shape[1])
            diff = x[:, None, start:stop] - y[None, :, start:stop]
            k = torch.exp(-0.5 * diff.pow(2) / s2)
            total = total + (k * v[:, None, start:stop] * w[None, :, start:stop]).sum(dim=-1)
        per_scale.append(total)
    values = torch.stack(per_scale, dim=0)
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
    statistic: str = "u",
) -> torch.Tensor:
    """Squared vector-flux MMD route for K(x,x') = diag_j k_j(x_j,x'_j).

    This treats each coordinate as an independent scalar RKHS channel, so the
    matrix-valued kernel is diagonal with one RBF per component.
    """
    if statistic not in {"u", "v"}:
        raise ValueError(f"Unknown vector-flux statistic {statistic!r}")
    drop_diagonal = statistic == "u" and not include_data_data
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


def normalize_velocity(v: torch.Tensor, eps: float) -> torch.Tensor:
    flat = flatten_img(v)
    scale = torch.sqrt(flat.pow(2).sum(dim=1).mean()).detach()
    return v / (scale + eps)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def step_epoch(step: int, cfg: MnistConfig) -> float:
    return float(step * cfg.batch_size / max(cfg.train_n, 1))


def method_slug(method: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in method).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "training"


def pixel_dit_config(cfg: MnistConfig) -> Dict[str, object]:
    image_size, channels, patch_size = space_network_shape(cfg)
    return {
        "model_family": "DriftDiT-Tiny-style",
        "image_size": image_size,
        "in_channels": channels,
        "patch_size": patch_size,
        "hidden_size": cfg.pixel_dit_hidden,
        "depth": cfg.pixel_dit_depth,
        "num_heads": cfg.pixel_dit_heads,
        "mlp_ratio": cfg.pixel_dit_mlp_ratio,
        "num_register_tokens": cfg.pixel_dit_register_tokens,
        "use_qk_norm": cfg.pixel_dit_use_qk_norm,
        "use_rope": True,
        "use_rmsnorm": True,
        "use_swiglu": True,
        "use_style_embed": cfg.pixel_dit_use_style_embed,
        "style_tokens": cfg.pixel_dit_style_tokens,
        "style_codebook": cfg.pixel_dit_style_codebook,
    }


def model_architecture_summary(model: nn.Module, cfg: MnistConfig) -> Dict[str, object]:
    return {
        "class_name": model.__class__.__name__,
        "total_parameters": count_parameters(model),
        "trainable_parameters": count_parameters(model, trainable_only=True),
        "model_space": cfg.model_space,
        "direct_backbone": cfg.direct_backbone,
        "pixel_dit": pixel_dit_config(cfg),
    }


def training_hyperparameters(cfg: MnistConfig) -> Dict[str, object]:
    return {
        "preset": cfg.preset,
        "seed": cfg.seed,
        "train_n": cfg.train_n,
        "val_n": cfg.val_n,
        "fid_samples": cfg.fid_samples,
        "steps": cfg.steps,
        "single_steps": cfg.single_steps,
        "estimated_epochs": step_epoch(cfg.steps, cfg),
        "batch_size": cfg.batch_size,
        "kernel_batch": cfg.kernel_batch,
        "learning_rate": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "grad_clip": cfg.grad_clip,
        "optimizer": "AdamW",
        "image_shape": list(IMAGE_SHAPE),
        "eval_image_size": EVAL_IMAGE_SIZE,
        "cfg": asdict(cfg),
    }


class TrainingLogger:
    def __init__(
        self,
        root: Path,
        method: str,
        cfg: MnistConfig,
        model: nn.Module,
        extra_hparams: Optional[Dict[str, object]] = None,
    ):
        self.method = method
        self.start_time = time.time()
        run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}"
        self.dir = root / f"{run_id}_{method_slug(method)}"
        self.dir.mkdir(parents=True, exist_ok=False)
        self.closed = False
        manifest: Dict[str, Any] = {
            "method": method,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "hyperparameters": training_hyperparameters(cfg),
            "architecture": model_architecture_summary(model, cfg),
            "extra_hyperparameters": extra_hparams or {},
        }
        (self.dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self.jsonl = open(self.dir / "steps.jsonl", "w", encoding="utf-8")
        self.csv_file = open(self.dir / "steps.csv", "w", newline="", encoding="utf-8")
        self.csv = csv.DictWriter(
            self.csv_file,
            fieldnames=["method", "step", "epoch", "loss", "lr", "elapsed_seconds", "metrics_json"],
        )
        self.csv.writeheader()

    def log_step(
        self,
        step: int,
        cfg: MnistConfig,
        loss: float,
        lr: float,
        metrics: Optional[Dict[str, object]] = None,
    ) -> None:
        metrics = metrics or {}
        record = {
            "method": self.method,
            "step": step,
            "epoch": step_epoch(step, cfg),
            "loss": float(loss),
            "lr": float(lr),
            "elapsed_seconds": time.time() - self.start_time,
            "metrics": metrics,
        }
        self.jsonl.write(json.dumps(record, sort_keys=True) + "\n")
        self.jsonl.flush()
        self.csv.writerow(
            {
                "method": self.method,
                "step": step,
                "epoch": record["epoch"],
                "loss": record["loss"],
                "lr": record["lr"],
                "elapsed_seconds": record["elapsed_seconds"],
                "metrics_json": json.dumps(metrics, sort_keys=True),
            }
        )
        self.csv_file.flush()

    def close(self, final_metrics: Optional[Dict[str, object]] = None) -> None:
        if self.closed:
            return
        summary = {
            "method": self.method,
            "elapsed_seconds": time.time() - self.start_time,
            "final_metrics": final_metrics or {},
        }
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.jsonl.close()
        self.csv_file.close()
        self.closed = True


class EarlyStopper:
    def __init__(self, cfg: MnistConfig):
        self.min_steps = int(cfg.early_stop_min_steps)
        self.patience = int(cfg.early_stop_patience)
        self.min_delta = float(cfg.early_stop_min_delta)
        self.metric = cfg.early_stop_metric
        self.values: List[float] = []
        self.stop_step: Optional[int] = None
        self.reason = ""

    @property
    def enabled(self) -> bool:
        return self.patience > 1 and self.min_delta > 0.0

    def _transform(self, loss: float) -> float:
        return abs(float(loss)) if self.metric == "abs_loss" else float(loss)

    def update(self, step: int, loss: float) -> bool:
        if not self.enabled:
            return False
        self.values.append(self._transform(loss))
        if step < self.min_steps or len(self.values) < self.patience:
            return False
        window = self.values[-self.patience :]
        start = sum(window[: max(1, self.patience // 4)]) / max(1, self.patience // 4)
        end_count = max(1, self.patience // 4)
        end = sum(window[-end_count:]) / end_count
        relative = abs(start - end) / max(abs(start), 1e-12)
        if relative < self.min_delta:
            self.stop_step = step
            self.reason = (
                f"{self.metric} relative change {relative:.3g} over "
                f"{self.patience} steps < {self.min_delta:.3g}"
            )
            return True
        return False

    def info(self, planned_steps: int) -> Dict[str, object]:
        return {
            "stopped_early": self.stop_step is not None,
            "stop_step": self.stop_step or planned_steps,
            "planned_steps": planned_steps,
            "early_stop_reason": self.reason,
            "early_stop_min_steps": self.min_steps,
            "early_stop_patience": self.patience,
            "early_stop_min_delta": self.min_delta,
            "early_stop_metric": self.metric,
        }


def make_training_logger(
    log_dir: Optional[Path],
    method: str,
    cfg: MnistConfig,
    model: nn.Module,
    extra_hparams: Optional[Dict[str, object]] = None,
) -> Optional[TrainingLogger]:
    if log_dir is None:
        return None
    return TrainingLogger(log_dir, method, cfg, model, extra_hparams)


@torch.no_grad()
def compute_wbvm_sigmas(
    train_loader: InfiniteLoader,
    cfg: MnistConfig,
    device: torch.device,
    codec: Optional[LeNetLatentCodec],
) -> Tuple[torch.Tensor, float]:
    x1 = cycle_space_batch(train_loader, cfg, device, codec)[: cfg.kernel_bandwidth_points]
    if x1.shape[0] < cfg.kernel_bandwidth_points:
        xs = [x1]
        while sum(t.shape[0] for t in xs) < cfg.kernel_bandwidth_points:
            xs.append(cycle_space_batch(train_loader, cfg, device, codec))
        x1 = torch.cat(xs, dim=0)[: cfg.kernel_bandwidth_points]
    x0 = torch.randn_like(x1)
    tau = torch.empty(x1.shape[0], 1, 1, 1, device=device).uniform_(cfg.all_tau_low, cfg.all_tau_high)
    bridge = (1.0 - tau) * x0 + tau * x1
    sigma0 = median_bandwidth_from_samples(flatten_img(bridge), cfg.kernel_bandwidth_points)
    scales = torch.tensor(cfg.kernel_scales, device=device, dtype=bridge.dtype)
    sigmas = (sigma0 * scales).detach()
    return sigmas, float(sigma0.detach().cpu())


def train_wbvm_all(
    train_loader: InfiniteLoader,
    cfg: MnistConfig,
    device: torch.device,
    codec: Optional[LeNetLatentCodec],
    log_dir: Optional[Path] = None,
    log_name: str = "WBVM-all",
    extra_hparams: Optional[Dict[str, object]] = None,
) -> Tuple[nn.Module, Dict[str, object]]:
    sigmas, sigma0 = compute_wbvm_sigmas(train_loader, cfg, device, codec)
    model = make_direct_generator(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    logger_extra = {
        "kernel_base_sigma": sigma0,
        "kernel_sigmas": [float(s) for s in sigmas.detach().cpu()],
        "tau_range": [cfg.all_tau_low, cfg.all_tau_high],
        **(extra_hparams or {}),
    }
    logger = make_training_logger(log_dir, log_name, cfg, model, logger_extra)
    start = time.time()
    last_loss = 0.0
    early = EarlyStopper(cfg)

    for step in trange(cfg.steps, desc="WBVM-all", leave=False):
        x1_full = cycle_space_batch(train_loader, cfg, device, codec)
        x1 = x1_full[: cfg.kernel_batch]
        x0 = torch.randn_like(x1)
        u = torch.randn_like(x1)
        tau = torch.empty(x1.shape[0], 1, 1, 1, device=device).uniform_(cfg.all_tau_low, cfg.all_tau_high)
        y = model(u)
        x_model = (1.0 - tau) * x0 + tau * y
        x_data = (1.0 - tau) * x0 + tau * x1
        v_model = normalize_velocity(y - x0, cfg.velocity_eps)
        v_data = normalize_velocity(x1 - x0, cfg.velocity_eps)
        loss = rkhs_flux_u_stat(
            flatten_img(x_model),
            flatten_img(v_model),
            flatten_img(x_data),
            flatten_img(v_data),
            sigmas=sigmas,
            include_data_data=True,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        last_loss = float(loss.detach().cpu())
        if not math.isfinite(last_loss):
            raise RuntimeError("WBVM-all loss became non-finite")
        if logger is not None:
            logger.log_step(
                step + 1,
                cfg,
                last_loss,
                current_lr(opt),
                {"tau_mean": float(tau.mean().detach().cpu()), "kernel_base_sigma": sigma0},
            )
        if early.update(step + 1, last_loss):
            break

    info = {
        "train_seconds": time.time() - start,
        "train_steps": early.stop_step or cfg.steps,
        "kernel_base_sigma": sigma0,
        "kernel_sigmas": ",".join(f"{float(s):.6g}" for s in sigmas.detach().cpu()),
        "loss": last_loss,
        "loss_statistic": "rkhs_flux_full_mmd",
        "nfe": 1,
        "model_space": cfg.model_space,
        "note": (
            "original h(U), independent U, common bridge X0, no tied_X0; "
            f"{cfg.direct_backbone} direct generator; {cfg.model_space} space"
        ),
        **early.info(cfg.steps),
    }
    if logger is not None:
        logger.close(info)
        info["log_dir"] = str(logger.dir)
    return model, info


def train_wbvm_vector(
    train_loader: InfiniteLoader,
    cfg: MnistConfig,
    device: torch.device,
    codec: Optional[LeNetLatentCodec],
    log_dir: Optional[Path] = None,
    log_name: str = "vMMD-WBVM",
    extra_hparams: Optional[Dict[str, object]] = None,
) -> Tuple[nn.Module, Dict[str, object]]:
    sigmas, sigma0 = compute_wbvm_sigmas(train_loader, cfg, device, codec)
    model = make_direct_generator(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    logger_extra = {
        "kernel_base_sigma": sigma0,
        "kernel_sigmas": [float(s) for s in sigmas.detach().cpu()],
        "tau_range": [cfg.all_tau_low, cfg.all_tau_high],
        "loss_statistic": cfg.vector_loss_statistic,
        **(extra_hparams or {}),
    }
    logger = make_training_logger(log_dir, log_name, cfg, model, logger_extra)
    start = time.time()
    last_loss = 0.0
    early = EarlyStopper(cfg)

    for step in trange(cfg.steps, desc="vMMD-WBVM", leave=False):
        x1_full = cycle_space_batch(train_loader, cfg, device, codec)
        x1 = x1_full[: cfg.kernel_batch]
        x0 = torch.randn_like(x1)
        u = torch.randn_like(x1)
        tau = torch.empty(x1.shape[0], 1, 1, 1, device=device).uniform_(cfg.all_tau_low, cfg.all_tau_high)
        y = model(u)
        x_model = (1.0 - tau) * x0 + tau * y
        x_data = (1.0 - tau) * x0 + tau * x1
        v_model = normalize_velocity(y - x0, cfg.velocity_eps)
        v_data = normalize_velocity(x1 - x0, cfg.velocity_eps)
        loss = vector_flux_mmd2(
            flatten_img(x_model),
            flatten_img(v_model),
            flatten_img(x_data),
            flatten_img(v_data),
            sigmas=sigmas,
            include_data_data=True,
            statistic=cfg.vector_loss_statistic,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        last_loss = float(loss.detach().cpu())
        if not math.isfinite(last_loss):
            raise RuntimeError("vMMD-WBVM loss became non-finite")
        if logger is not None:
            logger.log_step(
                step + 1,
                cfg,
                last_loss,
                current_lr(opt),
                {"tau_mean": float(tau.mean().detach().cpu()), "kernel_base_sigma": sigma0},
            )
        if early.update(step + 1, last_loss):
            break

    info = {
        "train_seconds": time.time() - start,
        "train_steps": early.stop_step or cfg.steps,
        "kernel_base_sigma": sigma0,
        "kernel_sigmas": ",".join(f"{float(s):.6g}" for s in sigmas.detach().cpu()),
        "loss": last_loss,
        "loss_statistic": f"vector_flux_component_diag_rbf_full_mmd_{cfg.vector_loss_statistic}",
        "nfe": 1,
        "model_space": cfg.model_space,
        "note": (
            "route-two vector-valued RKHS flux MMD with component-wise diagonal RBF kernel; "
            f"{cfg.direct_backbone} direct generator; {cfg.model_space} space"
        ),
        **early.info(cfg.steps),
    }
    if logger is not None:
        logger.close(info)
        info["log_dir"] = str(logger.dir)
    return model, info


def train_wbvm_single(
    train_loader: InfiniteLoader,
    val_loader: DataLoader,
    cfg: MnistConfig,
    device: torch.device,
    outdir: Path,
    feature_net: Optional[MnistFeatureNet],
    codec: Optional[LeNetLatentCodec],
    log_dir: Optional[Path] = None,
) -> Tuple[nn.Module, Dict[str, object], List[Dict[str, object]]]:
    if feature_net is None:
        raise ValueError("WBVM-single validation requires the MNIST LeNet feature extractor")
    best_model: Optional[nn.Module] = None
    best_info: Dict[str, object] = {}
    best_fid = float("inf")
    rows: List[Dict[str, object]] = []

    for tau_value in cfg.wbvm_single_taus:
        local_cfg = MnistConfig(**{**asdict(cfg), "steps": cfg.single_steps, "all_tau_low": tau_value, "all_tau_high": tau_value})
        model, info = train_wbvm_all(
            train_loader,
            local_cfg,
            device,
            codec,
            log_dir=log_dir,
            log_name=f"WBVM-single-tau-{tau_value:.2f}",
            extra_hparams={"parent_method": "WBVM-single", "candidate_tau": tau_value},
        )
        score = compute_mnist_fid_for_sampler(
            lambda n, b: sample_direct(model, n, b, device, cfg, codec),
            val_loader,
            cfg,
            device,
            feature_net,
            n_samples=min(cfg.val_n, len(val_loader.dataset)),
        )
        row = {
            "method": "WBVM-single-candidate",
            "tau": tau_value,
            "val_metric": "mnist_fid",
            "val_score": score,
            **info,
        }
        rows.append(row)
        print(f"WBVM-single tau={tau_value:.2f} validation MNIST-FID={score:.3f}", flush=True)
        if score < best_fid:
            if best_model is not None:
                del best_model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            best_model = model
            best_fid = score
            best_info = dict(info)
            best_info.update(
                {
                    "selected_tau": tau_value,
                    "validation_metric": "mnist_fid",
                    "validation_score": score,
                }
            )
        else:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if best_model is None:
        raise RuntimeError("No WBVM-single candidate finished")
    with open(outdir / "wbvm_single_selection.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)
    return best_model, best_info, rows


def train_wbvm_vector_single(
    train_loader: InfiniteLoader,
    val_loader: DataLoader,
    cfg: MnistConfig,
    device: torch.device,
    outdir: Path,
    feature_net: Optional[MnistFeatureNet],
    codec: Optional[LeNetLatentCodec],
    log_dir: Optional[Path] = None,
) -> Tuple[nn.Module, Dict[str, object], List[Dict[str, object]]]:
    if feature_net is None and cfg.selection_metric in {"mnist_fid", "mnist_kid"}:
        raise ValueError("vMMD-WBVM-single validation requires the MNIST LeNet feature extractor")
    best_model: Optional[nn.Module] = None
    best_info: Dict[str, object] = {}
    best_score = float("inf")
    rows: List[Dict[str, object]] = []

    for tau_value in cfg.wbvm_single_taus:
        local_cfg = MnistConfig(**{**asdict(cfg), "steps": cfg.single_steps, "all_tau_low": tau_value, "all_tau_high": tau_value})
        model, info = train_wbvm_vector(
            train_loader,
            local_cfg,
            device,
            codec,
            log_dir=log_dir,
            log_name=f"vMMD-WBVM-single-tau-{tau_value:.2f}",
            extra_hparams={"parent_method": "vMMD-WBVM-single", "candidate_tau": tau_value},
        )
        score = selection_score_for_sampler(
            lambda n, b, m=model: sample_direct(m, n, b, device, cfg, codec),
            val_loader,
            cfg,
            device,
            feature_net,
        )
        row = {
            "method": "vMMD-WBVM-single-candidate",
            "tau": tau_value,
            "val_metric": cfg.selection_metric,
            "val_score": score,
            **info,
        }
        rows.append(row)
        print(f"vMMD-WBVM tau={tau_value:.2f} validation {cfg.selection_metric}={score:.3f}", flush=True)
        if score < best_score:
            if best_model is not None:
                del best_model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            best_model = model
            best_score = score
            best_info = dict(info)
            best_info.update(
                {
                    "selected_tau": tau_value,
                    "validation_metric": cfg.selection_metric,
                    "validation_score": score,
                }
            )
        else:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if best_model is None:
        raise RuntimeError("No vMMD-WBVM-single candidate finished")
    with open(outdir / "vmmd_wbvm_single_selection.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)
    return best_model, best_info, rows


def sample_pixel_meanflow_tr(batch: int, cfg: MnistConfig, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    t = torch.sigmoid(torch.randn(batch, 1, device=device) * cfg.meanflow_logit_std + cfg.meanflow_logit_mean)
    r = torch.sigmoid(torch.randn(batch, 1, device=device) * cfg.meanflow_logit_std + cfg.meanflow_logit_mean)
    uniform_mask = torch.rand(batch, 1, device=device) < 0.1
    t = torch.where(uniform_mask, torch.rand_like(t), t)
    r = torch.where(uniform_mask, torch.rand_like(r), r)
    data_n = int(batch * cfg.meanflow_data_proportion)
    if data_n:
        r[:data_n] = t[:data_n]
    return torch.maximum(t, r), torch.minimum(t, r)


def adaptive_pmf_loss(error: torch.Tensor, cfg: MnistConfig) -> torch.Tensor:
    per_example = error.flatten(1).pow(2).sum(dim=1)
    weight = (per_example + cfg.meanflow_norm_eps).pow(cfg.meanflow_norm_p).detach()
    return (per_example / weight).mean()


def uses_cnn_pixel_velocity(cfg: MnistConfig) -> bool:
    return cfg.direct_backbone == "cnn" and cfg.model_space == "pixel"


def train_meanflow_cnn(
    train_loader: InfiniteLoader,
    cfg: MnistConfig,
    device: torch.device,
    log_dir: Optional[Path] = None,
    log_name: str = "MeanFlow",
    extra_hparams: Optional[Dict[str, object]] = None,
) -> Tuple[nn.Module, Dict[str, object]]:
    model = VelocityNet(cond_dim=2, hidden=cfg.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    logger = make_training_logger(log_dir, log_name, cfg, model, {"velocity_backbone": "CondConvNet", **(extra_hparams or {})})
    start = time.time()
    last_loss = 0.0
    last_raw_mse = 0.0
    early = EarlyStopper(cfg)

    for step in trange(cfg.steps, desc="MeanFlow", leave=False):
        x = cycle_batch(train_loader, device)
        e = torch.randn_like(x)
        t = torch.rand(x.shape[0], 1, device=device)
        r = torch.rand(x.shape[0], 1, device=device)
        t, r = torch.maximum(t, r), torch.minimum(t, r)
        t4 = t.view(-1, 1, 1, 1)
        z = (1.0 - t4) * x + t4 * e
        v_target = e - x

        def fn(z_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
            return model(z_in, torch.cat([r, t_in], dim=1))

        u, du_dt = torch.func.jvp(fn, (z, t), (v_target, torch.ones_like(t)))
        target = (v_target - (t - r).view(-1, 1, 1, 1) * du_dt).detach()
        loss = F.mse_loss(u, target)
        raw_mse_u = F.mse_loss(u, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        last_loss = float(loss.detach().cpu())
        last_raw_mse = float(raw_mse_u.detach().cpu())
        if not math.isfinite(last_loss):
            raise RuntimeError("MeanFlow loss became non-finite")
        if logger is not None:
            logger.log_step(
                step + 1,
                cfg,
                last_loss,
                current_lr(opt),
                {
                    "raw_mse_u": last_raw_mse,
                    "r_mean": float(r.mean().detach().cpu()),
                    "t_mean": float(t.mean().detach().cpu()),
                },
            )
        if early.update(step + 1, last_loss):
            break

    info = {
        "train_seconds": time.time() - start,
        "train_steps": early.stop_step or cfg.steps,
        "loss": last_loss,
        "raw_mse_u": last_raw_mse,
        "nfe": 1,
        "model_space": cfg.model_space,
        "note": "MeanFlow identity: sorted-uniform r,t; CNN VelocityNet; one-step x=e-u(e,0,1)",
        **early.info(cfg.steps),
    }
    if logger is not None:
        logger.close(info)
        info["log_dir"] = str(logger.dir)
    return model, info


def train_meanflow(
    train_loader: InfiniteLoader,
    cfg: MnistConfig,
    device: torch.device,
    codec: Optional[LeNetLatentCodec],
    log_dir: Optional[Path] = None,
    log_name: str = "MeanFlow",
    extra_hparams: Optional[Dict[str, object]] = None,
) -> Tuple[nn.Module, Dict[str, object]]:
    if uses_cnn_pixel_velocity(cfg):
        return train_meanflow_cnn(train_loader, cfg, device, log_dir=log_dir, log_name=log_name, extra_hparams=extra_hparams)
    model = make_conditioned_dit(cfg, dual_head=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    logger = make_training_logger(
        log_dir,
        log_name,
        cfg,
        model,
        {"velocity_heads": "dual_u_v", "meanflow_data_proportion": cfg.meanflow_data_proportion, **(extra_hparams or {})},
    )
    start = time.time()
    last_loss = 0.0
    last_loss_u = 0.0
    last_loss_v = 0.0
    last_raw_mse_u = 0.0
    last_raw_mse_v = 0.0
    early = EarlyStopper(cfg)

    for step in trange(cfg.steps, desc="MeanFlow", leave=False):
        x = cycle_space_batch(train_loader, cfg, device, codec)
        e = torch.randn_like(x)
        t, r = sample_pixel_meanflow_tr(x.shape[0], cfg, device)
        t4 = t.view(-1, 1, 1, 1)
        z = (1.0 - t4) * x + t4 * e
        v_target = e - x
        zeros = torch.zeros_like(t)
        with torch.no_grad():
            _, v_direction = model(z, t, zeros)

        def fn(z_in: torch.Tensor, t_in: torch.Tensor, r_in: torch.Tensor):
            u_pred, v_pred = model(z_in, t_in, t_in - r_in)
            return u_pred, v_pred

        u, dudt, v_pred = torch.func.jvp(
            fn,
            (z, t, r),
            (v_direction.detach(), torch.ones_like(t), torch.zeros_like(r)),
            has_aux=True,
        )
        compound_v = u + (t - r).view(-1, 1, 1, 1) * dudt.detach()
        error_u = compound_v - v_target.detach()
        error_v = v_pred - v_target.detach()
        loss_u = adaptive_pmf_loss(error_u, cfg)
        loss_v = adaptive_pmf_loss(error_v, cfg)
        raw_mse_u = F.mse_loss(compound_v, v_target.detach())
        raw_mse_v = F.mse_loss(v_pred, v_target.detach())
        loss = loss_u + loss_v
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        last_loss = float(loss.detach().cpu())
        last_loss_u = float(loss_u.detach().cpu())
        last_loss_v = float(loss_v.detach().cpu())
        last_raw_mse_u = float(raw_mse_u.detach().cpu())
        last_raw_mse_v = float(raw_mse_v.detach().cpu())
        if not math.isfinite(last_loss):
            raise RuntimeError("MeanFlow loss became non-finite")
        if logger is not None:
            logger.log_step(
                step + 1,
                cfg,
                last_loss,
                current_lr(opt),
                {
                    "loss_u": last_loss_u,
                    "loss_v": last_loss_v,
                    "raw_mse_u": last_raw_mse_u,
                    "raw_mse_v": last_raw_mse_v,
                    "r_mean": float(r.mean().detach().cpu()),
                    "t_mean": float(t.mean().detach().cpu()),
                },
            )
        if early.update(step + 1, last_loss):
            break

    info = {
        "train_seconds": time.time() - start,
        "train_steps": early.stop_step or cfg.steps,
        "loss": last_loss,
        "loss_u": last_loss_u,
        "loss_v": last_loss_v,
        "raw_mse_u": last_raw_mse_u,
        "raw_mse_v": last_raw_mse_v,
        "nfe": 1,
        "model_space": cfg.model_space,
        "note": "Pixel MeanFlow official core: DiT u/v heads; logit-normal (t,r); predicted-v JVP; adaptive velocity loss",
        **early.info(cfg.steps),
    }
    if logger is not None:
        logger.close(info)
        info["log_dir"] = str(logger.dir)
    return model, info


@torch.no_grad()
def update_ema_model(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
    for ema_buf, buf in zip(ema_model.buffers(), model.buffers()):
        ema_buf.copy_(buf)


def sample_shortcut_dt(batch: int, cfg: MnistConfig, device: torch.device, min_power: int = 0) -> torch.Tensor:
    min_dt = 1.0 / cfg.shortcut_min_steps
    powers = int(math.log2(cfg.shortcut_min_steps))
    k = torch.randint(min_power, powers + 1, (batch, 1), device=device)
    return (min_dt * (2.0 ** k.float())).clamp(max=1.0)


def sample_shortcut_self_times(batch: int, cfg: MnistConfig, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    min_dt = 1.0 / cfg.shortcut_min_steps
    powers = int(math.log2(cfg.shortcut_min_steps))
    k = torch.randint(1, powers + 1, (batch, 1), device=device)
    step_counts = (cfg.shortcut_min_steps // (2 ** k)).long().clamp_min(1)
    m = torch.floor(torch.rand(batch, 1, device=device) * step_counts.float()).long()
    dt = (min_dt * (2.0 ** k.float())).clamp(max=1.0)
    t = m.float() * dt
    return t, dt


def balanced_shortcut_dt_base(batch: int, sections: int, device: torch.device) -> torch.Tensor:
    values = torch.arange(sections - 1, -1, -1, device=device)
    repeats = math.ceil(batch / sections)
    return values.repeat(repeats)[:batch].float().view(-1, 1)


def train_shortcut_cnn(
    train_loader: InfiniteLoader,
    cfg: MnistConfig,
    device: torch.device,
    log_dir: Optional[Path] = None,
    log_name: str = "ShortcutFlow",
    extra_hparams: Optional[Dict[str, object]] = None,
) -> Tuple[nn.Module, Dict[str, object]]:
    model = VelocityNet(cond_dim=2, hidden=cfg.hidden).to(device)
    ema_model = VelocityNet(cond_dim=2, hidden=cfg.hidden).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    logger = make_training_logger(log_dir, log_name, cfg, model, {"velocity_backbone": "CondConvNet", **(extra_hparams or {})})
    start = time.time()
    last_loss = 0.0
    last_flow_loss = 0.0
    last_self_consistency_loss = 0.0
    flow_frac = float(cfg.shortcut_empirical_frac)
    early = EarlyStopper(cfg)

    for step in trange(cfg.steps, desc="ShortcutFlow", leave=False):
        x1 = cycle_batch(train_loader, device)
        x0 = torch.randn_like(x1)
        bsz = x1.shape[0]
        flow_n = min(max(1, int(round(bsz * flow_frac))), bsz - 1)
        shortcut_n = bsz - flow_n
        losses = []
        weights = []

        t_flow = torch.randint(0, cfg.shortcut_min_steps, (flow_n, 1), device=device).float() / cfg.shortcut_min_steps
        d_min = torch.full_like(t_flow, 1.0 / cfg.shortcut_min_steps)
        x_flow = (1.0 - t_flow.view(-1, 1, 1, 1)) * x0[:flow_n] + t_flow.view(-1, 1, 1, 1) * x1[:flow_n]
        target_flow = x1[:flow_n] - x0[:flow_n]
        pred_flow = model(x_flow, torch.cat([t_flow, d_min], dim=1))
        flow_loss = F.mse_loss(pred_flow, target_flow)
        losses.append(flow_loss)
        weights.append(flow_n)
        self_consistency_loss = x1.new_tensor(0.0)

        if shortcut_n > 0:
            x0_sc = x0[flow_n:]
            x1_sc = x1[flow_n:]
            t_sc, dt = sample_shortcut_self_times(shortcut_n, cfg, device)
            x_sc = (1.0 - t_sc.view(-1, 1, 1, 1)) * x0_sc + t_sc.view(-1, 1, 1, 1) * x1_sc
            half = 0.5 * dt
            with torch.no_grad():
                v1 = ema_model(x_sc, torch.cat([t_sc, half], dim=1))
                x_mid = (x_sc + half.view(-1, 1, 1, 1) * v1).clamp(-4.0, 4.0)
                v2 = ema_model(x_mid, torch.cat([t_sc + half, half], dim=1))
                target_sc = (0.5 * (v1 + v2)).clamp(-4.0, 4.0)
            pred_sc = model(x_sc, torch.cat([t_sc, dt], dim=1))
            self_consistency_loss = F.mse_loss(pred_sc, target_sc.detach())
            losses.append(self_consistency_loss)
            weights.append(shortcut_n)

        loss = sum(loss_i * weight for loss_i, weight in zip(losses, weights)) / sum(weights)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        update_ema_model(ema_model, model, cfg.shortcut_ema_decay)
        last_loss = float(loss.detach().cpu())
        last_flow_loss = float(flow_loss.detach().cpu())
        last_self_consistency_loss = float(self_consistency_loss.detach().cpu())
        if not math.isfinite(last_loss):
            raise RuntimeError("ShortcutFlow loss became non-finite")
        if logger is not None:
            logger.log_step(
                step + 1,
                cfg,
                last_loss,
                current_lr(opt),
                {
                    "flow_matching_loss": last_flow_loss,
                    "self_consistency_loss": last_self_consistency_loss,
                    "flow_n": flow_n,
                    "shortcut_n": shortcut_n,
                },
            )
        if early.update(step + 1, last_loss):
            break

    info = {
        "train_seconds": time.time() - start,
        "train_steps": early.stop_step or cfg.steps,
        "loss": last_loss,
        "flow_matching_loss": last_flow_loss,
        "self_consistency_loss": last_self_consistency_loss,
        "nfe": 1,
        "model_space": cfg.model_space,
        "note": "Shortcut Models: CNN VelocityNet; 75% flow grounding; EMA self-consistency",
        "shortcut_ema_decay": cfg.shortcut_ema_decay,
        "shortcut_empirical_frac": cfg.shortcut_empirical_frac,
        **early.info(cfg.steps),
    }
    if logger is not None:
        logger.close(info)
        info["log_dir"] = str(logger.dir)
    return ema_model, info


def train_shortcut(
    train_loader: InfiniteLoader,
    cfg: MnistConfig,
    device: torch.device,
    codec: Optional[LeNetLatentCodec],
    log_dir: Optional[Path] = None,
    log_name: str = "ShortcutFlow",
    extra_hparams: Optional[Dict[str, object]] = None,
) -> Tuple[nn.Module, Dict[str, object]]:
    if uses_cnn_pixel_velocity(cfg):
        return train_shortcut_cnn(train_loader, cfg, device, log_dir=log_dir, log_name=log_name, extra_hparams=extra_hparams)
    model = make_conditioned_dit(cfg, dual_head=False).to(device)
    ema_model = make_conditioned_dit(cfg, dual_head=False).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    logger = make_training_logger(
        log_dir,
        log_name,
        cfg,
        model,
        {
            "velocity_heads": "single",
            "shortcut_bootstrap_every": cfg.shortcut_bootstrap_every,
            "shortcut_ema_decay": cfg.shortcut_ema_decay,
            **(extra_hparams or {}),
        },
    )
    start = time.time()
    last_loss = 0.0
    last_flow_loss = 0.0
    last_self_consistency_loss = 0.0
    sections = int(math.log2(cfg.shortcut_min_steps))
    bootstrap_n = max(1, cfg.batch_size // cfg.shortcut_bootstrap_every)
    early = EarlyStopper(cfg)

    for step in trange(cfg.steps, desc="ShortcutFlow", leave=False):
        x1 = cycle_space_batch(train_loader, cfg, device, codec)
        x0 = torch.randn_like(x1)
        bsz = x1.shape[0]
        bst_n = min(bootstrap_n, bsz - 1)
        flow_n = bsz - bst_n

        dt_base = balanced_shortcut_dt_base(bst_n, sections, device)
        dt = 2.0 ** (-dt_base)
        step_count = (2.0**dt_base).long().clamp_min(1)
        t_index = torch.floor(torch.rand(bst_n, 1, device=device) * step_count.float())
        t_bst = t_index / step_count.float()
        x_bst = (1.0 - (1.0 - 1e-5) * t_bst.view(-1, 1, 1, 1)) * x0[:bst_n] + t_bst.view(-1, 1, 1, 1) * x1[:bst_n]
        half = 0.5 * dt
        child_base = dt_base + 1.0
        with torch.no_grad():
            v1 = ema_model(x_bst, t_bst, child_base)
            x_mid = (x_bst + half.view(-1, 1, 1, 1) * v1).clamp(-4.0, 4.0)
            v2 = ema_model(x_mid, t_bst + half, child_base)
            target_bst = (0.5 * (v1 + v2)).clamp(-4.0, 4.0)

        t_idx = torch.randint(0, cfg.shortcut_min_steps, (flow_n, 1), device=device)
        t_flow = t_idx.float() / cfg.shortcut_min_steps
        x_flow = (1.0 - (1.0 - 1e-5) * t_flow.view(-1, 1, 1, 1)) * x0[bst_n:] + t_flow.view(-1, 1, 1, 1) * x1[bst_n:]
        target_flow = x1[bst_n:] - (1.0 - 1e-5) * x0[bst_n:]
        flow_base = torch.full_like(t_flow, float(sections))

        x_train = torch.cat([x_bst, x_flow], dim=0)
        t_train = torch.cat([t_bst, t_flow], dim=0)
        dt_train = torch.cat([dt_base, flow_base], dim=0)
        target = torch.cat([target_bst, target_flow], dim=0)
        pred = model(x_train, t_train, dt_train)
        pred_bst = pred[:bst_n]
        pred_flow = pred[bst_n:]
        self_consistency_loss = F.mse_loss(pred_bst, target_bst)
        flow_loss = F.mse_loss(pred_flow, target_flow)
        loss = (self_consistency_loss * bst_n + flow_loss * flow_n) / bsz
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        update_ema_model(ema_model, model, cfg.shortcut_ema_decay)
        last_loss = float(loss.detach().cpu())
        last_flow_loss = float(flow_loss.detach().cpu())
        last_self_consistency_loss = float(self_consistency_loss.detach().cpu())
        if not math.isfinite(last_loss):
            raise RuntimeError("ShortcutFlow loss became non-finite")
        if logger is not None:
            logger.log_step(
                step + 1,
                cfg,
                last_loss,
                current_lr(opt),
                {
                    "flow_matching_loss": last_flow_loss,
                    "self_consistency_loss": last_self_consistency_loss,
                    "bootstrap_n": bst_n,
                    "flow_n": flow_n,
                },
            )
        if early.update(step + 1, last_loss):
            break

    info = {
        "train_seconds": time.time() - start,
        "train_steps": early.stop_step or cfg.steps,
        "loss": last_loss,
        "flow_matching_loss": last_flow_loss,
        "self_consistency_loss": last_self_consistency_loss,
        "nfe": 1,
        "model_space": cfg.model_space,
        "note": "Shortcut Models official core: shared DiT; 1/8 EMA bootstrap; balanced dt_base; flow grounding",
        "shortcut_ema_decay": cfg.shortcut_ema_decay,
        "shortcut_bootstrap_every": cfg.shortcut_bootstrap_every,
        **early.info(cfg.steps),
    }
    if logger is not None:
        logger.close(info)
        info["log_dir"] = str(logger.dir)
    return ema_model, info


def drifting_vectors(
    x: torch.Tensor,
    y_pos: torch.Tensor,
    y_neg: torch.Tensor,
    temperatures: Iterable[float],
    normalize_distances: bool,
    normalize_drift: bool,
) -> List[torch.Tensor]:
    """Algorithm-2 drifting field from Deng et al. (arXiv:2602.04770).

    Distances are optionally normalized by the in-batch distance scale before
    applying the paper's softmax temperatures. The returned vectors are detached
    targets, matching the stop-gradient target in Algorithm 1.
    """
    xf = flatten_img(x.detach())
    posf = flatten_img(y_pos.detach())
    negf = flatten_img(y_neg.detach())
    dist_pos = torch.cdist(xf, posf)
    dist_neg = torch.cdist(xf, negf)
    if xf.shape[0] == negf.shape[0]:
        dist_neg = dist_neg + torch.eye(xf.shape[0], device=xf.device, dtype=xf.dtype) * 1e6
    if normalize_distances:
        finite = torch.cat([dist_pos.flatten(), dist_neg[dist_neg < 1e5].flatten()])
        scale = finite.median().detach().clamp_min(1e-6) if finite.numel() else dist_pos.new_tensor(1.0)
        dist_pos = dist_pos / scale
        dist_neg = dist_neg / scale

    vectors: List[torch.Tensor] = []
    n_pos = posf.shape[0]
    for temperature in temperatures:
        temp = max(float(temperature), 1e-6)
        logit = torch.cat([-dist_pos / temp, -dist_neg / temp], dim=1)
        a_row = torch.softmax(logit, dim=1)
        a_col = torch.softmax(logit, dim=0)
        a = torch.sqrt((a_row * a_col).clamp_min(0.0))
        a_pos, a_neg = torch.split(a, [n_pos, negf.shape[0]], dim=1)
        w_pos = a_pos * a_neg.sum(dim=1, keepdim=True)
        w_neg = a_neg * a_pos.sum(dim=1, keepdim=True)
        field = (w_pos @ posf - w_neg @ negf).view_as(x)
        if normalize_drift:
            field = field / (field.reshape(field.shape[0], -1).pow(2).sum(dim=1).mean().sqrt().detach() + 1e-6)
        vectors.append(field.detach())
    return vectors


def train_drifting(
    train_loader: InfiniteLoader,
    cfg: MnistConfig,
    device: torch.device,
    codec: Optional[LeNetLatentCodec],
    log_dir: Optional[Path] = None,
    log_name: str = "Drifting",
    extra_hparams: Optional[Dict[str, object]] = None,
) -> Tuple[nn.Module, Dict[str, object]]:
    model = make_direct_generator(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    logger = make_training_logger(
        log_dir,
        log_name,
        cfg,
        model,
        {"drifting_temperatures": cfg.drifting_temperatures, "drifting_step_size": cfg.drifting_step_size, **(extra_hparams or {})},
    )
    start = time.time()
    last_loss = 0.0
    early = EarlyStopper(cfg)

    for step in trange(cfg.steps, desc="Drifting", leave=False):
        x_real = cycle_space_batch(train_loader, cfg, device, codec)
        u = torch.randn_like(x_real)
        y = model(u)
        y_for_loss = y
        x_real_for_loss = x_real
        y_neg_for_loss = y
        fields = drifting_vectors(
            y_for_loss,
            x_real_for_loss,
            y_neg_for_loss,
            temperatures=cfg.drifting_temperatures,
            normalize_distances=cfg.drifting_normalize_distances,
            normalize_drift=cfg.drifting_normalize_drift,
        )
        field = sum(fields)
        target = (y_for_loss.detach() + cfg.drifting_step_size * field).detach()
        loss = F.mse_loss(y_for_loss, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        last_loss = float(loss.detach().cpu())
        if not math.isfinite(last_loss):
            raise RuntimeError("Drifting loss became non-finite")
        if logger is not None:
            logger.log_step(
                step + 1,
                cfg,
                last_loss,
                current_lr(opt),
                {"drift_rms": float(field.reshape(field.shape[0], -1).pow(2).mean().sqrt().detach().cpu())},
            )
        if early.update(step + 1, last_loss):
            break

    info = {
        "train_seconds": time.time() - start,
        "train_steps": early.stop_step or cfg.steps,
        "loss": last_loss,
        "nfe": 1,
        "note": (
            "Drifting Models arXiv:2602.04770 Algorithm 1/2; "
            f"{cfg.direct_backbone} direct generator; {cfg.model_space} space; self negatives"
        ),
        "drifting_temperatures": ",".join(f"{t:g}" for t in cfg.drifting_temperatures),
        "drifting_step_size": cfg.drifting_step_size,
        "drifting_normalize_distances": cfg.drifting_normalize_distances,
        "drifting_normalize_drift": cfg.drifting_normalize_drift,
        "model_space": cfg.model_space,
        **early.info(cfg.steps),
    }
    if logger is not None:
        logger.close(info)
        info["log_dir"] = str(logger.dir)
    return model, info


@torch.no_grad()
def sample_direct(
    model: nn.Module,
    n: int,
    batch_size: int,
    device: torch.device,
    cfg: MnistConfig,
    codec: Optional[LeNetLatentCodec],
) -> torch.Tensor:
    model.eval()
    outs = []
    for start in range(0, n, batch_size):
        b = min(batch_size, n - start)
        generated = model(torch.randn(b, *space_tensor_shape(cfg), device=device))
        outs.append(decode_space_batch(generated, cfg, codec).detach().cpu())
    return torch.cat(outs, dim=0)


@torch.no_grad()
def sample_meanflow(
    model: nn.Module,
    n: int,
    batch_size: int,
    device: torch.device,
    cfg: MnistConfig,
    codec: Optional[LeNetLatentCodec],
) -> torch.Tensor:
    model.eval()
    outs = []
    for start in range(0, n, batch_size):
        b = min(batch_size, n - start)
        e = torch.randn(b, *space_tensor_shape(cfg), device=device)
        if uses_cnn_pixel_velocity(cfg):
            r = torch.zeros(b, 1, device=device)
            t = torch.ones_like(r)
            u = model(e, torch.cat([r, t], dim=1))
        else:
            t = torch.ones(b, 1, device=device)
            h = torch.ones_like(t)
            u, _ = model(e, t, h)
        outs.append(decode_space_batch(e - u, cfg, codec).detach().cpu())
    return torch.cat(outs, dim=0)


@torch.no_grad()
def sample_shortcut(
    model: nn.Module,
    n: int,
    batch_size: int,
    device: torch.device,
    cfg: MnistConfig,
    codec: Optional[LeNetLatentCodec],
) -> torch.Tensor:
    model.eval()
    outs = []
    for start in range(0, n, batch_size):
        b = min(batch_size, n - start)
        x = torch.randn(b, *space_tensor_shape(cfg), device=device)
        t = torch.zeros(b, 1, device=device)
        if uses_cnn_pixel_velocity(cfg):
            dt = torch.ones_like(t)
            v = model(x, torch.cat([t, dt], dim=1))
        else:
            dt_base = torch.zeros(b, 1, device=device)
            v = model(x, t, dt_base)
        outs.append(decode_space_batch(x + v, cfg, codec).detach().cpu())
    return torch.cat(outs, dim=0)


def prepare_fid_images(x_model: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = from_model_space(x_model.to(device))
    x = x.repeat(1, 3, 1, 1)
    return F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False).clamp(0.0, 1.0)


@torch.no_grad()
def compute_inception_fid_for_sampler(
    sampler: Callable[[int, int], torch.Tensor],
    real_loader: DataLoader,
    cfg: MnistConfig,
    device: torch.device,
    n_samples: Optional[int] = None,
) -> float:
    from torchmetrics.image.fid import FrechetInceptionDistance

    target_n = min(int(n_samples or cfg.fid_samples), len(real_loader.dataset))
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    fid.set_dtype(torch.float64)
    seen = 0
    for batch in real_loader:
        real = batch[0] if isinstance(batch, (tuple, list)) else batch
        real_model = to_model_space(real)
        fid.update(prepare_fid_images(real_model, device), real=True)
        seen += real.shape[0]
        if seen >= target_n:
            break
    fake_seen = 0
    while fake_seen < target_n:
        b = min(cfg.fid_batch_size, target_n - fake_seen)
        fake = sampler(b, b)
        fid.update(prepare_fid_images(fake, device), real=False)
        fake_seen += b
    return float(fid.compute().detach().cpu())


def sqrtm_compat(matrix: np.ndarray) -> np.ndarray:
    try:
        result = linalg.sqrtm(matrix, disp=False)
    except TypeError:
        result = linalg.sqrtm(matrix)
    return result[0] if isinstance(result, tuple) else result


def frechet_distance_np(real: np.ndarray, fake: np.ndarray) -> float:
    mu_r = real.mean(axis=0)
    mu_f = fake.mean(axis=0)
    cov_r = np.cov(real, rowvar=False)
    cov_f = np.cov(fake, rowvar=False)
    eps = 1e-6
    covmean = sqrtm_compat((cov_r + eps * np.eye(cov_r.shape[0])) @ (cov_f + eps * np.eye(cov_f.shape[0])))
    if not np.isfinite(covmean).all():
        covmean = sqrtm_compat((cov_r + 1e-4 * np.eye(cov_r.shape[0])) @ (cov_f + 1e-4 * np.eye(cov_f.shape[0])))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_r - mu_f
    fid = diff.dot(diff) + np.trace(cov_r + cov_f - 2.0 * covmean)
    return float(np.maximum(fid, 0.0))


def train_mnist_feature_net(
    train_loader: DataLoader,
    cfg: MnistConfig,
    device: torch.device,
    outdir: Path,
) -> Tuple[MnistFeatureNet, Dict[str, float]]:
    ckpt = outdir / "lenet5_feature_net.pt"
    model = LeNet5FeatureNet().to(device)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        return model, {"feature_train_seconds": 0.0, "feature_train_acc": float("nan")}

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.classifier_lr, weight_decay=1e-4)
    start = time.time()
    last_acc = 0.0
    for epoch in range(cfg.classifier_epochs):
        total = 0
        correct = 0
        model.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += y.numel()
            correct += int((logits.argmax(dim=1) == y).sum().detach().cpu())
        last_acc = correct / max(total, 1)
        print(f"MNIST feature net epoch {epoch + 1}/{cfg.classifier_epochs} acc={last_acc:.4f}", flush=True)
    torch.save(model.state_dict(), ckpt)
    model.eval()
    return model, {"feature_train_seconds": time.time() - start, "feature_train_acc": last_acc}


@torch.no_grad()
def compute_latent_stats(
    encoder: LeNet5FeatureNet,
    train_loader: DataLoader,
    cfg: MnistConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    latents = []
    seen = 0
    encoder.eval()
    for x, _ in train_loader:
        x = x.to(device, non_blocking=True)
        latents.append(encoder.encode_latent(x).detach())
        seen += x.shape[0]
        if seen >= cfg.latent_stats_samples:
            break
    values = torch.cat(latents, dim=0)[: cfg.latent_stats_samples]
    mean = values.mean(dim=0)
    std = values.std(dim=0).clamp_min(1e-4)
    return mean, std


def train_latent_codec(
    train_loader: DataLoader,
    cfg: MnistConfig,
    device: torch.device,
    outdir: Path,
    feature_net: LeNet5FeatureNet,
) -> Tuple[LeNetLatentCodec, Dict[str, float]]:
    ckpt = outdir / "lenet5_latent_decoder.pt"
    stats_path = outdir / "lenet5_latent_stats.pt"
    decoder = LeNet5Decoder().to(device)
    feature_net.eval()
    for parameter in feature_net.parameters():
        parameter.requires_grad_(False)
    start = time.time()
    last_loss = float("nan")
    if ckpt.exists():
        decoder.load_state_dict(torch.load(ckpt, map_location=device))
    else:
        opt = torch.optim.AdamW(decoder.parameters(), lr=cfg.codec_lr, weight_decay=1e-4)
        for epoch in range(cfg.codec_epochs):
            decoder.train()
            total_loss = 0.0
            total_n = 0
            for x, _ in train_loader:
                x = x.to(device, non_blocking=True)
                with torch.no_grad():
                    latent = feature_net.encode_latent(x)
                recon = decoder(latent)
                loss = F.binary_cross_entropy(recon, x)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total_loss += float(loss.detach().cpu()) * x.shape[0]
                total_n += x.shape[0]
            last_loss = total_loss / max(total_n, 1)
            print(f"LeNet latent decoder epoch {epoch + 1}/{cfg.codec_epochs} BCE={last_loss:.5f}", flush=True)
        torch.save(decoder.state_dict(), ckpt)
    if stats_path.exists():
        stats = torch.load(stats_path, map_location=device)
        mean, std = stats["mean"], stats["std"]
    else:
        mean, std = compute_latent_stats(feature_net, train_loader, cfg, device)
        torch.save({"mean": mean.detach().cpu(), "std": std.detach().cpu()}, stats_path)
    codec = LeNetLatentCodec(feature_net, decoder, mean.to(device), std.to(device)).to(device)
    return codec, {
        "codec_train_seconds": time.time() - start,
        "codec_reconstruction_bce": last_loss,
        "latent_std_min": float(std.min().detach().cpu()),
        "latent_std_median": float(std.median().detach().cpu()),
        "latent_std_max": float(std.max().detach().cpu()),
    }


@torch.no_grad()
def collect_real_features(
    feature_net: MnistFeatureNet,
    loader: DataLoader,
    cfg: MnistConfig,
    device: torch.device,
    n_samples: Optional[int] = None,
) -> np.ndarray:
    target_n = min(int(n_samples or cfg.fid_samples), len(loader.dataset))
    feature_net.eval()
    feats = []
    seen = 0
    for batch in loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        x = x.to(device, non_blocking=True)
        feats.append(feature_net(x, return_features=True).detach().cpu().numpy())
        seen += x.shape[0]
        if seen >= target_n:
            break
    return np.concatenate(feats, axis=0)[:target_n]


@torch.no_grad()
def collect_fake_features(
    sampler: Callable[[int, int], torch.Tensor],
    feature_net: MnistFeatureNet,
    cfg: MnistConfig,
    device: torch.device,
    n_samples: Optional[int] = None,
) -> np.ndarray:
    target_n = int(n_samples or cfg.fid_samples)
    feature_net.eval()
    feats = []
    seen = 0
    while seen < target_n:
        b = min(cfg.fid_batch_size, target_n - seen)
        fake_model = sampler(b, b)
        fake = from_model_space(fake_model).to(device)
        feats.append(feature_net(fake, return_features=True).detach().cpu().numpy())
        seen += b
    return np.concatenate(feats, axis=0)[:target_n]


@torch.no_grad()
def compute_mnist_fid_for_sampler(
    sampler: Callable[[int, int], torch.Tensor],
    real_loader: DataLoader,
    cfg: MnistConfig,
    device: torch.device,
    feature_net: MnistFeatureNet,
    n_samples: Optional[int] = None,
) -> float:
    target_n = min(int(n_samples or cfg.fid_samples), len(real_loader.dataset))
    real = collect_real_features(feature_net, real_loader, cfg, device, n_samples=target_n)
    fake = collect_fake_features(sampler, feature_net, cfg, device, n_samples=target_n)
    return frechet_distance_np(real, fake)


def polynomial_kid_np(real: np.ndarray, fake: np.ndarray, subset_size: int, n_subsets: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    n = min(real.shape[0], fake.shape[0])
    m = min(subset_size, n)
    if m < 2:
        return float("nan")
    dim = real.shape[1]
    vals = []
    for _ in range(n_subsets):
        ri = rng.choice(real.shape[0], size=m, replace=False)
        fi = rng.choice(fake.shape[0], size=m, replace=False)
        x = real[ri].astype(np.float64, copy=False)
        y = fake[fi].astype(np.float64, copy=False)
        k_xx = ((x @ x.T) / dim + 1.0) ** 3
        k_yy = ((y @ y.T) / dim + 1.0) ** 3
        k_xy = ((x @ y.T) / dim + 1.0) ** 3
        sum_xx = (k_xx.sum() - np.trace(k_xx)) / (m * (m - 1))
        sum_yy = (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
        vals.append(sum_xx + sum_yy - 2.0 * k_xy.mean())
    return float(np.mean(vals))


def normalize_features_real_zscore(
    real: np.ndarray,
    fake: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    real64 = real.astype(np.float64, copy=False)
    fake64 = fake.astype(np.float64, copy=False)
    mean = real64.mean(axis=0, keepdims=True)
    std = real64.std(axis=0, keepdims=True)
    std = np.maximum(std, eps)
    real_z = (real64 - mean) / std
    fake_z = (fake64 - mean) / std
    stats = {
        "raw_real_feature_rms": float(np.sqrt(np.mean(real64**2))),
        "raw_fake_feature_rms": float(np.sqrt(np.mean(fake64**2))),
        "normalized_real_feature_rms": float(np.sqrt(np.mean(real_z**2))),
        "normalized_fake_feature_rms": float(np.sqrt(np.mean(fake_z**2))),
        "feature_std_min": float(std.min()),
        "feature_std_median": float(np.median(std)),
        "feature_std_max": float(std.max()),
    }
    return real_z, fake_z, stats


@torch.no_grad()
def compute_mnist_feature_metrics_for_sampler(
    sampler: Callable[[int, int], torch.Tensor],
    real_loader: DataLoader,
    cfg: MnistConfig,
    device: torch.device,
    feature_net: MnistFeatureNet,
    n_samples: Optional[int] = None,
) -> Dict[str, float]:
    target_n = min(int(n_samples or cfg.fid_samples), len(real_loader.dataset))
    real = collect_real_features(feature_net, real_loader, cfg, device, n_samples=target_n)
    fake = collect_fake_features(sampler, feature_net, cfg, device, n_samples=target_n)
    real_z, fake_z, feature_stats = normalize_features_real_zscore(real, fake)
    raw_kid = polynomial_kid_np(real, fake, cfg.kid_subset_size, cfg.kid_subsets, cfg.seed + target_n)
    normalized_kid = polynomial_kid_np(real_z, fake_z, cfg.kid_subset_size, cfg.kid_subsets, cfg.seed + target_n)
    return {
        "mnist_fid": frechet_distance_np(real, fake),
        "mnist_kid": normalized_kid,
        "mnist_kid_x1000": normalized_kid * 1000.0,
        "mnist_kid_raw": raw_kid,
        "mnist_kid_raw_x1000": raw_kid * 1000.0,
        **feature_stats,
    }


@torch.no_grad()
def compute_fid_for_sampler(
    sampler: Callable[[int, int], torch.Tensor],
    real_loader: DataLoader,
    cfg: MnistConfig,
    device: torch.device,
    feature_net: Optional[MnistFeatureNet],
    n_samples: Optional[int] = None,
) -> float:
    if cfg.fid_backend == "mnist":
        if feature_net is None:
            raise ValueError("fid_backend=mnist requires a trained MNIST feature net")
        return compute_mnist_fid_for_sampler(sampler, real_loader, cfg, device, feature_net, n_samples=n_samples)
    if cfg.fid_backend == "inception":
        return compute_inception_fid_for_sampler(sampler, real_loader, cfg, device, n_samples=n_samples)
    raise ValueError(f"Unknown FID backend {cfg.fid_backend}")


@torch.no_grad()
def compute_eval_metrics_for_sampler(
    sampler: Callable[[int, int], torch.Tensor],
    real_loader: DataLoader,
    cfg: MnistConfig,
    device: torch.device,
    feature_net: Optional[MnistFeatureNet],
    n_samples: Optional[int] = None,
) -> Dict[str, float]:
    target_n = min(int(n_samples or cfg.fid_samples), len(real_loader.dataset))
    metrics = {}
    if cfg.fid_backend == "inception":
        metrics["inception_fid"] = compute_inception_fid_for_sampler(
            sampler, real_loader, cfg, device, n_samples=target_n
        )
    else:
        metrics["inception_fid"] = float("nan")
    if cfg.extra_mnist_metrics or cfg.fid_backend == "mnist":
        if feature_net is None:
            raise ValueError("MNIST feature metrics require a trained MNIST feature net")
        metrics.update(compute_mnist_feature_metrics_for_sampler(sampler, real_loader, cfg, device, feature_net, n_samples=target_n))
    metrics["eval_samples"] = float(target_n)
    return metrics


@torch.no_grad()
def selection_score_for_sampler(
    sampler: Callable[[int, int], torch.Tensor],
    real_loader: DataLoader,
    cfg: MnistConfig,
    device: torch.device,
    feature_net: Optional[MnistFeatureNet],
) -> float:
    n_samples = min(cfg.val_n, len(real_loader.dataset))
    if cfg.selection_metric == "inception_fid":
        return compute_inception_fid_for_sampler(sampler, real_loader, cfg, device, n_samples=n_samples)
    if cfg.selection_metric == "mnist_fid":
        if feature_net is None:
            raise ValueError("selection_metric=mnist_fid requires a trained MNIST feature net")
        return compute_mnist_fid_for_sampler(sampler, real_loader, cfg, device, feature_net, n_samples=n_samples)
    if cfg.selection_metric == "mnist_kid":
        if feature_net is None:
            raise ValueError("selection_metric=mnist_kid requires a trained MNIST feature net")
        return compute_mnist_feature_metrics_for_sampler(
            sampler, real_loader, cfg, device, feature_net, n_samples=n_samples
        )["mnist_kid"]
    raise ValueError(f"Unknown selection_metric {cfg.selection_metric}")


@torch.no_grad()
def save_samples_grid(samples: Dict[str, torch.Tensor], outpath: Path, per_method: int = 16) -> None:
    methods = list(samples)
    fig = plt.figure(figsize=(1.8 + 0.9 * per_method, 0.95 * len(methods)))
    grid = fig.add_gridspec(len(methods), per_method + 1, width_ratios=[1.8] + [1.0] * per_method, wspace=0.04, hspace=0.08)
    for row, method in enumerate(methods):
        label_ax = fig.add_subplot(grid[row, 0])
        label_ax.text(0.98, 0.5, method, ha="right", va="center", fontsize=10, fontweight="bold")
        label_ax.axis("off")
        images = from_model_space(samples[method][:per_method]).detach().cpu()
        for col in range(per_method):
            ax = fig.add_subplot(grid[row, col + 1])
            ax.imshow(images[col, 0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.995, top=0.99, bottom=0.01)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def real_samples_from_loader(loader: DataLoader, count: int) -> torch.Tensor:
    xs: List[torch.Tensor] = []
    seen = 0
    for batch in loader:
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        xs.append(to_model_space(x))
        seen += x.shape[0]
        if seen >= count:
            break
    if not xs:
        raise RuntimeError("test loader produced no real samples")
    return torch.cat(xs, dim=0)[:count]


def save_table(rows: List[Dict[str, object]], outpath: Path) -> None:
    methods = [str(r["method"]) for r in rows]
    inception = [float(r.get("inception_fid", r.get("fid", float("nan")))) for r in rows]
    mnist_fids = [float(r.get("mnist_fid", float("nan"))) for r in rows]
    kids = [float(r.get("mnist_kid_x1000", float("nan"))) for r in rows]
    raw_kids = [float(r.get("mnist_kid_raw_x1000", float("nan"))) for r in rows]
    nfe = [int(r.get("nfe", 1)) for r in rows]
    cell_text = [
        [
            m,
            f"{fid:.3f}" if np.isfinite(fid) else "NA",
            f"{mfid:.3f}" if np.isfinite(mfid) else "NA",
            f"{kid:.3f}" if np.isfinite(kid) else "NA",
            f"{raw_kid:.3f}" if np.isfinite(raw_kid) else "NA",
            str(n),
        ]
        for m, fid, mfid, kid, raw_kid, n in zip(methods, inception, mnist_fids, kids, raw_kids, nfe)
    ]
    fig, ax = plt.subplots(figsize=(9.2, 0.55 * len(rows) + 1.2))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=["Method", "Inception-FID", "MNIST-FID", "KID-z x1000", "KID-raw x1000", "NFE"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.35)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1f3a5f")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f1f5f9")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def write_csv(rows: List[Dict[str, object]], outpath: Path) -> None:
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_json_file(path: Path) -> Dict[str, object]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_run_hyperparameters(rows: List[Dict[str, object]], outdir: Path) -> None:
    summaries: List[Dict[str, object]] = []
    flat_rows: List[Dict[str, object]] = []
    for row in rows:
        log_dir = row.get("log_dir")
        manifest: Dict[str, object] = {}
        if log_dir:
            path = Path(str(log_dir))
            if not path.is_absolute():
                path = outdir.parent / path
            manifest = load_json_file(path / "manifest.json")
        hparams = manifest.get("hyperparameters", {}) if isinstance(manifest, dict) else {}
        arch = manifest.get("architecture", {}) if isinstance(manifest, dict) else {}
        extra = manifest.get("extra_hyperparameters", {}) if isinstance(manifest, dict) else {}
        if not isinstance(hparams, dict):
            hparams = {}
        if not isinstance(arch, dict):
            arch = {}
        if not isinstance(extra, dict):
            extra = {}
        pixel_dit = arch.get("pixel_dit", {})
        if not isinstance(pixel_dit, dict):
            pixel_dit = {}
        summary = {
            "method": row.get("method", ""),
            "mnist_fid": row.get("mnist_fid", ""),
            "kid_z_x1000": row.get("mnist_kid_x1000", ""),
            "loss": row.get("loss", ""),
            "log_dir": log_dir or "",
            "architecture": arch,
            "training_hyperparameters": hparams,
            "extra_hyperparameters": extra,
            "selected_or_tuned": {
                key: row.get(key)
                for key in sorted(row.keys())
                if key.startswith("selected_") or key.startswith("tuned_") or key in {"validation_metric", "validation_score"}
            },
        }
        summaries.append(summary)
        flat_rows.append(
            {
                "method": row.get("method", ""),
                "mnist_fid": row.get("mnist_fid", ""),
                "loss": row.get("loss", ""),
                "train_steps": row.get("train_steps", ""),
                "planned_steps": row.get("planned_steps", ""),
                "stopped_early": row.get("stopped_early", ""),
                "stop_step": row.get("stop_step", ""),
                "early_stop_reason": row.get("early_stop_reason", ""),
                "selected_tau": row.get("selected_tau", ""),
                "tuned_lr": row.get("tuned_lr", ""),
                "tuned_meanflow_norm_p": row.get("tuned_meanflow_norm_p", ""),
                "tuned_shortcut_bootstrap_every": row.get("tuned_shortcut_bootstrap_every", ""),
                "tuned_shortcut_ema_decay": row.get("tuned_shortcut_ema_decay", ""),
                "model_class": arch.get("class_name", ""),
                "trainable_parameters": arch.get("trainable_parameters", ""),
                "direct_backbone": arch.get("direct_backbone", ""),
                "model_space": arch.get("model_space", ""),
                "pixel_dit_hidden": pixel_dit.get("hidden_size", ""),
                "pixel_dit_depth": pixel_dit.get("depth", ""),
                "pixel_dit_heads": pixel_dit.get("num_heads", ""),
                "pixel_patch_size": pixel_dit.get("patch_size", ""),
                "pixel_register_tokens": pixel_dit.get("num_register_tokens", ""),
                "learning_rate": hparams.get("learning_rate", ""),
                "weight_decay": hparams.get("weight_decay", ""),
                "grad_clip": hparams.get("grad_clip", ""),
                "batch_size": hparams.get("batch_size", ""),
                "estimated_epochs": hparams.get("estimated_epochs", ""),
                "extra_hyperparameters_json": json.dumps(extra, sort_keys=True),
            }
        )
    (outdir / "run_hyperparameters.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    write_csv(flat_rows, outdir / "run_hyperparameters.csv")


def preset_config(preset: str, seed: int, methods: List[str], model_space: str = "pixel") -> MnistConfig:
    if preset == "smoke":
        return MnistConfig(
            preset=preset,
            model_space=model_space,
            seed=seed,
            train_n=2048,
            val_n=512,
            fid_samples=256,
            steps=30,
            single_steps=20,
            batch_size=64,
            kernel_batch=32,
            lr=2e-4,
            weight_decay=0.01,
            grad_clip=2.0,
            hidden=32,
            num_workers=2,
            all_tau_low=0.35,
            all_tau_high=0.90,
            wbvm_single_taus=[0.3, 0.5, 0.7],
            kernel_scales=[0.5, 1.0, 2.0, 4.0],
            kernel_bandwidth_points=512,
            vector_loss_statistic="u",
            velocity_eps=1e-6,
            meanflow_gap_prob=0.25,
            shortcut_empirical_frac=0.75,
            shortcut_min_steps=128,
            shortcut_ema_decay=0.999,
            drifting_step_size=0.15,
            drifting_temperatures=[0.05, 0.10, 0.20],
            drifting_normalize_distances=True,
            drifting_normalize_drift=True,
            drifting_space="pixel",
            fid_batch_size=64,
            fid_backend="inception",
            selection_metric="mnist_fid",
            extra_mnist_metrics=True,
            kid_subset_size=256,
            kid_subsets=10,
            classifier_epochs=1,
            classifier_lr=1e-3,
            codec_epochs=1,
            codec_lr=1e-3,
            latent_stats_samples=1024,
            latent_hidden=64,
            latent_depth=2,
            direct_backbone="dit",
            pixel_dit_hidden=256,
            pixel_dit_depth=6,
            pixel_dit_heads=4,
            pixel_patch_size=4,
            pixel_dit_mlp_ratio=4.0,
            pixel_dit_register_tokens=8,
            pixel_dit_use_qk_norm=True,
            pixel_dit_use_style_embed=True,
            pixel_dit_style_tokens=32,
            pixel_dit_style_codebook=64,
            meanflow_data_proportion=0.5,
            meanflow_logit_mean=-0.4,
            meanflow_logit_std=1.0,
            meanflow_norm_p=1.0,
            meanflow_norm_eps=0.01,
            shortcut_bootstrap_every=8,
            early_stop_min_steps=0,
            early_stop_patience=0,
            early_stop_min_delta=0.0,
            early_stop_metric="abs_loss",
            methods=methods,
        )
    if preset == "quick":
        return MnistConfig(
            preset=preset,
            model_space=model_space,
            seed=seed,
            train_n=12000,
            val_n=2000,
            fid_samples=2000,
            steps=1000,
            single_steps=600,
            batch_size=192,
            kernel_batch=96,
            lr=2e-4,
            weight_decay=0.01,
            grad_clip=2.0,
            hidden=48,
            num_workers=4,
            all_tau_low=0.35,
            all_tau_high=0.90,
            wbvm_single_taus=[0.3, 0.5, 0.7],
            kernel_scales=[0.5, 1.0, 2.0, 4.0],
            kernel_bandwidth_points=1024,
            vector_loss_statistic="u",
            velocity_eps=1e-6,
            meanflow_gap_prob=0.25,
            shortcut_empirical_frac=0.75,
            shortcut_min_steps=128,
            shortcut_ema_decay=0.999,
            drifting_step_size=0.12,
            drifting_temperatures=[0.05, 0.10, 0.20],
            drifting_normalize_distances=True,
            drifting_normalize_drift=True,
            drifting_space="pixel",
            fid_batch_size=128,
            fid_backend="inception",
            selection_metric="mnist_fid",
            extra_mnist_metrics=True,
            kid_subset_size=512,
            kid_subsets=20,
            classifier_epochs=2,
            classifier_lr=1e-3,
            codec_epochs=3,
            codec_lr=1e-3,
            latent_stats_samples=10000,
            latent_hidden=128,
            latent_depth=4,
            direct_backbone="dit",
            pixel_dit_hidden=256,
            pixel_dit_depth=6,
            pixel_dit_heads=4,
            pixel_patch_size=4,
            pixel_dit_mlp_ratio=4.0,
            pixel_dit_register_tokens=8,
            pixel_dit_use_qk_norm=True,
            pixel_dit_use_style_embed=True,
            pixel_dit_style_tokens=32,
            pixel_dit_style_codebook=64,
            meanflow_data_proportion=0.5,
            meanflow_logit_mean=-0.4,
            meanflow_logit_std=1.0,
            meanflow_norm_p=1.0,
            meanflow_norm_eps=0.01,
            shortcut_bootstrap_every=8,
            early_stop_min_steps=300,
            early_stop_patience=200,
            early_stop_min_delta=1e-4,
            early_stop_metric="abs_loss",
            methods=methods,
        )
    if preset == "standard":
        return MnistConfig(
            preset=preset,
            model_space=model_space,
            seed=seed,
            train_n=50000,
            val_n=5000,
            fid_samples=10000,
            steps=5000,
            single_steps=4000,
            batch_size=256,
            kernel_batch=128,
            lr=2e-4,
            weight_decay=0.01,
            grad_clip=2.0,
            hidden=64,
            num_workers=8,
            all_tau_low=0.35,
            all_tau_high=0.90,
            wbvm_single_taus=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            kernel_scales=[0.5, 1.0, 2.0, 4.0],
            kernel_bandwidth_points=2048,
            vector_loss_statistic="u",
            velocity_eps=1e-6,
            meanflow_gap_prob=0.25,
            shortcut_empirical_frac=0.75,
            shortcut_min_steps=128,
            shortcut_ema_decay=0.999,
            drifting_step_size=0.08,
            drifting_temperatures=[0.05, 0.10, 0.20],
            drifting_normalize_distances=True,
            drifting_normalize_drift=True,
            drifting_space="pixel",
            fid_batch_size=256,
            fid_backend="inception",
            selection_metric="mnist_fid",
            extra_mnist_metrics=True,
            kid_subset_size=1000,
            kid_subsets=20,
            classifier_epochs=3,
            classifier_lr=1e-3,
            codec_epochs=5,
            codec_lr=1e-3,
            latent_stats_samples=50000,
            latent_hidden=256,
            latent_depth=6,
            direct_backbone="dit",
            pixel_dit_hidden=256,
            pixel_dit_depth=6,
            pixel_dit_heads=4,
            pixel_patch_size=4,
            pixel_dit_mlp_ratio=4.0,
            pixel_dit_register_tokens=8,
            pixel_dit_use_qk_norm=True,
            pixel_dit_use_style_embed=True,
            pixel_dit_style_tokens=32,
            pixel_dit_style_codebook=64,
            meanflow_data_proportion=0.5,
            meanflow_logit_mean=-0.4,
            meanflow_logit_std=1.0,
            meanflow_norm_p=1.0,
            meanflow_norm_eps=0.01,
            shortcut_bootstrap_every=8,
            early_stop_min_steps=1200,
            early_stop_patience=500,
            early_stop_min_delta=1e-4,
            early_stop_metric="abs_loss",
            methods=methods,
        )
    raise ValueError(f"Unknown preset {preset}")


def parse_methods(text: str) -> List[str]:
    methods = [m.strip() for m in text.split(",") if m.strip()]
    allowed = {"wbvm_all", "wbvm_single", "wbvm_vector", "meanflow", "shortcut", "drifting"}
    bad = [m for m in methods if m not in allowed]
    if bad:
        raise ValueError(f"Unknown methods {bad}; allowed={sorted(allowed)}")
    return methods


def parse_float_list(text: str) -> List[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one float")
    return vals


def cfg_with(cfg: MnistConfig, **updates: object) -> MnistConfig:
    data = asdict(cfg)
    data.update(updates)
    return MnistConfig(**data)


def tune_meanflow(
    train_loader: InfiniteLoader,
    val_loader: DataLoader,
    cfg: MnistConfig,
    device: torch.device,
    feature_net: Optional[MnistFeatureNet],
    codec: Optional[LeNetLatentCodec],
    log_dir: Optional[Path] = None,
) -> Tuple[nn.Module, Dict[str, object], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    best_model: Optional[nn.Module] = None
    best_info: Dict[str, object] = {}
    best_score = float("inf")
    grid = [
        (1e-4, 1.0),
        (2e-4, 1.0),
        (5e-4, 1.0),
        (5e-4, 0.5),
        (1e-3, 0.5),
        (5e-4, 0.0),
    ]
    for i, (lr, norm_p) in enumerate(grid):
        set_seed(cfg.seed + 2000 + i)
        local_cfg = cfg_with(cfg, lr=lr, meanflow_norm_p=norm_p)
        model, info = train_meanflow(
            train_loader,
            local_cfg,
            device,
            codec,
            log_dir=log_dir,
            log_name=f"MeanFlow-tune-lr-{lr:g}-normp-{norm_p:g}",
            extra_hparams={"tune_lr": lr, "tune_meanflow_norm_p": norm_p},
        )
        score = selection_score_for_sampler(
            lambda n, b, m=model: sample_meanflow(m, n, b, device, cfg, codec), val_loader, cfg, device, feature_net
        )
        row = {
            "method": "MeanFlow",
            "tune_lr": lr,
            "tune_meanflow_norm_p": norm_p,
            "val_metric": cfg.selection_metric,
            "val_score": score,
            **info,
        }
        rows.append(row)
        print(f"MeanFlow tune lr={lr:g} norm_p={norm_p:g} {cfg.selection_metric}={score:.4f}", flush=True)
        if score < best_score:
            if best_model is not None:
                del best_model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            best_model = model
            best_score = score
            best_info = dict(info)
            best_info.update(
                {
                    "tuned_lr": lr,
                    "tuned_meanflow_norm_p": norm_p,
                    "validation_metric": cfg.selection_metric,
                    "validation_score": score,
                }
            )
        else:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if best_model is None:
        raise RuntimeError("MeanFlow tuning produced no model")
    return best_model, best_info, rows


def tune_shortcut(
    train_loader: InfiniteLoader,
    val_loader: DataLoader,
    cfg: MnistConfig,
    device: torch.device,
    feature_net: Optional[MnistFeatureNet],
    codec: Optional[LeNetLatentCodec],
    log_dir: Optional[Path] = None,
) -> Tuple[nn.Module, Dict[str, object], List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []
    best_model: Optional[nn.Module] = None
    best_info: Dict[str, object] = {}
    best_score = float("inf")
    grid = []
    for lr in [1e-4, 2e-4, 5e-4]:
        for bootstrap_every in [2, 4]:
            for ema in [0.99, 0.995]:
                grid.append((lr, bootstrap_every, ema))
    for i, (lr, bootstrap_every, ema) in enumerate(grid):
        set_seed(cfg.seed + 3000 + i)
        local_cfg = cfg_with(cfg, lr=lr, shortcut_bootstrap_every=bootstrap_every, shortcut_ema_decay=ema)
        model, info = train_shortcut(
            train_loader,
            local_cfg,
            device,
            codec,
            log_dir=log_dir,
            log_name=f"ShortcutFlow-tune-lr-{lr:g}-b{bootstrap_every}-ema-{ema:g}",
            extra_hparams={"tune_lr": lr, "tune_shortcut_bootstrap_every": bootstrap_every, "tune_shortcut_ema_decay": ema},
        )
        score = selection_score_for_sampler(
            lambda n, b, m=model: sample_shortcut(m, n, b, device, cfg, codec), val_loader, cfg, device, feature_net
        )
        row = {
            "method": "ShortcutFlow",
            "tune_lr": lr,
            "tune_shortcut_bootstrap_every": bootstrap_every,
            "tune_shortcut_ema_decay": ema,
            "val_metric": cfg.selection_metric,
            "val_score": score,
            **info,
        }
        rows.append(row)
        print(
            f"Shortcut tune lr={lr:g} bootstrap_every={bootstrap_every} ema={ema:g} "
            f"{cfg.selection_metric}={score:.4f}",
            flush=True,
        )
        if score < best_score:
            if best_model is not None:
                del best_model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            best_model = model
            best_score = score
            best_info = dict(info)
            best_info.update(
                {
                    "tuned_lr": lr,
                    "tuned_shortcut_bootstrap_every": bootstrap_every,
                    "tuned_shortcut_ema_decay": ema,
                    "validation_metric": cfg.selection_metric,
                    "validation_score": score,
                }
            )
        else:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if best_model is None:
        raise RuntimeError("ShortcutFlow tuning produced no model")
    return best_model, best_info, rows


def run(args: argparse.Namespace) -> None:
    methods = parse_methods(args.methods)
    cfg = preset_config(args.preset, args.seed, methods, model_space=args.model_space)
    if args.steps is not None:
        cfg.steps = args.steps
    if args.single_steps is not None:
        cfg.single_steps = args.single_steps
    if args.train_n is not None:
        cfg.train_n = args.train_n
    if args.fid_samples is not None:
        cfg.fid_samples = args.fid_samples
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.kernel_batch is not None:
        cfg.kernel_batch = args.kernel_batch
    if args.hidden is not None:
        cfg.hidden = args.hidden
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.wbvm_single_taus is not None:
        cfg.wbvm_single_taus = parse_float_list(args.wbvm_single_taus)
    if args.all_tau_low is not None:
        cfg.all_tau_low = args.all_tau_low
    if args.all_tau_high is not None:
        cfg.all_tau_high = args.all_tau_high
    if args.fid_backend is not None:
        cfg.fid_backend = args.fid_backend
    if args.selection_metric is not None:
        cfg.selection_metric = args.selection_metric
    if args.vector_loss_statistic is not None:
        cfg.vector_loss_statistic = args.vector_loss_statistic
    if args.direct_backbone is not None:
        cfg.direct_backbone = args.direct_backbone
    if args.early_stop_min_steps is not None:
        cfg.early_stop_min_steps = args.early_stop_min_steps
    if args.early_stop_patience is not None:
        cfg.early_stop_patience = args.early_stop_patience
    if args.early_stop_min_delta is not None:
        cfg.early_stop_min_delta = args.early_stop_min_delta
    if args.early_stop_metric is not None:
        cfg.early_stop_metric = args.early_stop_metric
    if args.no_extra_mnist_metrics:
        cfg.extra_mnist_metrics = False
    cfg.drifting_space = cfg.model_space

    runtime = configure_runtime(args.num_threads)
    set_seed(cfg.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log_dir = outdir / "training_logs"
    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "runtime": runtime, "device": str(device), "args": vars(args)}, f, indent=2)

    print(
        f"Running MNIST FID pilot on {device}; preset={cfg.preset}; model_space={cfg.model_space}; methods={cfg.methods}",
        flush=True,
    )
    print(f"FID backend: {cfg.fid_backend}", flush=True)
    print(f"Runtime: {runtime}", flush=True)
    train_loader, val_loader, test_loader = make_loaders(cfg, Path(args.data_dir))
    infinite = InfiniteLoader(train_loader)
    feature_net: Optional[MnistFeatureNet] = None
    need_feature_net = (
        cfg.extra_mnist_metrics
        or cfg.selection_metric in {"mnist_fid", "mnist_kid"}
        or cfg.fid_backend == "mnist"
        or cfg.model_space == "latent"
    )
    cache_dir = Path(args.data_dir) / "model_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if need_feature_net:
        feature_net, feature_info = train_mnist_feature_net(train_loader, cfg, device, cache_dir)
        print(
            f"MNIST feature FID net ready: acc={feature_info['feature_train_acc']:.4f}, "
            f"seconds={feature_info['feature_train_seconds']:.1f}",
            flush=True,
        )
    codec: Optional[LeNetLatentCodec] = None
    if cfg.model_space == "latent":
        if feature_net is None:
            raise RuntimeError("Latent experiments require the shared LeNet-5 feature extractor")
        codec, codec_info = train_latent_codec(train_loader, cfg, device, cache_dir, feature_net)
        print(
            f"LeNet latent codec ready: BCE={codec_info['codec_reconstruction_bce']:.5f}, "
            f"seconds={codec_info['codec_train_seconds']:.1f}",
            flush=True,
        )

    rows: List[Dict[str, object]] = []
    tuning_rows: List[Dict[str, object]] = []
    sample_grid: Dict[str, torch.Tensor] = {}
    primary_fid_key = "mnist_fid" if cfg.fid_backend == "mnist" else "inception_fid"

    trainers = {
        "wbvm_all": (
            "WBVM-all",
            lambda: train_wbvm_all(infinite, cfg, device, codec, log_dir=log_dir),
            lambda model, n, b: sample_direct(model, n, b, device, cfg, codec),
        ),
        "wbvm_vector": (
            "vMMD-WBVM",
            lambda: train_wbvm_vector(infinite, cfg, device, codec, log_dir=log_dir),
            lambda model, n, b: sample_direct(model, n, b, device, cfg, codec),
        ),
        "meanflow": (
            "MeanFlow",
            lambda: train_meanflow(infinite, cfg, device, codec, log_dir=log_dir),
            lambda model, n, b: sample_meanflow(model, n, b, device, cfg, codec),
        ),
        "shortcut": (
            "ShortcutFlow",
            lambda: train_shortcut(infinite, cfg, device, codec, log_dir=log_dir),
            lambda model, n, b: sample_shortcut(model, n, b, device, cfg, codec),
        ),
        "drifting": (
            "Drifting",
            lambda: train_drifting(infinite, cfg, device, codec, log_dir=log_dir),
            lambda model, n, b: sample_direct(model, n, b, device, cfg, codec),
        ),
    }

    if "wbvm_single" in cfg.methods:
        model, info, _ = train_wbvm_single(infinite, val_loader, cfg, device, outdir, feature_net, codec, log_dir=log_dir)
        sampler = lambda n, b, m=model: sample_direct(m, n, b, device, cfg, codec)
        metrics = compute_eval_metrics_for_sampler(lambda n, b: sampler(n, b), test_loader, cfg, device, feature_net)
        samples = sampler(32, min(32, cfg.fid_batch_size))
        row = {"method": "WBVM-single", "fid": metrics[primary_fid_key], **metrics, **info}
        rows.append(row)
        sample_grid["WBVM-single"] = samples
        print(
            f"WBVM-single Inception-FID={metrics['inception_fid']:.3f} "
            f"MNIST-FID={metrics.get('mnist_fid', float('nan')):.3f}",
            flush=True,
        )

    if "wbvm_vector" in cfg.methods:
        model, info, candidate_rows = train_wbvm_vector_single(
            infinite,
            val_loader,
            cfg,
            device,
            outdir,
            feature_net,
            codec,
            log_dir=log_dir,
        )
        tuning_rows.extend(candidate_rows)
        sampler = lambda n, b, m=model: sample_direct(m, n, b, device, cfg, codec)
        metrics = compute_eval_metrics_for_sampler(lambda n, b: sampler(n, b), test_loader, cfg, device, feature_net)
        samples = sampler(32, min(32, cfg.fid_batch_size))
        row = {"method": "vMMD-WBVM", "fid": metrics[primary_fid_key], **metrics, **info}
        rows.append(row)
        sample_grid["vMMD-WBVM"] = samples
        print(
            f"vMMD-WBVM  Inception-FID={metrics['inception_fid']:.3f} "
            f"MNIST-FID={metrics.get('mnist_fid', float('nan')):.3f}",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for key in [m for m in cfg.methods if m not in {"wbvm_single", "wbvm_vector"}]:
        label, trainer, sample_fn = trainers[key]
        if args.tune_baselines and key == "meanflow":
            model, info, candidate_rows = tune_meanflow(infinite, val_loader, cfg, device, feature_net, codec, log_dir=log_dir)
            tuning_rows.extend(candidate_rows)
        elif args.tune_baselines and key == "shortcut":
            model, info, candidate_rows = tune_shortcut(infinite, val_loader, cfg, device, feature_net, codec, log_dir=log_dir)
            tuning_rows.extend(candidate_rows)
        else:
            model, info = trainer()
        sampler = lambda n, b, m=model, fn=sample_fn: fn(m, n, b)
        metrics = compute_eval_metrics_for_sampler(lambda n, b: sampler(n, b), test_loader, cfg, device, feature_net)
        samples = sampler(32, min(32, cfg.fid_batch_size))
        row = {"method": label, "fid": metrics[primary_fid_key], **metrics, **info}
        rows.append(row)
        sample_grid[label] = samples
        print(
            f"{label:12s} Inception-FID={metrics['inception_fid']:.3f} "
            f"MNIST-FID={metrics.get('mnist_fid', float('nan')):.3f} "
            f"train_seconds={info['train_seconds']:.1f}",
            flush=True,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows = sorted(rows, key=lambda r: float(r["fid"]))
    write_csv(rows, outdir / "metrics_summary.csv")
    write_run_hyperparameters(rows, outdir)
    if tuning_rows:
        write_csv(tuning_rows, outdir / "baseline_tuning.csv")
    save_table(rows, outdir / "mnist_fid_table.png")
    sample_grid["Real"] = real_samples_from_loader(test_loader, 32)
    torch.save(sample_grid, outdir / "sample_grid.pt")
    save_samples_grid(sample_grid, outdir / "mnist_samples_grid.png")
    print(f"Saved outputs to {outdir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="MNIST FID pilot for RKHS-WBVM and one-step flow baselines")
    parser.add_argument("--preset", default="quick", choices=["smoke", "quick", "standard"])
    parser.add_argument("--model-space", default="pixel", choices=["pixel", "latent"])
    parser.add_argument("--outdir", default="outputs_mnist_quick")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--methods", default="wbvm_single,wbvm_vector,meanflow,shortcut,drifting")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--single-steps", type=int, default=None)
    parser.add_argument("--train-n", type=int, default=None)
    parser.add_argument("--fid-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--kernel-batch", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--wbvm-single-taus", default=None)
    parser.add_argument("--all-tau-low", type=float, default=None)
    parser.add_argument("--all-tau-high", type=float, default=None)
    parser.add_argument("--fid-backend", choices=["inception", "mnist"], default=None)
    parser.add_argument("--selection-metric", choices=["inception_fid", "mnist_fid", "mnist_kid"], default=None)
    parser.add_argument("--vector-loss-statistic", choices=["u", "v"], default=None)
    parser.add_argument("--direct-backbone", choices=["dit", "cnn"], default=None)
    parser.add_argument("--early-stop-min-steps", type=int, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--early-stop-min-delta", type=float, default=None)
    parser.add_argument("--early-stop-metric", choices=["loss", "abs_loss"], default=None)
    parser.add_argument("--no-extra-mnist-metrics", action="store_true")
    parser.add_argument("--tune-baselines", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
