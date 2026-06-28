import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from mnist_fid_experiment import (
    EarlyStopper,
    IMAGE_SHAPE,
    InfiniteLoader,
    LATENT_SHAPE,
    LeNet5Decoder,
    LeNet5FeatureNet,
    LeNetLatentCodec,
    PixelDiT,
    make_training_logger,
    preset_config,
    normalize_features_real_zscore,
    polynomial_kid_np,
    save_samples_grid,
    train_drifting,
    train_meanflow,
    train_shortcut,
    train_wbvm_all,
    train_wbvm_vector,
    vector_flux_mmd2,
    vector_flux_kernel_bilinear_means,
)


class MnistExperimentV2Tests(unittest.TestCase):
    def test_real_zscore_normalization_controls_feature_scale(self) -> None:
        rng = np.random.default_rng(7)
        real = rng.normal(size=(128, 16)) * 50.0 + 100.0
        fake = real + rng.normal(scale=2.0, size=real.shape)
        real_z, fake_z, stats = normalize_features_real_zscore(real, fake)

        self.assertTrue(np.allclose(real_z.mean(axis=0), 0.0, atol=1e-7))
        self.assertTrue(np.allclose(real_z.std(axis=0), 1.0, atol=1e-6))
        self.assertGreater(stats["raw_real_feature_rms"], 10.0)
        raw_kid = polynomial_kid_np(real, fake, 64, 4, 1)
        normalized_kid = polynomial_kid_np(real_z, fake_z, 64, 4, 1)
        self.assertLess(abs(normalized_kid), abs(raw_kid))

    def test_lenet_latent_shape(self) -> None:
        model = LeNet5FeatureNet()
        x = torch.rand(3, 1, 28, 28)
        latent = model.encode_latent(x)
        features = model(x, return_features=True)
        self.assertEqual(tuple(latent.shape[1:]), LATENT_SHAPE)
        self.assertEqual(features.shape, (3, 84))

        resized_latent = model.encode_latent(torch.rand(3, *IMAGE_SHAPE))
        self.assertEqual(tuple(resized_latent.shape[1:]), LATENT_SHAPE)

    def test_pixel_dit_dual_head_shapes(self) -> None:
        model = PixelDiT(hidden_size=32, depth=2, num_heads=4, patch_size=4, dual_head=True)
        x = torch.randn(2, *IMAGE_SHAPE)
        t = torch.rand(2, 1)
        h = torch.rand(2, 1)
        u, v = model(x, t, h)
        self.assertEqual(u.shape, x.shape)
        self.assertEqual(v.shape, x.shape)

    def test_standard_preset_uses_driftdittiny_mnist_shape(self) -> None:
        cfg = preset_config("standard", 7, ["drifting"], "pixel")
        self.assertEqual(IMAGE_SHAPE, (1, 32, 32))
        self.assertEqual(cfg.direct_backbone, "dit")
        self.assertEqual(cfg.pixel_dit_hidden, 256)
        self.assertEqual(cfg.pixel_dit_depth, 6)
        self.assertEqual(cfg.pixel_dit_heads, 4)
        self.assertEqual(cfg.pixel_patch_size, 4)
        self.assertEqual(cfg.pixel_dit_register_tokens, 8)
        self.assertTrue(cfg.pixel_dit_use_qk_norm)
        self.assertTrue(cfg.pixel_dit_use_style_embed)
        self.assertEqual(cfg.pixel_dit_style_tokens, 32)
        self.assertEqual(cfg.pixel_dit_style_codebook, 64)
        self.assertEqual(cfg.weight_decay, 0.01)
        self.assertEqual(cfg.grad_clip, 2.0)

    def test_training_logger_writes_manifest_and_step_files(self) -> None:
        cfg = preset_config("smoke", 7, ["drifting"], "pixel")
        cfg.train_n = 16
        cfg.batch_size = 8
        cfg.pixel_dit_hidden = 32
        cfg.pixel_dit_depth = 2
        cfg.pixel_dit_heads = 4
        model = PixelDiT(
            hidden_size=cfg.pixel_dit_hidden,
            depth=cfg.pixel_dit_depth,
            num_heads=cfg.pixel_dit_heads,
            patch_size=cfg.pixel_patch_size,
            dual_head=False,
            image_size=IMAGE_SHAPE[-1],
            in_channels=IMAGE_SHAPE[0],
            mlp_ratio=cfg.pixel_dit_mlp_ratio,
            num_register_tokens=cfg.pixel_dit_register_tokens,
            use_qk_norm=cfg.pixel_dit_use_qk_norm,
            use_style_embed=cfg.pixel_dit_use_style_embed,
            style_tokens=cfg.pixel_dit_style_tokens,
            style_codebook=cfg.pixel_dit_style_codebook,
        )
        with tempfile.TemporaryDirectory() as tmp:
            logger = make_training_logger(Path(tmp), "Drifting", cfg, model, {"example": True})
            self.assertIsNotNone(logger)
            assert logger is not None
            logger.log_step(1, cfg, loss=0.25, lr=cfg.lr, metrics={"grad_norm": 1.5})
            logger.close({"loss": 0.25})
            manifest = json.loads((logger.dir / "manifest.json").read_text(encoding="utf-8"))
            steps = (logger.dir / "steps.jsonl").read_text(encoding="utf-8").strip().splitlines()
            summary = json.loads((logger.dir / "summary.json").read_text(encoding="utf-8"))

            self.assertEqual(manifest["architecture"]["pixel_dit"]["hidden_size"], 32)
            self.assertEqual(manifest["hyperparameters"]["learning_rate"], cfg.lr)
            self.assertEqual(len(steps), 1)
            self.assertIn('"loss": 0.25', steps[0])
            self.assertEqual(summary["final_metrics"]["loss"], 0.25)

    def test_vector_flux_mmd_identical_flux_is_zero(self) -> None:
        torch.manual_seed(7)
        x = torch.randn(12, 5)
        v = torch.randn(12, 5)
        sigmas = torch.tensor([0.5, 1.0, 2.0])

        loss = vector_flux_mmd2(x, v, x, v, sigmas=sigmas, include_data_data=True, statistic="v")

        self.assertLess(abs(float(loss)), 1e-6)

    def test_vector_flux_mmd_detects_velocity_mismatch(self) -> None:
        torch.manual_seed(11)
        x = torch.randn(12, 4)
        v = torch.randn(12, 4)
        sigmas = torch.tensor([0.75, 1.5])

        loss = vector_flux_mmd2(x, v, x, -v, sigmas=sigmas, include_data_data=True, statistic="v")

        self.assertGreater(float(loss), 0.1)

    def test_vector_flux_full_mmd_is_nonnegative_for_mismatch(self) -> None:
        torch.manual_seed(13)
        x = torch.randn(8, 3)
        y = torch.randn(8, 3)
        v = torch.randn(8, 3)
        w = torch.randn(8, 3)
        sigmas = torch.tensor([0.75])

        loss = vector_flux_mmd2(x, v, y, w, sigmas=sigmas, include_data_data=True, statistic="u")

        self.assertGreaterEqual(float(loss), -1e-6)

    def test_vector_flux_kernel_uses_scalar_rbf_identity_matrix(self) -> None:
        x = torch.tensor([[0.0, 0.0]])
        y = torch.tensor([[1.0, 2.0]])
        v = torch.tensor([[2.0, 3.0]])
        w = torch.tensor([[5.0, 7.0]])
        sigmas = torch.tensor([1.0])

        value = vector_flux_kernel_bilinear_means(x, v, y, w, sigmas)
        expected = (2.0 * 5.0 + 3.0 * 7.0) * torch.exp(torch.tensor(-2.5))
        componentwise_diagonal_value = (
            2.0 * 5.0 * torch.exp(torch.tensor(-0.5)) + 3.0 * 7.0 * torch.exp(torch.tensor(-2.0))
        )

        self.assertTrue(torch.allclose(value.squeeze(), expected, atol=1e-6))
        self.assertFalse(torch.allclose(value.squeeze(), componentwise_diagonal_value, atol=1e-6))

    def test_early_stopper_detects_flat_loss_window(self) -> None:
        cfg = preset_config("smoke", 7, ["meanflow"], "pixel")
        cfg.early_stop_min_steps = 4
        cfg.early_stop_patience = 4
        cfg.early_stop_min_delta = 1e-3
        stopper = EarlyStopper(cfg)

        stopped = False
        for step, loss in enumerate([1.0000, 0.9999, 1.0001, 1.0000], start=1):
            stopped = stopper.update(step, loss)

        self.assertTrue(stopped)
        self.assertEqual(stopper.stop_step, 4)
        self.assertIn("relative change", stopper.reason)

    def test_quick_and_standard_presets_disable_early_stopping(self) -> None:
        for preset in ("quick", "standard"):
            cfg = preset_config(preset, 7, ["wbvm_single", "wbvm_vector"], "pixel")
            self.assertEqual(cfg.early_stop_min_steps, 0)
            self.assertEqual(cfg.early_stop_patience, 0)
            self.assertEqual(cfg.early_stop_min_delta, 0.0)
            self.assertEqual(cfg.vmmd_early_stop_min_steps, 0)
            self.assertEqual(cfg.vmmd_early_stop_patience, 0)
            self.assertEqual(cfg.vmmd_early_stop_min_delta, 0.0)

    def test_sample_grid_reserves_label_column(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "grid.png"
            samples = {"A-long-method-name": torch.zeros(4, 1, 28, 28)}
            save_samples_grid(samples, path, per_method=4)
            with Image.open(path) as image:
                self.assertGreater(image.width / image.height, 3.0)

    def test_all_methods_take_one_step_in_pixel_and_latent_space(self) -> None:
        images = torch.rand(8, *IMAGE_SHAPE)
        labels = torch.arange(8) % 10
        loader = InfiniteLoader(DataLoader(TensorDataset(images, labels), batch_size=8, drop_last=True))
        encoder = LeNet5FeatureNet()
        codec = LeNetLatentCodec(
            encoder,
            LeNet5Decoder(),
            torch.zeros(LATENT_SHAPE),
            torch.ones(LATENT_SHAPE),
        )
        for model_space in ("pixel", "latent"):
            cfg = preset_config("smoke", 3, ["wbvm_all", "wbvm_vector", "meanflow", "shortcut", "drifting"], model_space)
            cfg.steps = 1
            cfg.batch_size = 8
            cfg.kernel_batch = 4
            cfg.kernel_bandwidth_points = 8
            cfg.pixel_dit_hidden = 16
            cfg.pixel_dit_depth = 2
            cfg.pixel_dit_heads = 2
            if model_space == "pixel":
                cfg.direct_backbone = "cnn"
                cfg.hidden = 8
            active_codec = codec if model_space == "latent" else None
            train_wbvm_all(loader, cfg, torch.device("cpu"), active_codec)
            train_wbvm_vector(loader, cfg, torch.device("cpu"), active_codec)
            train_meanflow(loader, cfg, torch.device("cpu"), active_codec)
            train_shortcut(loader, cfg, torch.device("cpu"), active_codec)
            train_drifting(loader, cfg, torch.device("cpu"), active_codec)



if __name__ == "__main__":
    unittest.main()
