import unittest

import torch

from wbvm_vector_mmd_experiment import vector_flux_mmd2


class VectorFluxMMDTests(unittest.TestCase):
    def test_identical_flux_batches_have_zero_loss(self) -> None:
        torch.manual_seed(7)
        x = torch.randn(16, 3)
        v = torch.randn(16, 3)
        sigmas = torch.tensor([0.5, 1.0, 2.0])

        loss = vector_flux_mmd2(x, v, x, v, sigmas=sigmas)

        self.assertLess(abs(float(loss)), 1e-6)

    def test_velocity_mismatch_is_detected_without_moving_points(self) -> None:
        torch.manual_seed(11)
        x = torch.randn(16, 2)
        v = torch.randn(16, 2)
        sigmas = torch.tensor([0.75, 1.5])

        loss = vector_flux_mmd2(x, v, x, -v, sigmas=sigmas)

        self.assertGreater(float(loss), 0.1)


if __name__ == "__main__":
    unittest.main()
