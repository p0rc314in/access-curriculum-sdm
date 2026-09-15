"""Guard the numerical state used by both training and checkpoint reconstruction."""
import unittest
from unittest.mock import patch

import torch

from reproduction.initialization import initialization_sha256, verify_initialization


class InitializationTest(unittest.TestCase):
    def model(self):
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.tensor([1.001]))
        model.route = torch.nn.Parameter(torch.tensor([.25]), requires_grad=False)
        model.prior = torch.nn.Parameter(torch.tensor([.5]))
        model.register_buffer("rotary", torch.tensor([.75]), persistent=False)
        return model

    def test_discarded_fp32_bits_do_not_change_training_identity(self):
        left, right = self.model(), self.model()
        with torch.no_grad():
            right.weight.add_(1e-7)
        self.assertNotEqual(initialization_sha256(left), initialization_sha256(right))
        self.assertEqual(initialization_sha256(left.bfloat16()), initialization_sha256(right.bfloat16()))

    def test_rejects_changed_weight_route_prior_or_nonpersistent_buffer(self):
        key = ("recall", "BBBBBBBB", "unmodified")
        reference = initialization_sha256(self.model().bfloat16())
        with patch("reproduction.initialization.BF16_STATE_SHA256", {key: reference}):
            for name in ("weight", "route", "prior", "rotary"):
                model = self.model().bfloat16()
                verify_initialization(model, task=key[0], layout=key[1], router=key[2])
                with torch.no_grad():
                    getattr(model, name).add_(.03125)
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "identity changed"):
                    verify_initialization(model, task=key[0], layout=key[1], router=key[2])

    def test_fingerprint_includes_shape_dtype_and_preserves_state(self):
        model = self.model().bfloat16()
        rng = torch.get_rng_state().clone()
        reference = initialization_sha256(model)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(reference, initialization_sha256(model))
        model.weight = torch.nn.Parameter(model.weight.reshape(1, 1))
        self.assertNotEqual(reference, initialization_sha256(model))
        model.weight = torch.nn.Parameter(model.weight.reshape(1).float())
        self.assertNotEqual(reference, initialization_sha256(model))
