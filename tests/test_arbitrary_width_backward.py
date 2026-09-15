"""Exercise the production lane-masked dispatch against the released kernel."""
import unittest
from unittest import mock

import torch
import torch.nn.functional as F


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA accelerator")
class ArbitraryWidthBackwardTest(unittest.TestCase):
    def test_masked_dispatch_matches_zero_padded_released_kernel(self):
        from lingua.sparse_delta_memory import memory_ops
        from access_curriculum import arbitrary_width_kernel
        from access_curriculum.arbitrary_width import install_arbitrary_access_width_backward

        accepted = getattr(memory_ops, "_topk_arbitrary_width_accepted", memory_ops.fused_bwd_elementwise)
        generator = torch.Generator(device="cuda").manual_seed(71)
        with mock.patch.object(memory_ops, "_topk_arbitrary_width_installed", False, create=True), \
                mock.patch.object(memory_ops, "fused_bwd_elementwise", accepted), \
                mock.patch.object(arbitrary_width_kernel, "masked_fused_bwd_elementwise",
                                  wraps=arbitrary_width_kernel.masked_fused_bwd_elementwise) as masked:
            install_arbitrary_access_width_backward()
            for writes, reads in ((12, 12), (24, 24), (12, 24), (24, 12), (128, 32), (8, 8)):
                for dtype in (torch.float32, torch.bfloat16):
                    with self.subTest(writes=writes, reads=reads, dtype=dtype):
                        widths = [writes] * 8 + [reads] * 4
                        inputs = [torch.randn(2, 5, width, generator=generator, device="cuda") * .1
                                  for width in widths]
                        inputs[2] = inputs[2].to(dtype)
                        inputs[9] = inputs[9].to(dtype)
                        pw, pr = 1 << (writes - 1).bit_length(), 1 << (reads - 1).bit_length()
                        padded = [F.pad(x, (0, (pw if i < 8 else pr) - x.shape[-1]))
                                  for i, x in enumerate(inputs)]
                        output_widths = [writes] * 4 + [reads] * 2
                        outputs = [torch.full((2, 5, w), float("nan"), device="cuda",
                                              dtype=dtype if i in (0, 4) else torch.float32)
                                   for i, w in enumerate(output_widths)]
                        references = [torch.empty((2, 5, pw if i < 4 else pr), device="cuda", dtype=x.dtype)
                                      for i, x in enumerate(outputs)]
                        accepted(*padded, *references, pw, pr)
                        before = masked.call_count
                        memory_ops.fused_bwd_elementwise(*inputs, *outputs, writes, reads)
                        self.assertEqual(masked.call_count - before, int((writes, reads) != (pw, pr)))
                        for observed, reference in zip(outputs, references):
                            torch.testing.assert_close(observed, reference[..., :observed.shape[-1]],
                                                       rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
