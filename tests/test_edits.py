import unittest
import torch
from wsp.edits import edited_mlp


class GatedMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(13, 29, bias=False, dtype=torch.float64)
        self.up_proj = torch.nn.Linear(13, 29, bias=False, dtype=torch.float64)
        self.down_proj = torch.nn.Linear(29, 13, bias=False, dtype=torch.float64)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class EditTests(unittest.TestCase):
    def test_preservation_and_exception_restore(self):
        torch.manual_seed(20261103)
        model = GatedMLP()
        initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
        x = torch.randn(7, 11, 13, dtype=torch.float64)
        expected = model(x)
        op = {"units": [1, 7, 13, 21], "factors": [8., .125, -4., -.5]}
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with edited_mlp(model, op):
                torch.testing.assert_close(model(x), expected, rtol=1e-12, atol=1e-12)
                self.assertTrue(torch.equal(initial['gate_proj.weight'], model.gate_proj.weight))
                raise RuntimeError("sentinel")
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, initial[name]))

    def test_reject_invalid_edits(self):
        model = GatedMLP()
        for op in [{"units": [1,1], "factors": [2,2]},
                   {"units": [29], "factors": [2]},
                   {"units": [1], "factors": [0]},
                   {"units": [1], "factors": [3]}]:
            with self.assertRaises(ValueError):
                with edited_mlp(model, op):
                    pass


if __name__ == '__main__':
    unittest.main()

