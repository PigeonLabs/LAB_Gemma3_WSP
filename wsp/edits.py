"""Exact-real-arithmetic symmetries of Gemma's gated MLP.

MLP(x) = D [gelu(Gx) * Ux]. Scale a row of U by s and the
matching column of D by 1/s; G is NOT rescaled. This works for
GELU without assuming positive homogeneity of the activation.
Floating-point equivalence is an independent empirical check.
"""
from contextlib import contextmanager
import math
import torch


def validate_operation(mlp, operation):
    units = operation["units"]
    factors = operation["factors"]
    if not units or len(units) != len(factors) or len(set(units)) != len(units):
        raise ValueError("Units must be nonempty, unique and match factors")
    if any(type(i) is not int or i < 0 or i >= mlp.up_proj.weight.shape[0] for i in units):
        raise ValueError("Unit outside intermediate dimension")
    for s in factors:
        if not math.isfinite(s) or s == 0 or not -3 <= math.log2(abs(s)) <= 3:
            raise ValueError("Scale must be a nonzero signed power of two in [1/8,8]")
        if not math.log2(abs(s)).is_integer():
            raise ValueError("Scale must be a power of two")


@contextmanager
def edited_mlp(mlp, operation):
    validate_operation(mlp, operation)
    ids = torch.tensor(operation["units"], device=mlp.up_proj.weight.device)
    factors = torch.tensor(operation["factors"], device=ids.device,
                           dtype=mlp.up_proj.weight.dtype)
    with torch.no_grad():
        old_up = mlp.up_proj.weight.index_select(0, ids).clone()
        old_down = mlp.down_proj.weight.index_select(1, ids).clone()
        try:
            new_up = old_up * factors[:, None]
            new_down = old_down / factors[None, :]
            if not torch.isfinite(new_up).all() or not torch.isfinite(new_down).all():
                raise ValueError("Nonfinite edited weights")
            mlp.up_proj.weight.index_copy_(0, ids, new_up)
            mlp.down_proj.weight.index_copy_(1, ids, new_down)
            yield
        finally:
            mlp.up_proj.weight.index_copy_(0, ids, old_up)
            mlp.down_proj.weight.index_copy_(1, ids, old_down)


def text_layers(model):
    candidates = [(name, module) for name, module in model.named_modules()
                  if name.endswith(".mlp") and all(hasattr(module, x) for x in
                      ("gate_proj", "up_proj", "down_proj"))]
    if not candidates:
        raise ValueError("No Gemma gated text MLPs found")
    return candidates

