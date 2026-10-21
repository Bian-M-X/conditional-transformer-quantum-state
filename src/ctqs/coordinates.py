"""Affine-coordinate controls; a physical-coordinate Adam replay, not a new optimizer."""
import math
import torch
from .models import ConditionalMLP, ConditionalTransformer


def lambda_weight(model):
    if isinstance(model, ConditionalMLP):
        return model.net[0].weight
    if isinstance(model, ConditionalTransformer):
        return model.condition.weight
    raise TypeError('unsupported model: lambda must enter a known affine column')


def validate_scale(scale):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('scale must be finite and positive')


@torch.no_grad()
def compensate_initialization(model, scale):
    validate_scale(scale)
    lambda_weight(model)[:, -1].mul_(scale)


@torch.no_grad()
def canonical_adam_step(optimizer, model, scale):
    """Call after backward, before zero_grad. Moments are in physical coordinates.

    No closures, clipping, weight decay, or gradient accumulation. Transform only
    the lambda column; all other parameters and their gradients remain unchanged.
    The caller must use this convention from the first optimizer step onward.
    """
    validate_scale(scale)
    if type(optimizer) is not torch.optim.Adam:
        raise TypeError('this control requires plain torch.optim.Adam')
    if any(g['weight_decay'] != 0 or g.get('differentiable', False)
           for g in optimizer.param_groups):
        raise ValueError('control requires zero weight decay and nondifferentiable Adam')
    weight = lambda_weight(model)
    if weight.grad is None:
        raise ValueError('backward must populate the lambda weight gradient')
    if not any(weight is p for g in optimizer.param_groups for p in g['params']):
        raise ValueError('lambda weight missing from optimizer')
    weight[:, -1].div_(scale)
    weight.grad[:, -1].mul_(scale)
    try:
        optimizer.step()
    finally:
        weight[:, -1].mul_(scale)
        weight.grad[:, -1].div_(scale)
