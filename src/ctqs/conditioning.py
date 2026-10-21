"""Domain scaling and differentiable amplitude covariance; no spectral inputs."""
import torch


def conditioned_state(model, bits, h, lam, scale=1., covariant=False):
    if scale <= 0:
        raise ValueError('positive domain scale required')
    direct=model.psi(bits,[h,lam/scale])
    if not covariant:
        return direct,1
    opposite=model.psi(bits,[h,-lam/scale]) if lam else direct
    result=direct+opposite.flip(0)
    return result/torch.linalg.vector_norm(result),2 if lam else 1
