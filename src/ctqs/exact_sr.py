"""Real normalized-state SR by exact enumeration; not a stochastic VMC solver."""
import math
import time
import torch
from torch import nn
from torch.func import functional_call, jacrev, vmap
from torch.nn.attention import sdpa_kernel, SDPBackend


class Amplitude(nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model

    def forward(self, bits, condition):
        return self.model.psi(bits, condition)


def amplitude_jacobian(model, bits, condition):
    """One-state reverse derivatives vmapped over configurations, no H backward."""
    wrapper = Amplitude(model)
    params = dict(wrapper.named_parameters())
    buffers = dict(wrapper.named_buffers())
    def single(p, bit):
        value = functional_call(wrapper, (p, buffers), (bit[None], condition))[0]
        return value, value
    # The math attention kernel supports batched differentiation on CPU.
    with sdpa_kernel(SDPBackend.MATH):
        blocks, psi = vmap(jacrev(single, has_aux=True), in_dims=(None, 0), chunk_size=32)(params, bits)
    J = torch.cat([blocks[k].reshape(len(bits), -1) for k in params], dim=1)
    return psi.detach(), J.detach()


def tangent_system(states, jacobians, matrices, n):
    """A.T A = mean real QGT, A.T b = half gradient of mean E/n."""
    aa, bb, energies = [], [], []
    for psi, J, H in zip(states, jacobians, matrices):
        if abs(float(psi @ psi)-1) > 1e-9:
            raise ValueError('SR requires normalized amplitudes')
        centered = J - psi[:, None] * (psi @ J)[None, :]
        hp = H @ psi; energy = psi @ hp
        aa.append(centered); bb.append((hp-energy*psi)/n); energies.append(energy/n)
    factor = math.sqrt(len(states))
    return torch.cat(aa)/factor, torch.cat(bb)/factor, float(torch.stack(energies).mean())


def damped_direction(A, b, shift):
    """Solve (A.T A + shift I)d=A.T b through its exact dual identity."""
    if shift <= 0: raise ValueError('positive diagonal shift required')
    kernel = A @ A.T
    kernel.diagonal().add_(shift)
    dual = torch.linalg.solve(kernel, b)
    direction = A.T @ dual
    force = A.T @ b
    residual = torch.linalg.vector_norm(A.T @ (A @ direction)+shift*direction-force)
    relative = float(residual / torch.linalg.vector_norm(force).clamp_min(1e-30))
    if not torch.isfinite(direction).all() or relative > 1e-6:
        raise ValueError('nonfinite or inaccurate SR solve')
    return direction, relative


def fit_sr(model, bits, conditions, matrices, n, steps=300, shift=1e-4,
           rate=1., radius=.1, max_trials=6, deadline=None):
    """Fixed outer updates; train-energy-only backtracking, all trials charged.

    Proposed delta is -rate*d with d using HALF the energy gradient. Radius
    bounds the first-order mean state displacement, not exact fidelity loss.
    """
    model.train(); parameters=list(model.parameters()); history=[]
    h_calls=model_calls=jacobian_rows=accepted=0
    for step in range(steps):
        if deadline is not None and time.monotonic()>deadline: raise TimeoutError('SR run cap exceeded')
        states=[]; jacobians=[]
        for condition in conditions:
            psi,J=amplitude_jacobian(model,bits,condition); states.append(psi); jacobians.append(J)
        model_calls += len(conditions); jacobian_rows += len(conditions)*len(bits)
        A,b,loss=tangent_system(states,jacobians,matrices,n); h_calls+=len(conditions)
        direction,residual=damped_direction(A,b,shift)
        velocity=float(torch.linalg.vector_norm(A @ direction))
        eta=min(rate,radius/max(velocity,1e-30))
        slope=float(2*(A.T @ b) @ direction)
        before=torch.nn.utils.parameters_to_vector(parameters).detach().clone()
        chosen=0.; after=loss; trials=0
        for trial in range(max_trials):
            if deadline is not None and time.monotonic()>deadline:
                torch.nn.utils.vector_to_parameters(before,parameters)
                raise TimeoutError('SR run cap exceeded')
            candidate=before-eta*direction
            torch.nn.utils.vector_to_parameters(candidate,parameters)
            with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
                predictions=[model.psi(bits,c) for c in conditions]
                value=float(torch.stack([p@H@p/n for p,H in zip(predictions,matrices)]).mean())
            trials+=1;h_calls+=len(conditions);model_calls+=len(conditions)
            if math.isfinite(value) and value<=loss-1e-4*eta*slope:
                chosen=eta;after=value;accepted+=1;break
            eta*=.5
        if not chosen: torch.nn.utils.vector_to_parameters(before,parameters)
        history.append(dict(step=step+1,energy_before=loss,energy_after=after,accepted_rate=chosen,
                            trials=trials,linear_residual=residual,tangent_velocity=velocity))
    return dict(history=history,outer_steps=steps,accepted_steps=accepted,training_h_actions=h_calls,
                training_model_calls=model_calls,jacobian_rows=jacobian_rows,
                max_solve_dimension=len(bits)*len(conditions),shift=shift)
