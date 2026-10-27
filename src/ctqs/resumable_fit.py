"""CPU optimizer sessions with exact per-block closure budgets and safe snapshots.

L-BFGS line-search work is charged even if a block ends during its trial step.
Both parameters and optimizer state then revert to the pre-step accepted state.
Consequently, restarting the SAME block schedule is reproducible; splitting a
budget can change the trajectory compared with one uninterrupted L-BFGS call.
No spectrum, validation selection or test-query logic belongs in this module.
"""
import copy
import random
import time

import numpy as np
import torch


class _BlockBudgetReached(Exception):
    pass


def _cpu_copy(value):
    """Detach checkpoint storage; keep only weights_only-loadable containers."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'unsupported checkpoint value: {type(value).__name__}')


def _rng_state():
    kind, keys, pos, has_gauss, cached = np.random.get_state()
    return dict(torch_cpu=torch.get_rng_state().clone(), python=random.getstate(),
                numpy=dict(kind=kind, keys=keys.tolist(), position=int(pos),
                           has_gauss=int(has_gauss), cached_gaussian=float(cached)))


def _restore_rng(state):
    torch.set_rng_state(state['torch_cpu'].cpu())
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n['kind'], np.asarray(n['keys'], dtype=np.uint32),
                         n['position'], n['has_gauss'], n['cached_gaussian']))


class ResumableFit:
    """Persistent Adam or L-BFGS, with complete accepted-state CPU snapshots.

    Counters count completed objective-and-gradient evaluations and enclosing
    optimizer calls, not necessarily distinct physical parameter updates.
    attempted_closures additionally records an attempt that fails numerically.
    """
    def __init__(self, model, optimizer_name='lbfgs', lr=1., history_size=20):
        if any(p.device.type != 'cpu' for p in model.parameters()):
            raise ValueError('ResumableFit currently supports CPU RNG/device only')
        if optimizer_name not in ('adam', 'lbfgs'):
            raise ValueError('optimizer_name must be adam or lbfgs')
        self.model = model
        self.optimizer_name = optimizer_name
        self.settings = dict(lr=float(lr), history_size=int(history_size))
        if optimizer_name == 'adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, foreach=False)
        else:
            self.optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=1,
                max_eval=25, history_size=history_size, tolerance_grad=0.,
                tolerance_change=0., line_search_fn='strong_wolfe')
        self.counters = dict(closure_evaluations=0, optimizer_steps_completed=0,
                             discarded_final_step_evaluations=0,
                             attempted_closures=0, completed_blocks=0)

    def state_dict(self):
        """Deep snapshot, including optimizer memory and global CPU RNG state."""
        return dict(schema_version=1, optimizer_name=self.optimizer_name,
                    settings=copy.deepcopy(self.settings),
                    model_state=_cpu_copy(self.model.state_dict()),
                    optimizer_state=_cpu_copy(self.optimizer.state_dict()),
                    counters=copy.deepcopy(self.counters), training=self.model.training,
                    rng_state=_cpu_copy(_rng_state()))

    def load_state_dict(self, state):
        if state['schema_version'] != 1 or state['optimizer_name'] != self.optimizer_name:
            raise ValueError('incompatible checkpoint schema or optimizer')
        if state['settings'] != self.settings:
            raise ValueError('constructor settings differ from saved session')
        counters = state['counters']
        if set(counters) != set(self.counters) or any(
                not isinstance(v, int) or isinstance(v, bool) or v < 0
                for v in counters.values()):
            raise ValueError('invalid checkpoint counters')
        self.model.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(copy.deepcopy(state['optimizer_state']))
        self.counters = copy.deepcopy(counters)
        self.model.train(state['training'])
        _restore_rng(state['rng_state'])

    def advance(self, objective, evaluations, deadline=None):
        """Consume exactly evaluations finite closures, unless an error raises.

        deadline is an absolute time.monotonic() deadline. All rejected trials
        consume budget. History accepted_step_completed only describes whether
        the enclosing optimizer call completed; it is not a trial acceptance flag.
        """
        if not isinstance(evaluations, int) or isinstance(evaluations, bool) or evaluations <= 0:
            raise ValueError('positive integer closure budget required')
        calls = steps = discarded = attempts = 0
        history = []

        def closure():
            nonlocal calls, attempts
            if calls >= evaluations:
                raise _BlockBudgetReached()
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError('training deadline exceeded')
            attempts += 1
            self.counters['attempted_closures'] += 1
            self.optimizer.zero_grad()
            loss = objective(self.model)
            if not torch.is_tensor(loss) or loss.ndim != 0 or not torch.isfinite(loss):
                raise ValueError('objective must return a finite scalar tensor')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all()
                   for p in self.model.parameters()):
                raise ValueError('nonfinite gradient')
            calls += 1
            self.counters['closure_evaluations'] += 1
            history.append(dict(evaluation=calls,
                cumulative_evaluation=self.counters['closure_evaluations'],
                optimizer_call=self.counters['optimizer_steps_completed'] + 1,
                loss=float(loss.detach()), accepted_step_completed=None))
            return loss

        self.model.train()
        while calls < evaluations:
            before = calls
            if self.optimizer_name == 'adam':
                closure()
                self.optimizer.step()
            else:
                accepted_model = _cpu_copy(self.model.state_dict())
                accepted_optimizer = _cpu_copy(self.optimizer.state_dict())
                accepted_rng = _rng_state()
                try:
                    self.optimizer.step(closure)
                except _BlockBudgetReached:
                    self.model.load_state_dict(accepted_model)
                    self.optimizer.load_state_dict(accepted_optimizer)
                    self.optimizer.zero_grad(set_to_none=True)
                    _restore_rng(accepted_rng)
                    discarded = calls - before
                    self.counters['discarded_final_step_evaluations'] += discarded
                    for row in history[before:]:
                        row['accepted_step_completed'] = False
                    break
                except Exception:
                    # A failed numerical run must not leave speculative params
                    # or curvature memory masquerading as an accepted state.
                    self.model.load_state_dict(accepted_model)
                    self.optimizer.load_state_dict(accepted_optimizer)
                    self.optimizer.zero_grad(set_to_none=True)
                    _restore_rng(accepted_rng)
                    raise
            steps += 1
            self.counters['optimizer_steps_completed'] += 1
            for row in history[before:]:
                row['accepted_step_completed'] = True
        self.counters['completed_blocks'] += 1
        return dict(closure_evaluations=calls, optimizer_steps_completed=steps,
                    discarded_final_step_evaluations=discarded,
                    attempted_closures=attempts, history=history,
                    cumulative_counters=copy.deepcopy(self.counters))
