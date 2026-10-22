"""Count every gradient closure, including rejected L-BFGS trials."""
import copy
import time
import torch


class ClosureBudgetReached(Exception):
    pass


def fit(model, objective, optimizer_name, evaluations, adam_lr=.005,
        lbfgs_lr=1., history_size=20, deadline=None):
    """Objective returns a scalar; no ED, selection or unseen-point query here.

    The budget is in complete objective-and-gradient evaluations. L-BFGS may
    exhaust it inside a speculative line search; restore the pre-step accepted
    parameters. Optimizer state is discarded at the end, so need not be restored.
    History losses are closure evaluations, NOT all accepted iterates.
    """
    if evaluations <= 0: raise ValueError('positive closure budget required')
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr, foreach=False)
    elif optimizer_name == 'lbfgs':
        optimizer = torch.optim.LBFGS(model.parameters(), lr=lbfgs_lr, max_iter=1,
            max_eval=25, history_size=history_size, tolerance_grad=0.,
            tolerance_change=0., line_search_fn='strong_wolfe')
    else: raise ValueError('unknown optimizer')
    calls = steps = discarded = 0; history = []
    def closure():
        nonlocal calls
        if calls >= evaluations: raise ClosureBudgetReached()
        if deadline is not None and time.monotonic() > deadline: raise TimeoutError('run cap exceeded')
        optimizer.zero_grad(); loss = objective(model)
        if not torch.isfinite(loss): raise ValueError('nonfinite loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError('nonfinite gradient')
        calls += 1
        history.append(dict(evaluation=calls, optimizer_call=steps+1, loss=float(loss.detach()),
                            accepted_step_completed=None))
        return loss
    model.train()
    while calls < evaluations:
        before = calls
        if optimizer_name == 'adam':
            closure(); optimizer.step(); steps += 1
        else:
            accepted = copy.deepcopy(model.state_dict())
            try:
                optimizer.step(closure); steps += 1
            except ClosureBudgetReached:
                model.load_state_dict(accepted); discarded = calls-before
                for row in history[before:]: row['accepted_step_completed'] = False
                break
        for row in history[before:]: row['accepted_step_completed'] = True
    return dict(closure_evaluations=calls, optimizer_steps_completed=steps,
                discarded_final_step_evaluations=discarded, history=history)
