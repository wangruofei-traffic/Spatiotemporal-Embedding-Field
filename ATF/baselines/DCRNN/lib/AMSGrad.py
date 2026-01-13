import math
import torch
from torch.optim import Optimizer

class AMSGrad(Optimizer):
    """Implements AMSGrad algorithm (Adam variant with max of past squared gradients)."""

    def __init__(self, params, lr=0.01, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AMSGrad, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AMSGrad does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['v'] = torch.zeros_like(p.data)
                    # Maximum of v
                    state['vhat'] = torch.zeros_like(p.data)

                m, v, vhat = state['m'], state['v'], state['vhat']
                beta1, beta2 = group['betas']
                eps = group['eps']
                lr = group['lr']

                state['step'] += 1
                step = state['step']

                # Update biased first moment estimate
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second raw moment estimate
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # Compute max of past squared gradients
                torch.maximum(vhat, v, out=vhat)
                # Bias-corrected learning rate
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                # Parameter update
                p.data.addcdiv_(m, vhat.sqrt().add(eps), value=-step_size)

        return loss
