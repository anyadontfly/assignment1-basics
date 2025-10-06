import math

import torch
import torch.optim as optim
from torch import Tensor

from typing import Tuple


def cross_entropy_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    assert len(inputs.shape) == 2, f"Expected inputs have shape (batch_size, vocab_size), but got {inputs.shape}"
    assert len(targets.shape) == 1 and targets.shape[0] == inputs.shape[0], (
        f"Expected targets to have shape (batch_size,), matching inputs' batch size {inputs.shape[0]}, but got {targets.shape}"
    )

    inputs_max = torch.max(inputs, dim=-1, keepdim=True).values
    log_sum_exp = inputs_max + torch.log(torch.sum(torch.exp(inputs - inputs_max), dim=-1, keepdim=True))
    target_inputs = inputs[torch.arange(inputs.size(0)), targets]
    return (log_sum_exp - target_inputs).mean()

def learning_rate_schedule(t, lr_max, lr_min, t_w, t_c) -> float:
    if t < t_w:
        return t / t_w * lr_max
    elif t >= t_w and t <= t_c:
        return lr_min + 0.5 * (1.0 + math.cos((t - t_w) / (t_c - t_w) * math.pi)) * (lr_max - lr_min)
    else:
        return lr_min

def gradient_clipping(params, max_norm: float, eps: float=1e-6):
    grads = [param.grad for param in params if param.grad is not None]

    total_norm = torch.norm(torch.stack([grad.detach() for grad in grads]), 2)

    if total_norm < max_norm:
        return
    
    for grad in grads:
        grad.mul_(max_norm / (total_norm + eps))


class AdamW(optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float=1e-3,
        weight_decay: float=1e-2,
        betas: Tuple[float, float]=(0.9, 0.999),
        eps: float = 1e-8,
    ):
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            for param in group["params"]:
                if param.requires_grad:
                    self.state[param]["m"] = torch.zeros_like(param)
                    self.state[param]["v"] = torch.zeros_like(param)
        

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad.data
                t = self.state[param].get("t", 0) + 1
                m = self.state[param]["m"]
                v = self.state[param]["v"]

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                lr_t = lr * (math.sqrt(1 - beta2 ** t)) / (1- beta1 ** t)

                param.data = param.data - lr_t * torch.div(m, torch.sqrt(v) + eps)
                param.data = param.data - lr * weight_decay * param.data

                self.state[param]["t"] = t
        return loss
