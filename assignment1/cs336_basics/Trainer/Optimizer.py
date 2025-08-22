import torch
import math
from typing import Optional
from collections.abc import Callable, Iterable


def learning_rate_cosine_schedule(it, alpha_max, alpha_min, T_w, T_c):
    if it < T_w:
        return it / T_w * alpha_max
    elif T_w <= it < T_c:
        return alpha_min + 0.5 * (1 + math.cos(math.pi * (it - T_w) / (T_c - T_w))) * (alpha_max - alpha_min)
    else:
        return alpha_min


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1E-3,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.01,
        eps: float = 1E-8
    ):
        if lr < 0:
            raise ValueError(f'Invalid lr: {lr}')
        if not (0 <= betas[0] < 1) or not (0 <= betas[1] < 1):
            raise ValueError(f'Invalid betas: {betas}')
        if weight_decay < 0:
            raise ValueError(f'Invalid weight_decay: {weight_decay}') 
        if eps <= 0:
            raise ValueError(f'Invalid eps: {eps}')
        # 父类规定变量名为defaults
        defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
            'eps': eps
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get('step', 1)
                grad = p.grad.data

                m = state.get('m', torch.zeros_like(p.data))
                v = state.get('v', torch.zeros_like(p.data))
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * torch.square(grad)

                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data -= lr_t * m / (torch.sqrt(v) + eps) + lr * weight_decay * p.data

                state['m'] = m
                state['v'] = v
                state['step'] = t + 1

        return loss
                
                
            





        
            