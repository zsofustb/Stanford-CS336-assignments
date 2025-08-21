import torch
import torch.nn as nn
from einops import einsum
from typing import Optional
from cs336_basics.Module.Linear import Linear


def Swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: Optional[int],
                device: Optional[torch.device] = None,
                dtype: Optional[torch.dtype] = None,
                _weight1: Optional[torch.Tensor]= None,
                _weight2: Optional[torch.Tensor]= None,
                _weight3: Optional[torch.Tensor]= None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else int(d_model * 8 / 3)

        self.w1 = Linear(self.d_model, self.d_ff, _weight=_weight1)
        self.w2 = Linear(self.d_ff, self.d_model, _weight=_weight2)
        self.w3 = Linear(self.d_model, self.d_ff, _weight=_weight3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(Swish(self.w1(x)) * self.w3(x))
    
            
