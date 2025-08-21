import torch
import torch.nn as nn
import math
from einops import einsum
from typing import Optional

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, 
                device: Optional[torch.device] = None,
                dtype: Optional[torch.dtype] = None,
                _weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        if _weight is not None:
            self.weight = nn.Parameter(_weight)
        else:
            self.weight = torch.nn.Parameter(
                torch.empty((out_features, in_features), device=device, dtype=dtype)
            )
            self._init_weight()
        # torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def _init_weight(self):
        variance = 2.0 / (self.in_features + self.out_features)
        std = variance ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3.0, b=3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, '... din, dout din -> ... dout')



