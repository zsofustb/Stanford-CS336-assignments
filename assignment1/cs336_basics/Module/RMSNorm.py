import torch
import torch.nn as nn
from typing import Optional

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1E-5, 
                device: Optional[torch.device] = None,
                dtype: Optional[torch.dtype] = None,
                _weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        if _weight is not None:
            self.weight = nn.Parameter(_weight)
        else:
            self.weight = nn.Parameter(torch.ones((d_model), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return x * self.weight
