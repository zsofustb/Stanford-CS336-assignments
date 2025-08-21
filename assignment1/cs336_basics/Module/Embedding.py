import torch
import torch.nn as nn
from typing import Optional


class Embedding(nn.Module):
    def __init__(self, num_embedding: int, embedding_dim: int, 
                device: Optional[torch.device] = None,
                dtype: Optional[torch.dtype] = None,
                _weight: Optional[torch.Tensor] = None):
        super().__init__()

        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim

        if _weight is not None:
            self.weight = nn.Parameter(_weight)
        else:
            self.weight = nn.Parameter(
                torch.empty((self.num_embedding, self.embedding_dim), device=device, dtype=dtype)
            )
            self._init_weight()

    def _init_weight(self):
        variance = 2.0 / (self.num_embedding + self.embedding_dim)
        std = variance ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3, b=3)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.weight[indices]


