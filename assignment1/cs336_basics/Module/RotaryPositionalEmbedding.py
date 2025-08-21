import torch
import torch.nn as nn
from einops import einsum, rearrange, reduce, repeat

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer('inv_freq', inv_freq)

    def _rotate_half(self, x:torch.Tensor) -> torch.Tensor:
        """
        x -> [x1, x2] -> [-x2, x1]
        """
        x = rearrange(x, '... (d r) -> ... d r', r=2)
        x1, x2 = torch.unbind(x, dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return rearrange(x, '... d r -> ... (d r)')

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(-2)

        if token_positions is None:
            token_positions = torch.arange(seq_len).to(x.device)
            token_positions = token_positions.unsqueeze(0).expand((x.size(0), seq_len))

        theta = einsum(token_positions, self.inv_freq, '... n, d -> ... n d').float()

        cos = theta.cos().repeat_interleave(2, dim=-1)
        sin = theta.sin().repeat_interleave(2, dim=-1)

        x = x * cos + self._rotate_half(x) * sin
        return x

