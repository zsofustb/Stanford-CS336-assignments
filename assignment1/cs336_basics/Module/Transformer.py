import torch
import torch.nn as nn
import math
from einops import einsum, rearrange, repeat, reduce
from typing import Optional
from cs336_basics.Module.MultiHeadAttention import MultiHeadAttention
from cs336_basics.Module.SwiGLU import SwiGLU
from cs336_basics.Module.Linear import Linear
from cs336_basics.Module.RMSNorm import RMSNorm
from cs336_basics.Module.Embedding import Embedding


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        use_rope: bool = False,
        max_seq_len: int = 512,
        theta: float = 10000.0
    ):
        super().__init__()
        
        self.MHA = MultiHeadAttention(d_model, num_heads, use_rope=use_rope, max_seq_len=max_seq_len, theta=theta)
        self.FFN = SwiGLU(d_model, d_ff=d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.MHA(self.norm1(x))
        x = x + self.FFN(self.norm2(x))

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float
    ):
        super().__init__()
        
        self.embedding = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    use_rope=True,
                    max_seq_len=context_length,
                    theta=rope_theta
                ) for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.output_layer = Linear(d_model, vocab_size)


    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.output_layer(x)

        return x