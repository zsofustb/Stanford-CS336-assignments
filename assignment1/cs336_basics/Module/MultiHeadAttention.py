import torch
import torch.nn as nn
import math
from einops import einsum, rearrange
from typing import Optional
from cs336_basics.Module.Linear import Linear
from cs336_basics.Module.utils import scaled_dot_product_attention
from cs336_basics.Module.RotaryPositionalEmbedding import RotaryPositionalEmbedding


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool = False,
        max_seq_len: Optional[int] = None,
        theta: Optional[float] = None,
        token_positions: Optional[torch.Tensor] = None,
        q_weight: Optional[torch.Tensor]= None,
        k_weight: Optional[torch.Tensor]= None,
        v_weight: Optional[torch.Tensor]= None,
        o_weight: Optional[torch.Tensor]= None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'

        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        self.theta = theta
        self.token_positions = token_positions
        
        self.q_proj = Linear(d_model, d_model, _weight=q_weight)
        self.k_proj = Linear(d_model, d_model, _weight=k_weight)
        self.v_proj = Linear(d_model, d_model, _weight=v_weight)
        self.out_proj = Linear(d_model, d_model, _weight=o_weight)

    def _causal_mask(self, seq_len: int) -> torch.Tensor:
        mask = torch.tril(torch.ones((seq_len, seq_len))).bool()
        mask = rearrange(mask, '... -> 1 1 ...')
        return mask
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_model = x.size()
        
        Q = rearrange(
            self.q_proj(x),
            '... seq_len (num_head head_dim) -> ... num_head seq_len head_dim',
            num_head=self.num_heads
        )
        K = rearrange(
            self.k_proj(x),
            '... seq_len (num_head head_dim) -> ... num_head seq_len head_dim',
            num_head=self.num_heads
        )
        V = rearrange(
            self.v_proj(x),
            '... seq_len (num_head head_dim) -> ... num_head seq_len head_dim',
            num_head=self.num_heads
        )

        if self.use_rope:
            RoPE = RotaryPositionalEmbedding(self.theta, self.head_dim, self.max_seq_len)
            Q = RoPE(Q, self.token_positions)
            K = RoPE(K, self.token_positions)

        mask = self._causal_mask(seq_len)

        attn_output = scaled_dot_product_attention(Q, K, V, mask)
        attn_output = rearrange(attn_output, '... num_heads seq_len head_dim -> ... seq_len (num_heads head_dim)')
        output = self.out_proj(attn_output)
        return output
