import torch
import torch.nn as nn
from einops import einsum
from typing import Optional, BinaryIO, IO
import math
import os


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    沿着dim维进行softmax
    """
    mx = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - mx)
    sum_exp_x = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / sum_exp_x

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Float[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA

    size of args:
    Q: (batch_size, ..., seq_len, d_k)
    K: (batch_size, ..., seq_len, d_k)
    V: (batch_size, ..., seq_len, d_v)
    mask: (seq_len, seq_len)
    """
    d_k = Q.size(-1)
    attn_scores = einsum(Q, K, '... i d_k, ... j d_k -> ... i j') / math.sqrt(d_k)

    if mask is not None:
        # mask == 0 返回一个bool类型张量
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

    weights = softmax(attn_scores, dim=-1)
    output = weights @ V
    return output
    
def clip_gradient(parameters, max_l2_norm) -> None:
    """
    防止梯度过大，对梯度进行裁剪
    """
    grads = [p.grad for p in parameters if p.grad is not None]

    # norm为梯度的二范数
    norm = 0
    for grad in grads:
        norm += (grad ** 2).sum()
    norm = torch.sqrt(norm)

    clip_coef = min(1, max_l2_norm / (norm + 1E-6))
    # norm > max_l2_norm的话, 说明梯度过大, clip_coef < 1, grad按照同比例缩小
    if clip_coef < 1:
        for grad in grads:
            grad *= clip_coef

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes]
) -> None:
    """
    Given a model, optimizer, and an iteration number, serialize them to disk.

    Args:
        model (torch.nn.Module): Serialize the state of this model.
        optimizer (torch.optim.Optimizer): Serialize the state of this optimizer.
        iteration (int): Serialize this value, which represents the number of training iterations
            we've completed.
        out (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialize the model, optimizer, and iteration to.
    """
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Given a serialized checkpoint (path or file-like object), restore the
    serialized state to the given model and optimizer.
    Return the number of iterations that we previously serialized in
    the checkpoint.

    Args:
        src (str | os.PathLike | BinaryIO | IO[bytes]): Path or file-like object to serialized checkpoint.
        model (torch.nn.Module): Restore the state of this model.
        optimizer (torch.optim.Optimizer): Restore the state of this optimizer.
    Returns:
        int: the previously-serialized number of iterations.
    """
    all = torch.load(src)
    model.load_state_dict(all['model_state_dict'])
    optimizer.load_state_dict(all['optimizer_state_dict'])
    iteration = all['iteration']

    return iteration
    