import torch

def log_sum_exp(x: torch.Tensor) -> torch.Tensor:
    """
    log(sum(e ^ x))
    """
    # Max防止上溢
    Max = torch.max(x, dim=-1, keepdim=True).values
    return Max + torch.log(torch.exp(x - Max).sum(dim=-1, keepdim=True))

def CrossEntropyLoss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor():
    """
    size:
    inputs: (bsz, num_classes)
    targets: (bsz, )
    """
    
    """ torch.gather()
    >>> t = torch.tensor([[1,2],[3,4]])
    >>> torch.gather(t, 1, torch.tensor([[0,0],[1,0]]))
    tensor([[ 1,  1],
            [ 4,  3]])
    """
    # target_logits size: (bsz, )
    target_logits = torch.gather(inputs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    # loss size: (bsz,)
    loss = -target_logits + log_sum_exp(inputs)
    return loss.mean()