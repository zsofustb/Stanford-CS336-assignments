import torch

def log_sum_exp(x: torch.Tensor) -> torch.Tensor:
    """
    log(sum(e ^ x))
    """
    # Max防止上溢
    Max = torch.max(x, dim=-1, keepdim=True).values
    return Max + torch.log(torch.exp(x - Max).sum(dim=-1, keepdim=True))

def CrossEntropyLoss(inputs: totch.Tensor, targets: torch.Tensor) -> torch.Tensor():
    """
    size:
    inputs: (bsz, num_classes)
    targets: (bsz, )
    """