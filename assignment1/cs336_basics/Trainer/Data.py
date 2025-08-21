import torch
import numpy as np
import numpy.typing as npt
import os


def get_batch(
        dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given a dataset (a 1D numpy array of integers) and a desired batch size and
    context length, sample language modeling input sequences and their corresponding
    labels from the dataset.

    Args:
        dataset (np.array): 1D numpy array of integer token IDs in the dataset.
        batch_size (int): Desired batch size to sample.
        context_length (int): Desired context length of each sampled example.
        device (str): PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.


    举个极简例子
        假设 dataset = [10, 11, 12, 13, 14, 15]，batch_size = 2，context_length = 3。
        函数可能随机抽到两条样本：
        样本	inputs	        targets
        1	[10, 11, 12]	[11, 12, 13]
        2	[12, 13, 14]	[13, 14, 15]
    """
    data_len = len(dataset)
    inputs = torch.empty((batch_size, context_length), dtype=torch.long, device=device)
    targets = torch.empty((batch_size, context_length), dtype=torch.long, device=device)

    for i in range(batch_size):
        start_idx = np.random.randint(0, data_len - context_length)
        inputs[i] = torch.from_numpy(dataset[start_idx: start_idx + context_length])
        targets[i] = torch.from_numpy(dataset[start_idx + 1: start_idx + context_length + 1])

    return inputs, targets




