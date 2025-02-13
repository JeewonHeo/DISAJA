import torch
from torch import Tensor
import torch.nn.functional as F


def object_wise_cross_entropy(
        input: Tensor,
        target: Tensor,
        mask: Tensor,
        length: Tensor,
        class_last: bool = True,
        reduction: str = 'mean',
        weight: Tensor = None,
) -> Tensor:
    """Object-wise cross-entropy loss
    Args:
        input: (B, L, C)
        target:
        mask:
        length:
        class_last:
        reduction:
    Returns:
        output:
    """
    if class_last:
        input = input.permute(0, 2, 1)
    loss = F.cross_entropy(input, target, reduction='none', weight=weight)
    loss.masked_fill_(mask, 0)

    length = length.to(input.dtype)
    loss = loss.sum(dim=1) / length

    if reduction == 'mean':
        loss = loss.mean()
    return loss
