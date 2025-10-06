import torch
from torch import Tensor


def softmax(x: Tensor, dim: int) -> Tensor:
    assert len(x.shape) > dim, (
        f"dimension {dim} out of range of input dimension {x.shape}"
    )
    normalized_x = x - torch.max(x, dim=dim, keepdim=True)[0]
    return torch.exp(normalized_x) / torch.sum(torch.exp(normalized_x), dim=dim, keepdim=True)

def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)
