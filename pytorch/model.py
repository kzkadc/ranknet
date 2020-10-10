from typing import Callable
from torch import nn
import torch

T = torch.Tensor


def relu():
    return nn.ReLU(inplace=True)


def get_ranknet_model():
    kwds = {
        "kernel_size": 4,
        "stride": 2,
        "padding": 1,
        "bias": False
    }
    N = 32
    return nn.Sequential(
        nn.Conv2d(1, N, **kwds),    # (14,14)
        nn.BatchNorm2d(N),
        relu(),
        nn.Conv2d(N, N * 2, **kwds),  # (7,7)
        nn.BatchNorm2d(N * 2),
        relu(),
        nn.Conv2d(N * 2, N * 4, kernel_size=2, stride=1,
                  padding=0, bias=False),  # (6,6)
        nn.BatchNorm2d(N * 4),
        relu(),
        nn.Conv2d(N * 4, N * 8, **kwds),  # (3,3)
        nn.BatchNorm2d(N * 8),
        relu(),
        nn.Conv2d(N * 8, 1, kernel_size=1, stride=1, padding=0),
        Lambda(lambda x: x.mean(dim=(1, 2, 3)))
    )


class Lambda(nn.Module):
    def __init__(self, func: Callable[[T], T]):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
