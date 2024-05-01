from torch import nn


def create_ranknet_model():
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
        nn.ReLU(),
        nn.Conv2d(N, N * 2, **kwds),  # (7,7)
        nn.BatchNorm2d(N * 2),
        nn.ReLU(),
        nn.Conv2d(N * 2, N * 4, kernel_size=2, stride=1,
                  padding=0, bias=False),  # (6,6)
        nn.BatchNorm2d(N * 4),
        nn.ReLU(),
        nn.Conv2d(N * 4, N * 8, **kwds),  # (3,3)
        nn.BatchNorm2d(N * 8),
        nn.ReLU(),
        nn.Conv2d(N * 8, 1, kernel_size=1, stride=1, padding=0),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten()
    )
