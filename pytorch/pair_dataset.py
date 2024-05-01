from torchvision.datasets import MNIST
import numpy as np


class MNISTPairDataset(MNIST):
    def __getitem__(self, i: int):
        x1, t1 = super().__getitem__(i)
        r = np.random.randint(len(self))
        x2, t2 = super().__getitem__(r)

        if t1 > t2:
            label = 1.0
        elif t1 < t2:
            label = 0.0
        else:
            label = 0.5

        return x1, x2, label
