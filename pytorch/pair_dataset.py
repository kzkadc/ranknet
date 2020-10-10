from torchvision.datasets import MNIST
import numpy as np


class MNISTPairDataset(MNIST):
    def __getitem__(self, i: int):
        x1, t1 = super().__getitem__(i)
        r = np.random.randint(len(self))
        x2, t2 = super().__getitem__(r)

        return x1, x2, (1.0 if t1 > t2 else 0.0 if t1 < t2 else 0.5)
