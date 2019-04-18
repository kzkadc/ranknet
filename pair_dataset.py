# coding: utf-8

import chainer
from chainer import datasets, dataset

import numpy as np

class MNISTPairDataset(dataset.DatasetMixin):
	def __init__(self, train=False):
		self.mnist = datasets.get_mnist(ndim=3)[0 if train else 1]
		
	def __len__(self):
		return len(self.mnist)
		
	def get_example(self, i):
		x1, t1 = self.mnist[i]
		r = np.random.randint(len(self.mnist))
		x2, t2 = self.mnist[r]
		
		if t1 > t2:
			t = 1.
		elif t1 < t2:
			t = 0.
		else:
			t = 0.5
		
		return (x1, x2), t
		
