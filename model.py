# coding: utf-8

import chainer
from chainer import Chain, Variable
import chainer.links as L
import chainer.functions as F

import numpy as np

def compose(x, funcs) -> Variable:
	y = x
	for f in funcs:
		y = f(y)
		
	return y

class RankNet(Chain):
	def __init__(self):
		super().__init__()
		with self.init_scope():
			kwds = {
				"ksize": 4,
				"stride": 2,
				"pad": 1,
				"nobias": True
			}
			N = 32
			self.conv1 = L.Convolution2D(1, N, **kwds)	# -> 14x14
			self.bn1 = L.BatchNormalization(N)
			self.conv2 = L.Convolution2D(N, N*2, **kwds)	# -> 7x7
			self.bn2 = L.BatchNormalization(N*2)
			self.conv3 = L.Convolution2D(N*2, N*4, ksize=2, stride=1, pad=0, nobias=True)	# -> 6x6
			self.bn3 = L.BatchNormalization(N*4)
			self.conv4 = L.Convolution2D(N*4, N*8, **kwds)	# -> 3x3
			self.bn4 = L.BatchNormalization(N*8)
			self.conv5 = L.Convolution2D(N*8, 1, ksize=1, stride=1, pad=0)
			
	def predict(self, x):
		h = compose(x, [
			self.conv1, self.bn1, F.relu,
			self.conv2, self.bn2, F.relu,
			self.conv3, self.bn3, F.relu,
			self.conv4, self.bn4, F.relu,
			self.conv5, 
			lambda v:F.mean(v, axis=(1,2,3))
		])
		return h
		
	def evaluate(self, *in_data):
		(x1, x2), t = in_data
		
		# 0.5 -> -1にして無視する
		t = (t*2).astype(np.int32)
		t[t==1] = -1
		t //= 2
		
		s1 = self.predict(x1)
		s2 = self.predict(x2)
		
		o = s1 - s2
		acc = F.binary_accuracy(o, t)
		chainer.report({"accuracy":acc}, self)
		
		return acc
		
	def __call__(self, pair:tuple, t):
		# t: x1が大きい場合1、x2が大きい場合0
		x1, x2 = pair
		s1 = self.predict(x1)
		s2 = self.predict(x2)
		
		#loss = F.sigmoid_cross_entropy(s1-s2, t)
		p = F.sigmoid(s1-s2)
		loss = F.mean(-t*F.log(p)-(1.0-t)*F.log(1.0-p))
		
		chainer.report({"loss":loss}, self)
		self.evaluate(pair ,t)
		return loss
		
		
