# coding: utf-8

import pprint

import model
from pair_dataset import MNISTPairDataset

import chainer
from chainer import iterators, optimizers, training
from chainer.training import extensions

import numpy as np

def parse_args():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("-b", "--batch_size", type=int, default=64)
	parser.add_argument("-e", "--epoch", type=int, default=10)
	
	args = parser.parse_args()
	pprint.pprint(vars(args))
	main(args)
	
	
def main(args):
	train_dataset = MNISTPairDataset()
	test_dataset = MNISTPairDataset(train=False)
	
	train_iter = iterators.SerialIterator(train_dataset, args.batch_size, repeat=True, shuffle=True)
	test_iter = iterators.SerialIterator(test_dataset, args.batch_size, repeat=False, shuffle=False)
	
	predictor = model.RankNet()
	
	opt = optimizers.Adam()
	opt.setup(predictor)
	
	updater = training.updaters.StandardUpdater(train_iter, opt, 
		loss_func=predictor, converter=converter)
	trainer = training.Trainer(updater, (args.epoch, "epoch"))
	
	evaluator = extensions.Evaluator(test_iter, predictor, converter=converter)
	trainer.extend(evaluator)
	trainer.extend(extensions.LogReport())
	trainer.extend(extensions.ProgressBar())
	trainer.extend(extensions.PrintReport(["main/loss", "validation/main/loss", "main/accuracy", "validation/main/accuracy"]))
	trainer.extend(extensions.PlotReport(["main/loss", "validation/main/loss"], file_name="loss_plot.png"))
	trainer.extend(extensions.PlotReport(["main/accuracy", "validation/main/accuracy"], file_name="loss_plot.png"))
	
	trainer.run()
	
def converter(batch, device=None):
	x1_array, x2_array, t_array = [], [], []
	for b in batch:
		(x1, x2), t = b
		x1_array.append(x1)
		x2_array.append(x2)
		t_array.append(t)
		
	x1_array = np.stack(x1_array)
	x2_array = np.stack(x2_array)
	t_array = np.array(t_array)
	
	return (x1_array, x2_array), t_array
	
	
if __name__ == "__main__":
	parse_args()
	
	
