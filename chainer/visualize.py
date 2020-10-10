# coding: utf-8

import pprint
import tqdm
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import chainer
from chainer import dataset, datasets, serializers, iterators

from model import RankNet

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="visualizes scores for test dataset")
    parser.add_argument("-m", required=True, help="model file generated from train.py")
    parser.add_argument("-b", type=int, default=100, help="batch size")
    parser.add_argument("-o", default="out.pdf", help="output file")
    parser.add_argument("-t", default=None, help="title of the figure")
    args = parser.parse_args()
    pprint.pprint(vars(args))
    main(args)
    
def main(args):
    chainer.config.train = False

    predictor = RankNet()
    serializers.load_npz(args.m, predictor)
    
    test_dataset = datasets.get_mnist(ndim=3)[1]
    test_iter = iterators.SerialIterator(test_dataset, batch_size=args.b, repeat=False, shuffle=False)
    
    score_table = [[] for _ in range(10)]
    for batch in tqdm.tqdm(test_iter):
        x, t = dataset.concat_examples(batch)
        score = predictor.predict(x).array
        for s,l in zip(score, t):
            score_table[l].append(s)
            
    for i,scores in enumerate(score_table):
        plt.hist(scores, bins=50, alpha=0.4, histtype="stepfilled", label=i)
        
    plt.legend(loc="upper right")
    plt.xlabel("score")
    plt.ylabel("frequency")
    plt.ylim(0,100)
    plt.xlim(-25,25)
    if args.t is not None:
        plt.title(args.t)
    plt.savefig(args.o)

if __name__ == "__main__":
    parse_args()
