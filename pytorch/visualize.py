# coding: utf-8

import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import pprint
import tqdm
import matplotlib.pyplot as plt
import numpy as np


from model import get_ranknet_model


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="visualizes scores for test dataset")
    parser.add_argument("-m", required=True,
                        help="model file generated from train.py")
    parser.add_argument("-b", type=int, default=100, help="batch size")
    parser.add_argument("-o", default="out.pdf", help="output file")
    parser.add_argument("-t", default=None, help="title of the figure")
    parser.add_argument("-g", type=int, default=-1, help="GPU id")
    args = parser.parse_args()
    pprint.pprint(vars(args))
    main(args)


def main(args):
    if args.g >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.g:d}")
        print(f"GPU mode: {args.g:d}")
    else:
        device = torch.device("cpu")
        print("CPU mode")

    predictor = get_ranknet_model().to(device)
    predictor.load_state_dict(torch.load(args.m))
    predictor.eval()

    test_dataset = MNIST(root=".", download=True, train=False,
                         transform=lambda x: np.expand_dims(np.asarray(x, dtype=np.float32), 0)/255)
    test_loader = DataLoader(test_dataset, batch_size=args.b, shuffle=False)

    score_table = [[] for _ in range(10)]
    for batch in tqdm.tqdm(test_loader):
        x, t = batch
        t = t.numpy()
        x.to(device)
        with torch.no_grad():
            score = predictor(x).cpu().numpy()
        for s, l in zip(score, t):
            score_table[l].append(s)

    for i, scores in enumerate(score_table):
        plt.hist(scores, bins=50, alpha=0.4,
                 histtype="stepfilled", label=str(i))

    plt.legend(loc="upper right")
    plt.xlabel("score")
    plt.ylabel("frequency")
    if args.t is not None:
        plt.title(args.t)
    plt.savefig(args.o)


if __name__ == "__main__":
    parse_args()
