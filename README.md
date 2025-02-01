# PyTorch implementation of RankNet (unofficial)

Burges, Christopher, et al. "Learning to rank using gradient descent." Proceedings of the 22nd International Conference on Machine learning (ICML-05). 2005.

## Requirements
pytorch, pytorch-ignite, torchviz, numpy tqdm matplotlib

pytorch: see [the official document](https://pytorch.org/get-started/locally/).

```bash
$ pip install pytorch-ignite torchviz numpy tqdm matplotlib
```

## Usage
1. Train a ranking model

```bash
$ python train.py
```

`-h` option shows help.

```bash
$ python train.py -h
usage: train.py [-h] [-b BATCH_SIZE] [-e EPOCH] [-g G] [-d D] [--compile_model]

trains a ranking model for mnist

options:
  -h, --help            show this help message and exit
  -b, --batch_size BATCH_SIZE
                        batch size
  -e, --epoch EPOCH     epoch
  -g G                  GPU ID (negative value indicates CPU)
  -d D                  result directory
  --compile_model       enable torch.compile
```

2. Visualize scores for test data

```bash
$ python visualize.py -m model_file -o output_file
```

`-h` option shows help.

```bash
$ python visualize.py -h
usage: visualize.py [-h] -m M [-b B] [-o O] [-t T]

visualizes scores for test dataset

optional arguments:
  -h, --help  show this help message and exit
  -m M        model file generated from train.py
  -b B        batch size
  -o O        output file
  -t T        title of the figure
```
