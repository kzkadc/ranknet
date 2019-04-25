# Chainer implementation of RankNet

## Requirementsc
chainer, matplotlib, numpy, tqdm

```bash
$ pip install chainer matplotlib numpy tqdm
```

## Usage
1. Train a ranking model

```bash
$ python train.py
```

`-h` option shows help.

```bash
$ python train.py -h

```

2. Visualize scores for test data

```bash
$ python visualize.py -m model_file -o output_file
```

`-h` option shows help.

```bash
$ python visualize.py -h

```
