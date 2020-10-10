import torch
from torch import nn
from torch.optim import Optimizer
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ignite.engine.engine import Engine, Events
from ignite.metrics import Average, Accuracy
from torchviz import make_dot

import numpy as np
import matplotlib.pyplot as plt

import pprint
import copy
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Callable, Union

from model import get_ranknet_model
from pair_dataset import MNISTPairDataset

T = torch.Tensor
Batch = Tuple[T, T, T]
Handler = Callable[[Engine], None]
MetricKey = Tuple[str, str]


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="trains a ranking model for mnist")
    parser.add_argument("-b", "--batch_size", type=int,
                        default=64, help="batch size")
    parser.add_argument("-e", "--epoch", type=int, default=10, help="epoch")
    parser.add_argument("-g", type=int, default=-1,
                        help="GPU ID (negative value indicates CPU)")
    parser.add_argument("-d", default="result", help="result directory")

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

    result_path = Path(args.d)
    try:
        result_path.mkdir(parents=True)
    except FileExistsError:
        pass

    def transform(x):
        return np.expand_dims(np.asarray(x, dtype=np.float32), 0) / 255
    train_dataset = MNISTPairDataset(
        root=".", download=True, train=True, transform=transform)
    test_dataset = MNISTPairDataset(
        root=".", download=True, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, args.batch_size)
    test_loader = DataLoader(test_dataset, args.batch_size)

    predictor = get_ranknet_model().to(device)
    opt = torch.optim.Adam(predictor.parameters())

    trainer = get_ranknet_trainer(predictor, opt, device)
    evaluator = get_evaluator(predictor, device)

    log = {}
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              run_evaluator(log, evaluator, train_loader, test_loader))
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              print_log(log, [("train", "loss"), ("test", "loss"),
                                              ("train", "accuracy"), ("test", "accuracy")]))
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              plot_log(log, [("train", "loss"), ("test", "loss")], result_path/"loss.pdf"))
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              plot_log(log, [("train", "accuracy"), ("test", "accuracy")], result_path/"accuracy.pdf"))
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              save_model(predictor, result_path/"models"))
    trainer.add_event_handler(Events.EPOCH_STARTED(once=1),
                              computational_graph(predictor, train_loader,
                                                  result_path/"computational_graph.dot", device))

    trainer.run(train_loader, max_epochs=args.epoch)


def prepare_batch(batch: Batch, device: torch.device) -> Batch:
    x1, x2, t = batch
    return x1.to(device), x2.to(device), t.to(device)


def get_ranknet_trainer(net: nn.Module, opt: Optimizer,
                        device: torch.device = torch.device("cpu")) -> Engine:
    def _update(engine: Engine, batch: Batch) -> T:
        x1, x2, t = prepare_batch(batch, device)
        net.train()
        opt.zero_grad()

        s1, s2 = net(x1), net(x2)
        loss = ranknet_loss(s1, s2, t)
        loss.backward()
        opt.step()

        return loss

    return Engine(_update)


def ranknet_loss(s1: T, s2: T, t: T) -> T:
    o = torch.sigmoid(s1 - s2)
    loss = (-t * o + F.softplus(o)).mean()
    return loss


def get_evaluator(net: nn.Module, device: torch.device) -> Engine:
    def _inference(engine: Engine, batch: Batch) -> Tuple[T, T, T, T]:
        x1, x2, t = prepare_batch(batch, device)
        net.eval()
        s1, s2 = net(x1), net(x2)
        return s1, s2, t, ranknet_loss(s1, s2, t)

    def _acc_output_transform(output: Tuple[T, T, T, T]) -> Tuple[T, T]:
        s1, s2, t, _ = output
        return (s1-s2 > 0).long(), (t > 0.5).long()

    ev = Engine(_inference)
    Average(lambda t: t[3].item()).attach(ev, "loss")
    Accuracy(_acc_output_transform).attach(ev, "accuracy")

    return ev


def run_evaluator(log: dict, evaluator: Engine,
                  train_loader: DataLoader, test_loader: DataLoader) -> Handler:
    def _run(engine: Engine):
        _append("epoch", engine.state.epoch)

        evaluator.run(train_loader)
        _append("train", copy.copy(evaluator.state.metrics))

        evaluator.run(test_loader)
        _append("test", copy.copy(evaluator.state.metrics))

    def _append(key: str, value: Any):
        if key in log:
            log[key].append(value)
        else:
            log[key] = [value]

    return _run


def save_model(net: nn.Module, save_dir: Path) -> Handler:
    try:
        save_dir.mkdir(parents=True)
    except FileExistsError:
        pass

    def _save(engine: Engine):
        p = save_dir/f"model_epoch-{engine.state.epoch:04d}.pt"
        torch.save(net.state_dict(), str(p))

    return _save


def get_metric(log: Dict[str, Any], i: int, key: MetricKey) -> Tuple[str, Any]:
    chunk, name = key
    return log[chunk][i][name]


def print_log(log: Dict[str, Any], keys: Sequence[MetricKey]) -> Handler:
    def _print(engine: Engine):
        s = [f"epoch: {log['epoch'][-1]}"]
        for k in keys:
            val = get_metric(log, -1, k)
            s.append("{}_{}: {}".format(*k, val))

        print(", ".join(s))

    return _print


def plot_log(log: dict, y_keys: Sequence[MetricKey],
             out_path: Path) -> Handler:
    def _plot(engine: Engine):
        plt.figure()
        for k in y_keys:
            y_vals = [get_metric(log, i, k) for i in range(len(log["epoch"]))]
            plt.plot(log["epoch"], y_vals, label="{}_{}".format(*k))
        plt.legend()
        plt.xlabel("epoch")
        plt.savefig(str(out_path))
        plt.close()

    return _plot


def computational_graph(net: nn.Module, loader: DataLoader,
                        out_path: Path, device: torch.device) -> Handler:
    def _make_graph(engine: Engine):
        itr = iter(loader)
        x1, x2, t = prepare_batch(next(itr), device)
        del itr
        net.eval()
        s1, s2 = net(x1), net(x2)
        loss = ranknet_loss(s1, s2, t)
        graph = make_dot(loss, params=dict(net.named_parameters()))
        with out_path.open("w") as f:
            f.write(str(graph))

        print("generated computational graph")

    return _make_graph


if __name__ == "__main__":
    parse_args()
