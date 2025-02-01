import pprint
import copy
from pathlib import Path
from dataclasses import dataclass, InitVar

import torch
from torch import nn, Tensor
from torch.optim import Optimizer
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from ignite.metrics import Average, Accuracy
from torchviz import make_dot

import numpy as np
import matplotlib.pyplot as plt

from model import create_ranknet_model
from pair_dataset import MNISTPairDataset


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
    parser.add_argument("--compile_model", action="store_true",
                        help="enable torch.compile")

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
    result_path.mkdir(parents=True, exist_ok=True)

    transform = lambda x: np.expand_dims(
        np.asarray(x, dtype=np.float32), 0) / 255
    train_dataset = MNISTPairDataset(
        root=".", download=True, train=True, transform=transform)
    test_dataset = MNISTPairDataset(
        root=".", download=True, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, args.batch_size)
    test_loader = DataLoader(test_dataset, args.batch_size)

    predictor = create_ranknet_model().to(device)
    opt = torch.optim.Adam(predictor.parameters())

    trainer = RankNetTrainer(predictor, opt, device, args.compile_model)
    evaluator = RankNetEvaluator(predictor, device, args.compile_model)

    log = {}
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              run_evaluator(log, evaluator, train_loader, test_loader))
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              print_log(log, [("train", "loss"), ("test", "loss"),
                                              ("train", "accuracy"), ("test", "accuracy")]))
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              plot_log(log, [("train", "loss"), ("test", "loss")], result_path / "loss.pdf"))
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              plot_log(log, [("train", "accuracy"), ("test", "accuracy")], result_path / "accuracy.pdf"))
    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              save_model(predictor, result_path / "models"))
    trainer.add_event_handler(Events.EPOCH_STARTED(once=1),
                              computational_graph(predictor, train_loader,
                                                  result_path / "computational_graph.dot", device))

    trainer.run(train_loader, max_epochs=args.epoch)


def prepare_batch(batch: tuple[Tensor, Tensor, Tensor], device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    x1, x2, t = batch
    return x1.to(device), x2.to(device), t.to(device)


@dataclass
class RankNetTrainer(Engine):
    net: InitVar[nn.Module]
    opt: Optimizer
    device: torch.device = torch.device("cpu")
    compile_model: InitVar[bool] = False

    def __post_init__(self, net: nn.Module, compile_model: bool):
        super().__init__(self.update)

        self._net = net
        if compile_model:
            try:
                self._net = torch.compile(net)
            except RuntimeError as e:
                print(f"torch.compile failed: {e}")

    def update(self, engine: Engine, batch: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        x1, x2, t = prepare_batch(batch, self.device)
        self._net.train()
        self.opt.zero_grad()

        s1, s2 = self._net(x1), self._net(x2)
        loss = ranknet_loss(s1, s2, t)
        loss.backward()
        self.opt.step()

        return loss


@dataclass
class RankNetEvaluator(Engine):
    net: InitVar[nn.Module]
    device: torch.device = torch.device("cpu")
    compile_model: InitVar[bool] = False

    def __post_init__(self, net: nn.Module, compile_model: bool):
        super().__init__(self.inference)

        def _acc_output_transform(output: tuple[Tensor, Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor]:
            s1, s2, t, _ = output
            return (s1 - s2 > 0).long(), (t > 0.5).long()

        Average(lambda t: t[3].item()).attach(self, "loss")
        Accuracy(_acc_output_transform).attach(self, "accuracy")

        self._net = net
        if compile_model:
            try:
                self._net = torch.compile(net)
            except RuntimeError as e:
                print(f"torch.compile failed: {e}")

    @torch.no_grad()
    def inference(self, engine: Engine, batch: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x1, x2, t = prepare_batch(batch, self.device)
        self._net.eval()
        s1, s2 = self._net(x1), self._net(x2)
        return s1, s2, t, ranknet_loss(s1, s2, t)


def ranknet_loss(s1: Tensor, s2: Tensor, t: Tensor) -> Tensor:
    o = torch.sigmoid(s1 - s2)
    loss = (-t * o + F.softplus(o)).mean()
    return loss


def run_evaluator(log: dict, evaluator: Engine,
                  train_loader: DataLoader, test_loader: DataLoader):
    def _run(engine: Engine):
        _append("epoch", engine.state.epoch)

        evaluator.run(train_loader)
        _append("train", copy.deepcopy(evaluator.state.metrics))

        evaluator.run(test_loader)
        _append("test", copy.deepcopy(evaluator.state.metrics))

    def _append(key: str, value: Any):
        if key in log:
            log[key].append(value)
        else:
            log[key] = [value]

    return _run


def save_model(net: nn.Module, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    def _save(engine: Engine):
        p = save_dir / f"model_epoch-{engine.state.epoch:04d}.pt"
        torch.save(net.state_dict(), str(p))

    return _save


def get_metric(log: dict, i: int, key: tuple[str, str]) -> tuple[str, Any]:
    chunk, name = key
    return log[chunk][i][name]


def print_log(log: dict, keys: Sequence[tuple[str, str]]):
    def _print(engine: Engine):
        s = [f"epoch: {log['epoch'][-1]}"]
        for k in keys:
            val = get_metric(log, -1, k)
            s.append("{}_{}: {}".format(*k, val))

        print(", ".join(s))

    return _print


def plot_log(log: dict, y_keys: Sequence[tuple[str, str]],
             out_path: Path):
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
                        out_path: Path, device: torch.device):
    def _make_graph(engine: Engine):
        itr = iter(loader)
        x1, x2, t = prepare_batch(next(itr), device)
        net.eval()
        s1, s2 = net(x1), net(x2)
        loss = ranknet_loss(s1, s2, t)
        graph = make_dot(loss, params=dict(net.named_parameters()))
        with out_path.open("w", encoding="utf-8") as f:
            f.write(str(graph))

        print("generated computational graph")

    return _make_graph


if __name__ == "__main__":
    parse_args()
