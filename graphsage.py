"""Two-layer mean-aggregator GraphSAGE without PyG convolution layers."""

import argparse
import random
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class GraphData:
    x: torch.Tensor
    edge_index: torch.Tensor
    y: torch.Tensor
    split: dict[str, torch.Tensor]
    num_classes: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synthetic_graph() -> GraphData:
    x = torch.tensor([
        [1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3],
        [0.2, 0.8], [0.1, 0.9], [0.0, 1.0], [0.3, 0.7],
    ])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
             (6, 7), (7, 4), (3, 4)]
    edges += [(b, a) for a, b in edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    split = {
        "train": torch.tensor([0, 1, 4, 5]),
        "valid": torch.tensor([2, 6]),
        "test": torch.tensor([3, 7]),
    }
    return GraphData(x, edge_index, y, split, 2)


def load_ogb(name: str, root: str, max_nodes: int | None, seed: int) -> GraphData:
    from ogb.nodeproppred import PygNodePropPredDataset

    dataset = PygNodePropPredDataset(name=name, root=root)
    graph = dataset[0]
    x, y, edge_index = graph.x.float(), graph.y.view(-1).long(), graph.edge_index.long()
    edge_index = torch.cat((edge_index, edge_index.flip(0)), dim=1)
    split = {key: value.view(-1) for key, value in dataset.get_idx_split().items()}
    if max_nodes is not None and max_nodes < x.size(0):
        n = max_nodes
        keep = (edge_index[0] < n) & (edge_index[1] < n)
        x, y, edge_index = x[:n], y[:n], edge_index[:, keep]
        order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        a, b = int(0.6 * n), int(0.8 * n)
        split = {"train": order[:a], "valid": order[a:b], "test": order[b:]}
        print("Using a random 60/20/20 split for the node subset.")
    return GraphData(x, edge_index, y, split, dataset.num_classes)


class SAGEConv(nn.Module):
    """Mean neighbors, concatenate with self, then apply a learned projection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(2 * in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        num_nodes = x.size(0)

        neighbor_sum = torch.zeros_like(x)
        neighbor_sum.index_add_(0, dst, x[src])
        degree = torch.bincount(dst, minlength=num_nodes).to(x.dtype).clamp_min(1)
        neighbor_mean = neighbor_sum / degree.unsqueeze(1)

        combined = torch.cat((x, neighbor_mean), dim=1)
        return self.linear(combined)


class GraphSAGE(nn.Module):
    def __init__(self, in_channels: int, hidden: int, classes: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor, index: torch.Tensor) -> float:
    return (logits[index].argmax(1) == y[index]).float().mean().item()


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    data = synthetic_graph() if args.synthetic else load_ogb(
        args.dataset, args.data_root, args.max_nodes, args.seed
    )
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    data.x, data.edge_index, data.y = data.x.to(device), data.edge_index.to(device), data.y.to(device)
    data.split = {key: value.to(device) for key, value in data.split.items()}
    model = GraphSAGE(data.x.size(1), args.hidden, data.num_classes, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_valid = -1.0

    print(f"device={device} nodes={data.x.size(0):,} edges={data.edge_index.size(1):,}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.split["train"]], data.y[data.split["train"]])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            scores = {key: accuracy(logits, data.y, ids) for key, ids in data.split.items()}
        if scores["valid"] > best_valid:
            best_valid = scores["valid"]
            torch.save(model.state_dict(), args.output)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch={epoch:03d} loss={loss.item():.4f} "
                  f"train={scores['train']:.4f} valid={scores['valid']:.4f} test={scores['test']:.4f}")
    print(f"Best validation accuracy: {best_valid:.4f}; checkpoint: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GraphSAGE implemented with plain PyTorch")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--dataset", default="ogbn-arxiv")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--max-nodes", type=int)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output", default="best_graphsage.pt")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
