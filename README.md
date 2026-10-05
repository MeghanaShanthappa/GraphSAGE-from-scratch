# GraphSAGE from scratch

A two-layer GraphSAGE model with mean aggregation. Neighbor aggregation uses plain
PyTorch (`index_add_`), not `torch_geometric.nn.SAGEConv`.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Quick test

```bash
python3 graphsage.py --synthetic --epochs 50
```

## OGB experiment

```bash
python3 graphsage.py --dataset ogbn-arxiv --epochs 100
```

This educational version performs full-graph training. For very large graphs such
as `ogbn-products`, production GraphSAGE normally uses neighbor mini-batches.
