import torch
import torch.nn as nn
from torch_geometric.nn.conv import GATConv

from typing import Optional


class GATDiGraphNet(nn.Module):
    def __init__(
        self,
        num_layers: int,
        node_features: int,
        hidden_features: int,
        heads: int,
        concat: Optional[bool] = True,
        edge_features: Optional[int] = None,
    ):
        super().__init__()

        self.W1 = nn.Linear(node_features, hidden_features, bias=True)
        self.W2 = nn.Linear(hidden_features, hidden_features, bias=True)

        self.gat_fw = LayeredGAT(
            num_layers=num_layers,
            hidden_features=hidden_features,
            heads=heads,
            concat=concat,
            edge_dim=edge_features,
            flow="source_to_target",
        )
        self.gat_bw = LayeredGAT(
            num_layers=num_layers,
            hidden_features=hidden_features,
            heads=heads,
            concat=concat,
            edge_dim=edge_features,
            flow="target_to_source",
        )

        self.scorer = nn.Linear(2 * hidden_features, out_features=1, bias=True)

    def forward(self, x, edge_index, edge_attr=None):
        h = self.W1(x)
        h = torch.relu(h)
        h = self.W2(h)
        h_fw = self.gat_fw(x=h, edge_index=edge_index)
        h_bw = self.gat_bw(x=h, edge_index=edge_index)
        score = self.scorer(torch.cat([h_fw, h_bw], dim=1))

        return score


class LayeredGAT(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_features: int,
        flow: str,
        heads: int,
        concat: Optional[bool] = True,
        edge_dim: Optional[int] = None,
    ):
        super().__init__()

        self.use_edge_features = False if edge_dim is None else True
        if concat:
            self.gnn = nn.ModuleList(
                [
                    GATMultiHeadBlock(
                        hidden_features=hidden_features,
                        flow=flow,
                        heads=heads,
                        concat=concat,
                        edge_dim=edge_dim,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.gnn = nn.ModuleList(
                [
                    GATConv(
                        in_channels=hidden_features,
                        out_channels=hidden_features,
                        heads=heads,
                        concat=concat,
                        edge_dim=edge_dim,
                        flow=flow,
                    )
                    for _ in range(num_layers)
                ]
            )
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_attr=None):
        if not self.use_edge_features:
            edge_attr = None
        for idx in range(len(self.gnn) - 1):
            x = self.gnn[idx](x=x, edge_index=edge_index, edge_attr=edge_attr)
            x = self.relu(x)
        h = self.gnn[-1](x=x, edge_index=edge_index, edge_attr=edge_attr)
        return h


class GATMultiHeadBlock(nn.Module):
    def __init__(
        self,
        hidden_features: int,
        flow: str,
        heads: int,
        concat: bool = True,
        edge_dim: Optional[int] = None,
    ):
        super().__init__()

        self.use_edge_features = False if edge_dim is None else True
        self.gat = GATConv(
            in_channels=hidden_features,
            out_channels=hidden_features,
            heads=heads,
            concat=concat,
            edge_dim=edge_dim,
            flow=flow,
        )
        self.lin = nn.Linear(in_features=hidden_features * heads, out_features=hidden_features, bias=True)

    def forward(self, x, edge_index, edge_attr=None):
        if not self.use_edge_features:
            edge_attr = None
        h = self.gat(x=x, edge_index=edge_index, edge_attr=edge_attr)
        h = self.lin(h)
        return h
