import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn.conv import GATConv


class GATMultiDiGraphNet(nn.Module):
    def __init__(
        self,
        num_layers: int,
        node_features: int,
        edge_features: int,
        hidden_features: int,
        graph_features: int = -1,
        gate: str = "bidirect",
        heads: int = 3,
        concat: bool = True,
        layer_norm: bool = True,
    ):
        super().__init__()

        self.graph_features = graph_features

        self.W11 = nn.Linear(node_features, hidden_features, bias=True)
        self.W12 = nn.Linear(hidden_features, hidden_features, bias=True)
        self.W21 = nn.Linear(edge_features, hidden_features, bias=True)
        self.W22 = nn.Linear(hidden_features, hidden_features, bias=True)
        if self.graph_features > 0:
            self.W31 = nn.Linear(graph_features, hidden_features, bias=True)
            self.W32 = nn.Linear(hidden_features, hidden_features, bias=True)

        self.gate = LayeredGAT(
            num_layers=num_layers,
            hidden_features=hidden_features,
            gate=gate,
            heads=heads,
            concat=concat,
            layer_norm=layer_norm,
        )

        self.scorer1 = nn.Linear(3 * hidden_features, hidden_features, bias=True)
        if self.graph_features > 0:
            scorer2_features = 2 * hidden_features
        else:
            scorer2_features = hidden_features

        self.scorer2 = nn.Linear(scorer2_features, out_features=1, bias=True)
        # this is unnecessary but removing it requires modification of tests
        self.scorer = nn.Linear(3 * hidden_features, out_features=1, bias=True)
        self.reset_parameters()

    def forward(self, x, edge_attr, edge_index, graph_attr, ei_ptr=None) -> Tensor:
        h = self.W12(torch.relu(self.W11(x)))
        e = self.W22(torch.relu(self.W21(edge_attr)))

        h, e = self.gate(h=h, edge_attr=e, edge_index=edge_index)

        src, dst = edge_index
        score = self.scorer1(torch.cat((h[src], h[dst], e), dim=1))
        score = torch.relu(score)
        if self.graph_features > 0:
            g = self.W32(torch.relu(self.W31(graph_attr)))
            # TODO: Can we remove the following two lines?
            # repeat_interleave = ei_ptr
            # features = g.repeat_interleave(repeat_interleave, dim=0)
            features = g.repeat_interleave(score.shape[0], dim=0)
            score = torch.cat([score, features], dim=1)

        score = self.scorer2(score)

        return score

    def reset_parameters(self):
        self.W11.reset_parameters()
        self.W12.reset_parameters()
        self.W21.reset_parameters()
        self.W22.reset_parameters()
        # TODO: W31 and W32 don't need to be included?
        self.gate.reset_parameters()
        self.scorer1.reset_parameters()
        self.scorer2.reset_parameters()


class LayeredGAT(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_features: int,
        gate: str = "bidirect",
        heads: int = 3,
        concat: bool = True,
        layer_norm: bool = True,
    ):
        super().__init__()
        if gate == "bidirect":
            self.gnn = nn.ModuleList(
                GATMultiHeadBlockBidirect(
                    hidden_features=hidden_features,
                    heads=heads,
                    concat=concat,
                    layer_norm=layer_norm,
                )
                for _ in range(num_layers)
            )
        elif gate == "unidirect":
            self.gnn = nn.ModuleList(
                GATMultiHeadBlockUnidirect(
                    hidden_features=hidden_features,
                    heads=heads,
                    concat=concat,
                    layer_norm=layer_norm,
                )
                for _ in range(num_layers)
            )
        else:
            raise ValueError(f"Unsupportade gate option {gate}, should be one of `bidirect` or `unidirect`")

    def forward(self, h, edge_attr, edge_index):
        for idx in range(len(self.gnn) - 1):
            h = self.gnn[idx](h=h, edge_attr=edge_attr, edge_index=edge_index)
            h = torch.relu(h)
        h = self.gnn[-1](h=h, edge_attr=edge_attr, edge_index=edge_index)
        # Since GAT doesn't update edge features
        # The returned edge_attr is the same as the input edge_ettr
        return h, edge_attr

    def reset_parameters(self):
        for gnn_layer in self.gnn:
            gnn_layer.reset_parameters()


class GATMultiHeadBlockUnidirect(nn.Module):
    def __init__(self, hidden_features: int, heads: int, concat: bool, layer_norm: bool = True):
        super().__init__()
        self.concat = concat
        self.layer_norm = layer_norm

        self.gat = GATConv(
            in_channels=hidden_features,
            out_channels=hidden_features,
            heads=heads,
            concat=concat,
            edge_dim=hidden_features,
            flow="source_to_target",
        )
        if self.concat:
            self.lin = nn.Linear(in_features=hidden_features * heads, out_features=hidden_features, bias=True)
        if self.layer_norm:
            self.ln = nn.LayerNorm(hidden_features)

    def forward(self, h, edge_attr, edge_index):
        h = self.gat(x=h, edge_index=edge_index, edge_attr=edge_attr)
        if self.concat:
            h = self.lin(h)
        if self.layer_norm:
            h = self.ln(h)  # TODO: Do we want LayerNorm with GAT?
        return h

    def reset_parameters(self):
        self.gat.reset_parameters()
        if self.concat:
            self.lin.reset_parameters()
        if self.layer_norm:
            self.ln.reset_parameters()


class GATMultiHeadBlockBidirect(nn.Module):
    def __init__(self, hidden_features: int, heads: int, concat: bool, layer_norm: bool = True):
        super().__init__()
        self.concat = concat
        self.layer_norm = layer_norm

        self.gat_fw = GATConv(
            in_channels=hidden_features,
            out_channels=hidden_features,
            heads=heads,
            concat=concat,
            edge_dim=hidden_features,
            flow="source_to_target",
        )
        self.gat_bw = GATConv(
            in_channels=hidden_features,
            out_channels=hidden_features,
            heads=heads,
            concat=concat,
            edge_dim=hidden_features,
            flow="target_to_source",
        )
        if self.concat:
            self.lin = nn.Linear(in_features=hidden_features * heads, out_features=hidden_features, bias=True)
        if self.layer_norm:
            self.ln = nn.LayerNorm(hidden_features)

    def forward(self, h, edge_attr, edge_index):
        h_fw = self.gat_fw(x=h, edge_index=edge_index, edge_attr=edge_attr)
        h_bw = self.gat_bw(x=h, edge_index=edge_index, edge_attr=edge_attr)
        h = h_fw + h_bw  # Simple addition of forward and backward messages
        if self.concat:
            h = self.lin(h)
        if self.layer_norm:
            h = self.ln(h)  # TODO: Do we want LayerNorm with GAT?
        return h

    def reset_parameters(self):
        self.gat_fw.reset_parameters()
        self.gat_bw.reset_parameters()
        if self.concat:
            self.lin.reset_parameters()
        if self.layer_norm:
            self.ln.reset_parameters()
