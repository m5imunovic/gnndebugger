import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing


class ResGatedMultiDiGraphNet(nn.Module):
    def __init__(
        self,
        num_layers: int,
        node_features: int,
        edge_features: int,
        hidden_features: int,
        graph_features: int = -1,
        gate: str = "bidirect",
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

        self.gate = LayeredGatedGCN(num_layers=num_layers, hidden_features=hidden_features, gate=gate)
        self.ln1 = nn.LayerNorm(hidden_features)  # TODO: Unused. Delete?
        self.ln2 = nn.LayerNorm(hidden_features)  # TODO: Unused. Delete?

        # TODO: check if increasing capacity or adding dropout here helps
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

        h, e = self.gate.forward(h=h, edge_attr=e, edge_index=edge_index)

        src, dst = edge_index
        score = self.scorer1(torch.cat(([h[src], h[dst], e]), dim=1))
        score = torch.relu(score)
        if self.graph_features > 0:
            repeat_interleave = ei_ptr
            g = self.W32(torch.relu(self.W31(graph_attr)))
            features = g.repeat_interleave(repeat_interleave, dim=0)
            # TODO: Suggestion to replace the line above with
            # features = g.repeat_interleave(score.shape[0], dim=0)
            score = torch.cat([score, features], dim=1)

        score = self.scorer2(score)

        return score

    def reset_parameters(self):
        self.W11.reset_parameters()
        self.W12.reset_parameters()
        self.W21.reset_parameters()
        self.W22.reset_parameters()
        self.gate.reset_parameters()
        self.scorer1.reset_parameters()
        self.scorer2.reset_parameters()


class LayeredGatedGCN(nn.Module):
    def __init__(self, num_layers: int, hidden_features: int, gate_norm: str = "feature", gate: str = "bidirect"):
        super().__init__()
        if gate == "bidirect":
            self.gnn = nn.ModuleList(
                GatedGCN(hidden_features=hidden_features, gate_norm=gate_norm) for _ in range(num_layers)
            )
        elif gate == "unidirect":
            self.gnn = nn.ModuleList(
                GatedGCNUni(hidden_features=hidden_features, gate_norm=gate_norm) for _ in range(num_layers)
            )
        else:
            raise ValueError(f"Unsupportade gate option {gate}, should be one of `bidirect` or `unidirect`")

    def forward(self, h, edge_attr, edge_index, **kwargs):
        for gnn_layer in self.gnn:
            h, edge_attr = gnn_layer(h=h, edge_attr=edge_attr, edge_index=edge_index)
        return h, edge_attr

    def reset_parameters(self):
        for gnn_layer in self.gnn:
            gnn_layer.reset_parameters()


class GatedGCNUni(MessagePassing):
    def __init__(self, hidden_features: int, gate_norm: str, layer_norm: bool = True):
        super().__init__(aggr="add")
        self.gate_norm = gate_norm
        self.layer_norm = layer_norm

        self.A1 = nn.Linear(hidden_features, hidden_features)
        self.A2 = nn.Linear(hidden_features, hidden_features)
        self.A3 = nn.Linear(hidden_features, hidden_features)

        self.B1 = nn.Linear(hidden_features, hidden_features)
        self.B2 = nn.Linear(hidden_features, hidden_features)
        self.B3 = nn.Linear(hidden_features, hidden_features)

        if self.layer_norm:
            self.ln_h = nn.LayerNorm(hidden_features)
            self.ln_e = nn.LayerNorm(hidden_features)

    def forward(self, h, edge_attr, edge_index):
        A1h = self.A1(h)
        A2h = self.A2(h)

        B1h = self.B1(edge_attr)
        B2h = self.B2(h)
        B3h = self.B3(h)

        src, dst = edge_index

        e_fw = B1h + B2h[src] + B3h[dst]
        e_fw = F.relu(e_fw)

        if self.layer_norm:
            e_fw = self.ln_e(e_fw)

        # residual connection
        e_fw = edge_attr + e_fw

        sigmoid_fw = torch.sigmoid(e_fw)
        h_fw = self.propagate(edge_index=edge_index, x=A2h, sigma=sigmoid_fw)

        h_new = A1h + h_fw
        h_new = F.relu(h_new)
        if self.layer_norm:
            h_new = self.ln_h(h_new)
        h = h + h_new
        return h, e_fw

    def message(self, x_j, sigma) -> Tensor:
        # in pyg (j->i) represents the flow from source to target and (i->j) the reverse
        # generally, i is the node that accumulates information and {j} its neighbors
        message = x_j * sigma
        if self.gate_norm == "feature":
            message = message / (torch.sum(sigma, dim=1).unsqueeze(dim=1) + 1e-6)
        return message


class GatedGCN(MessagePassing):
    def __init__(self, hidden_features: int, gate_norm: str, layer_norm: bool = True):
        super().__init__(aggr="add")
        self.gate_norm = gate_norm
        self.layer_norm = layer_norm

        self.A1 = nn.Linear(hidden_features, hidden_features)
        self.A2 = nn.Linear(hidden_features, hidden_features)
        self.A3 = nn.Linear(hidden_features, hidden_features)

        self.B1 = nn.Linear(hidden_features, hidden_features)
        self.B2 = nn.Linear(hidden_features, hidden_features)
        self.B3 = nn.Linear(hidden_features, hidden_features)

        if self.layer_norm:
            self.ln_h = nn.LayerNorm(hidden_features)
            self.ln_e = nn.LayerNorm(hidden_features)

    def reset_parameters(self):
        super().reset_parameters()
        self.A1.reset_parameters()
        self.A2.reset_parameters()
        self.A3.reset_parameters()
        self.B1.reset_parameters()
        self.B2.reset_parameters()
        self.B3.reset_parameters()

    def forward(self, h, edge_attr, edge_index):
        A1h = self.A1(h)
        A2h = self.A2(h)
        A3h = self.A3(h)

        B1h = self.B1(edge_attr)
        B2h = self.B2(h)
        B3h = self.B3(h)

        src, dst = edge_index
        bw_edge_index = torch.vstack((dst, src))

        e_fw = B1h + B2h[src] + B3h[dst]
        e_bw = B1h + B2h[dst] + B3h[src]

        if self.layer_norm:
            e_fw = self.ln_e(e_fw)
            e_bw = self.ln_e(e_bw)

        e_fw = F.relu(e_fw)
        e_bw = F.relu(e_bw)

        # residual connection
        e_fw = edge_attr + e_fw
        e_bw = edge_attr + e_bw

        node_size = (h.size(0), h.size(0))
        sigma_fw = torch.sigmoid(e_fw)
        h_fw_sum = self.propagate(edge_index=edge_index, x=A2h, edge_weight=sigma_fw, size=node_size)
        h_fw_sigma_sum = self.propagate(edge_index=edge_index, x=None, edge_weight=sigma_fw, size=node_size)
        h_fw = h_fw_sum / (h_fw_sigma_sum + 1e-6)

        sigma_bw = torch.sigmoid(e_bw)
        h_bw_sum = self.propagate(edge_index=bw_edge_index, x=A3h, edge_weight=sigma_bw, size=node_size)
        h_bw_sigma_sum = self.propagate(edge_index=bw_edge_index, x=None, edge_weight=sigma_bw, size=node_size)
        h_bw = h_bw_sum / (h_bw_sigma_sum + 1e-6)

        h_new = A1h + h_fw + h_bw
        if self.layer_norm:
            h_new = self.ln_h(h_new)
        h_new = F.relu(h_new)
        h = h + h_new

        return h, e_fw

    def message(self, x_j, edge_weight) -> Tensor:
        # in pyg (j->i) represents the flow from source to target and (i->j) the reverse
        # generally, i is the node that accumulates information and {j} its neighbors
        return x_j * edge_weight if x_j is not None else edge_weight
