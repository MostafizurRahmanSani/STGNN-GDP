import torch
import torch.nn as nn
import torch.nn.functional as F

class MessagePassingLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        aggregation,
        combination,
        normalize=False
    ):
        super().__init__()

        self.aggregation = aggregation
        self.combination = combination
        self.normalize = normalize

        self.msg_mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU()
        )

        if combination == "gru":
            self.update_fn = nn.GRUCell(out_dim, out_dim)
        elif combination == "concat":
            self.update_fn = nn.Sequential(
                nn.Linear(in_dim+out_dim, out_dim),
                nn.ReLU()
            )
        elif combination == "add":
            self.update_fn = nn.Sequential(
                nn.Linear(out_dim, out_dim),
                nn.ReLU()
            )
        else:
            raise ValueError("Invalid Combination Type!")

    def aggregate(self, src, msg, num_nodes):
        out = torch.zeros(num_nodes, msg.size(1), device=msg.device)

        if self.aggregation == "sum":
            out.index_add_(0, src, msg)
        elif self.aggregation == "mean":
            out.index_add_(0, src, msg)
            deg = torch.bincount(src, minlength=num_nodes).clamp(min=1).unsqueeze(1)
            out = out / deg
        elif self.aggregation == "max":
            out.scatter_reduce_(0, src.unsqueeze(1).expand_as(msg), msg, reduce="amax")
        else:
            raise ValueError("Invalid Aggregation!")

        return out

    def forward(self, x, edge_index, edge_weight):
        src_idx, dst_idx = edge_index
        num_nodes = x.size(0)

        src_feat = x[src_idx]
        msg = self.msg_mlp(src_feat)

        if edge_weight is not None:
            msg = msg * edge_weight.unsqueeze(1)

        agg = self.aggregate(dst_idx, msg, num_nodes)

        if self.combination == "gru":
            out = self.update_fn(agg, x)
        elif self.combination == "concat":
            out = self.update_fn(torch.cat([x, agg], dim=1))
        elif self.combination == "add":
            out = self.update_fn(x + agg)

        if self.normalize:
            out = F.normalize(out, p=2, dim=1)

        return out

class STGNN(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        edge_dim=10,
        agg="mean",
        comb="gru",
        norm=True
    ):
        super().__init__()

        self.comb = comb
        self.pre = nn.Linear(in_dim, hidden_dim)
        

        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        self.layer1 = MessagePassingLayer(
            hidden_dim, hidden_dim, aggregation=agg,
            combination=comb, normalize=norm
        )
        self.layer2 = MessagePassingLayer(
            hidden_dim, hidden_dim, aggregation=agg,
            combination=comb, normalize=norm
        )

       
        self.temporal_gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            batch_first=False
        )

        self.post = nn.Linear(hidden_dim, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, 3)

    def forward(self, xs, edge_indices, edge_attrs=None):
        T, N, _ = xs.shape
        spatial_out = []

        for t in range(T):
            xt = xs[t]
            xt = F.relu(self.pre(xt))

            edge_index = edge_indices[t]

          
            edge_weight = None
            if edge_attrs is not None:
                edge_weight = self.edge_proj(edge_attrs[t]).squeeze(-1)

            h1 = self.layer1(xt, edge_index, edge_weight)
            xt = h1 if self.comb == "gru" else xt + h1

            h2 = self.layer2(xt, edge_index, edge_weight)
            xt = h2 if self.comb == "gru" else xt + h2

            spatial_out.append(xt)

   
        h = torch.stack(spatial_out, dim=0)  
        h, _ = self.temporal_gru(h)          
        h = h[-1, :, :]                    

        h = F.relu(self.post(h))
        return self.regressor(h)             