# -*- coding: utf-8 -*-
from torch_geometric.nn import GCNConv
import torch


from typing import Optional, Tuple, Union

from torch import Tensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import cumsum, scatter

class SAGPool(torch.nn.Module):
    def __init__(self,in_channels,ratio=0.8,Conv=GCNConv,non_linearity=torch.tanh):
        super(SAGPool,self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels,1)

        self.non_linearity = non_linearity
        
    def forward(self, x, edge_index, edge_attr=None, batch=None, mode = 1):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
       
        if mode==1:
            xx = x + torch.pow(x,2)  
        else:
            xx = x

        score = self.score_layer(xx,edge_index).squeeze()
        score_x = self.score_layer(x, edge_index).squeeze()
        score_x = torch.cat((score_x.unsqueeze(1),score.unsqueeze(1)),dim=1)

        min_value = torch.min(score)
        score = score - min_value
        
        perm = topk(score, self.ratio, batch)
        xx = xx[perm] * self.non_linearity(score[perm]).view(-1, 1)

        batch = batch[perm]

        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return xx, edge_index, edge_attr, batch, perm
    
    
def topk(
        x: Tensor,
        ratio: Optional[Union[float, int]],
        batch: Tensor,
        min_score: Optional[float] = None,
        tol: float = 1e-7,
) -> Tensor:
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter(x, batch, reduce='max')[batch] - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (x > scores_min).nonzero().view(-1)
        return perm

    if ratio is not None:
        num_nodes = scatter(batch.new_ones(x.size(0)), batch, reduce='sum')

        if ratio >= 1:
            k = num_nodes.new_full((num_nodes.size(0),), int(ratio))
        else:
            k = (float(ratio) * num_nodes.to(x.dtype)).ceil().to(torch.long)

        x, x_perm = torch.sort(x.view(-1), descending=True)
        batch = batch[x_perm]
        batch, batch_perm = torch.sort(batch, descending=False, stable=True)

        arange = torch.arange(x.size(0), dtype=torch.long, device=x.device)
        ptr = cumsum(num_nodes)
        batched_arange = arange - ptr[batch]
        mask = batched_arange < k[batch]

        return x_perm[batch_perm[mask]]


def filter_adj(
        edge_index: Tensor,
        edge_attr: Optional[Tensor],
        node_index: Tensor,
        cluster_index: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if cluster_index is None:
        cluster_index = torch.arange(node_index.size(0),
                                     device=node_index.device)

    mask = node_index.new_full((num_nodes,), -1)
    mask[node_index] = cluster_index

    row, col = edge_index[0], edge_index[1]
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return torch.stack([row, col], dim=0), edge_attr