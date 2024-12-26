# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from PAE_fused import PAE

class Node_GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(Node_GCN, self).__init__()
        
        self.nhid = nhid #隐藏层神经元数
        self.nfeat = nfeat #特征数
        self.num_classes = nclass
        self.dropout_ratio = dropout
        
        self.edge_net = PAE(input_dim = 2, dropout = dropout)
        
        self.conv1 = GCNConv(self.nfeat, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        
        self.lin1 = Linear(self.nhid, self.nhid // 2)
        self.lin2 = Linear(self.nhid // 2, self.num_classes)
        
    def forward(self, x, edge_index, edgenet_input):
        edge_weight = torch.squeeze(self.edge_net(edgenet_input))
        
        x = self.conv1(x.to(torch.float32), edge_index, edge_weight)
        x = F.relu(x)
    
        x = self.conv2(x.to(torch.float32), edge_index, edge_weight)
        x = F.relu(x)
            
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training = True)
        out = self.lin2(x)
        
        return out
        
        