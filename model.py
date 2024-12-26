# -*- coding: utf-8 -*-

import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import GCNConv

from scipy.spatial import distance

from Node_classification_fused import Node_GCN
from layers_fused import SAGPool


'''
Normalized 
'''
class PairNorm(torch.nn.Module):
    def __init__(self, mode = 'PN', scale = 1):
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale
        
    def forward(self, x):
        if self.mode == 'None':
            return x

        col_mean = x.mean(dim=0)
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt() 
            x = self.scale * x / rownorm_mean

        if self.mode == 'PN-SI': 
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual

        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean
            
        if self.mode == 'PN-St':
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True)

            epsilon = 1e-8 
            x_normalized = (x - mean) / (std + epsilon)
            x = x_normalized
        return x

class feature_coding(torch.nn.Module):
    def __init__(self, args):
        super(feature_coding, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.pool1 = SAGPool(self.nhid, ratio=self.pooling_ratio)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.pool2 = SAGPool(self.nhid, ratio=self.pooling_ratio)
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x = self.conv1(x.to(torch.float32), edge_index, edge_weight = edge_attr)   
        x = F.relu(x)

        x, edge_index, edge_attr, batch, perm1, score1, score_sort1, new2old1_dict = self.pool1(x = x.to(torch.float32), edge_index = edge_index, edge_attr = edge_attr, batch = batch, mode=1)  
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.conv2(x, edge_index, edge_weight = edge_attr)
        x = F.relu(x)
        x, edge_index, edge_attr, batch = self.pool2(x = x.to(torch.float32), edge_index = edge_index, edge_attr = edge_attr, batch = batch, mode=1)    # 再乘pooling ratio
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        xx = (x1 + x2) / 2  
        return xx

'''
Adjacency matrix
'''
def create_affinity_graph_from_scores(scores, pd_dict):
    num_nodes = len(pd_dict[scores[0]]) 
    graph = np.zeros((num_nodes, num_nodes))

    for l in scores:
        label_dict = pd_dict[l]

        if l in ['Age', 'FIQ']:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict[k]) - float(label_dict[j]))
                        if val < 5:  # val<2
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:  # missing label
                        pass

        else:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if label_dict[k] == label_dict[j] :#and label_dict[j]==flag:
                        graph[k, j] += 1
                        graph[j, k] += 1

    return graph


'''
Population Graph Construction
'''
def get_PAE_inputs_full(node_ftr, data, k=10):
    age = data.age.detach().cpu().numpy()
    gender = data.gender.detach().cpu().numpy()
    site = data.site.detach().cpu().numpy()
    hmmd  = data.hmmd.detach().cpu().numpy()

    num_nodes = age.shape[0]
    phonetic_data = np.zeros([num_nodes, 4], dtype=np.float32)
    phonetic_data[:,0] = age
    phonetic_data[:,1] = gender 
    phonetic_data[:,2] = site
    phonetic_data[:,3] = hmmd 
    nonimg = phonetic_data

    pd_dict = {}   
    pd_dict['age'] = np.copy(phonetic_data[:,0])
    pd_dict['gender'] = np.copy(phonetic_data[:,1])
    pd_dict['SITE_ID'] = np.copy(phonetic_data[:,2])
    pd_dict['HMMD'] = np.copy(phonetic_data[:,3]) 
  
    # construct edge network inputs 
    node_ftr = node_ftr.detach().cpu().numpy()
    n = node_ftr.shape[0] 
    num_edge = n*(1+n)//2 - n 
    pd_ftr_dim = nonimg.shape[1]
    edge_index = np.zeros([2, num_edge], dtype=np.int64) 
    edgenet_input = np.zeros([num_edge, 2*pd_ftr_dim], dtype=np.float32)  
    aff_score = np.zeros(num_edge, dtype=np.float32) 
        
    # static affinity score used to pre-prune edges 
    pd_affinity = create_affinity_graph_from_scores(['SITE_ID','HMMD'], pd_dict)
    distv = distance.pdist(node_ftr, metric='correlation') 
    dist = distance.squareform(distv)  
    sigma = np.mean(dist)
    feature_sim = np.exp(- dist ** 2 / (2 * sigma ** 2))
    aff_adj = pd_affinity*feature_sim

    aff_adj = aff_adj-np.diag(np.diag(aff_adj))
    idx = np.argsort(-aff_adj,axis=1)
    idx = idx[:,0:k]
    aff_new = np.zeros_like(aff_adj)
    for i in range(num_nodes):
        aff_new[i,idx[i,:]] = 1

    aff_adj = aff_adj*aff_new

    flatten_ind = 0
    for i in range(n):
        for j in range(i+1, n):
            edge_index[:,flatten_ind] = [i,j]
            edgenet_input[flatten_ind]  = np.concatenate((nonimg[i], nonimg[j]))
            aff_score[flatten_ind] = aff_adj[i][j]  
            flatten_ind +=1

    assert flatten_ind == num_edge, "Error in computing edge input"

    ordered_aff_score = sorted(aff_score)
    avg_aff_score = sum(ordered_aff_score) / len(ordered_aff_score)
    keep_ind = np.where(ordered_aff_score > avg_aff_score)[0]

    edge_index = edge_index[:, keep_ind]
    edgenet_input = edgenet_input[keep_ind]

    edgenet_input = (edgenet_input- edgenet_input.mean(axis=0)) / (edgenet_input.std(axis=0)+1e-8)
    return edge_index, edgenet_input

'''
Affinity-separability feature module
'''
def compute_fea_loss(batch_fea, batch_label):
    pos_feature = []
    neg_feature = []
    batch_fea = batch_fea.cpu()
    for i in range(len(batch_label)):
        n_label = batch_label[i]
        indices = torch.tensor([i])
        select_feature = torch.index_select(batch_fea, 0, indices)

        if n_label == 1:
            pos_feature.append(select_feature)
        else:
            neg_feature.append(select_feature)
        
    same_loss = 0.0
    iters = 0
    for i in range(len(pos_feature)):
        for j in range(i, len(pos_feature)):
            if i == j:
                continue
            
            vector1 = (pos_feature[i][0]).cpu().detach().numpy()
            vector2 = (pos_feature[j][0]).cpu().detach().numpy()
            
            dot_product = np.dot(vector1, vector2)
            if dot_product == 0:
                euclidean_distance = 0
            else:
                norm_vector1 = np.linalg.norm(vector1)
                norm_vector2 = np.linalg.norm(vector2)
                euclidean_distance = dot_product / (norm_vector1 * norm_vector2)
                    
            iters += 1
            same_loss += euclidean_distance
    try:
        same_loss = same_loss / iters
    except:
        same_loss = torch.tensor(0.0)
    
    nosame_loss = 0.0
    iters = 0
    for i in range(len(neg_feature)):
        for j in range(i, len(neg_feature)):

            vector1 = (neg_feature[i][0]).cpu().detach().numpy()
            vector2 = (neg_feature[j][0]).cpu().detach().numpy()
            
            dot_product = np.dot(vector1, vector2)
            if dot_product == 0:
                euclidean_distance = 0
            else:
                norm_vector1 = np.linalg.norm(vector1)
                norm_vector2 = np.linalg.norm(vector2)
                euclidean_distance = dot_product / (norm_vector1 * norm_vector2)

            iters += 1
            nosame_loss += euclidean_distance
    try:
        nosame_loss = nosame_loss / iters
    except:
        nosame_loss = torch.tensor(0.0)
    
    difflabel_loss = 0.0
    iters = 0
    for i in range(len(pos_feature)):
        for j in range(len(neg_feature)):
            
            vector1 = (pos_feature[i][0]).cpu().detach().numpy()
            vector2 = (neg_feature[j][0]).cpu().detach().numpy()
            
            dot_product = np.dot(vector1, vector2)
            if dot_product == 0:
                euclidean_distance = 0
            else:
                norm_vector1 = np.linalg.norm(vector1)
                norm_vector2 = np.linalg.norm(vector2)
                euclidean_distance = dot_product / (norm_vector1 * norm_vector2)

            iters += 1
            difflabel_loss += euclidean_distance
    try:
        difflabel_loss = difflabel_loss / iters
    except:
        difflabel_loss = torch.tensor(0.0)
    
    out_loss = (same_loss + nosame_loss) / 3
    out_loss = out_loss.item()
    difflabel_loss = difflabel_loss.item()
    return out_loss, difflabel_loss
