# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
import torch


'''
find triangle clique
'''
def detect_tri(self, G):
    adj = list(G.edges())
    d = dict(nx.degree(G))
    avg_degree = sum(d.values())/len(G.nodes)
    X = {} 
    Tri = []
    i = 1
    for e in adj:
        star_u = []
        star_v = []
        Tri_e = []

        u_neighbor = []
        for w in G.neighbors(e[0]):
            u_neighbor.append(w)
            if w == e[1]:
                continue
            star_u.append((e[0], e[1], w))
            X[w] = 1

        v_neighbor = []
        for w in G.neighbors(e[1]):
            v_neighbor.append(w)
            if w == e[0]:
                continue

            if X.get(w) and X[w] == 1:
                Tri_e.append((e[0], e[1], w))
                star_u.remove((e[0], e[1], w))
            else:
                star_v.append((e[0], e[1], w))
        
        for k,v in X.items():
            X[k] = 0
        
        i += 1

        if Tri_e:
            for Tri_ in Tri_e:
                temp_Tri = sorted(Tri_)
                if temp_Tri:
                    if G.degree(temp_Tri[0]) >= avg_degree and G.degree(temp_Tri[1]) >= avg_degree and G.degree(temp_Tri[2]) >= avg_degree:
                        Tri.append((temp_Tri[0], temp_Tri[1], temp_Tri[2]))
                        
    Tri_set = set(Tri)
    Tri = list(Tri_set)
    return Tri

'''
Transfer original graph to higher-order graph
'''
def node2line_HON(self, G, node_feature, flag):
    Tri = detect_tri(G)
    node2three_clique = {idx: sublist for idx, sublist in zip(range(len(Tri)), Tri)}
    new_num = len(Tri)
    
    edge_index = []
    feature_num = node_feature.shape[1]
    node_feature_new = np.zeros((len(node2three_clique), feature_num)) #线图节点*feature
    
    for node in node2three_clique.keys():
        node_set = node2three_clique[node]
        
        temp_node_feature = (node_feature[node_set[0], :]  
                             + node_feature[node_set[1], :] + node_feature[node_set[2], :]) / 3
        node_feature_new[node, :] = temp_node_feature

    line_Tri = []
    line_Tri_attr = []
    for node_i in range(new_num):
        for node_j in range(node_i, new_num):
            if node_i == node_j:
                continue
            
            node_i_Tri = node2three_clique[node_i]
            node_j_Tri = node2three_clique[node_j]

            inter_ij_set = set(node_i_Tri).intersection(set(node_j_Tri))

            diff_i_Tri = set(node_i_Tri) - inter_ij_set
            diff_i_Tri = (list(diff_i_Tri))[0]
            diff_j_Tri = set(node_j_Tri) - inter_ij_set
            diff_j_Tri = (list(diff_j_Tri))[0]
            
            if len(set(node_i_Tri + node_j_Tri)) == 4 and G.has_edge(diff_i_Tri, diff_j_Tri):
                
                line_Tri.append([node_i, node_j])
                
                new_weight = (G.edges[node_i_Tri[0], node_i_Tri[1]]['weight'] +G.edges[node_i_Tri[0], node_i_Tri[2]]['weight'] + G.edges[node_i_Tri[1], node_i_Tri[2]]['weight'] \
                             + G.edges[node_j_Tri[0], node_j_Tri[1]]['weight'] + G.edges[node_j_Tri[0], node_j_Tri[2]]['weight'] + G.edges[node_j_Tri[1], node_j_Tri[2]]['weight']) / 6
                line_Tri_attr.append(new_weight)

    edge_index = np.array(line_Tri)
    edge_index = torch.tensor(edge_index, dtype = torch.long)
    line_node_feature = torch.tensor(node_feature_new, dtype = torch.float)

    min_value = min(line_Tri_attr)
    max_value = max(line_Tri_attr)
    
    normalized_data = [(x - min_value) / (max_value - min_value) for x in line_Tri_attr]
    line_Tri_attr = normalized_data

    edge_attr = np.array(line_Tri_attr)
    edge_attr = torch.tensor(edge_attr, dtype = torch.double)

    new2clique_arr = np.empty((len(node2three_clique), 4), dtype=int)
    
    for i, (key, (value1, value2, value3)) in enumerate(node2three_clique.items()):
        new2clique_arr[i] = [key, value1, value2, value3]
    new2clique_arr = torch.tensor(new2clique_arr, dtype = torch.long)
    
    return edge_index, line_node_feature, new2clique_arr, edge_attr