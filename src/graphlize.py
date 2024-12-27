"""
Project: TaCo
File: graphlize.py
Description: This module provides functions to graphlize the data.

Author:  Lance
Created: 2024-12-19
Email: lance.lz.kong@gmail.com
"""
import torch

# If the program crashes, the following four lines related to torch_geometric might be the issue.
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GINEConv
from torch_geometric.nn import LayerNorm
from torch_geometric.nn import global_add_pool, global_mean_pool

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import plot,savefig
import pandas as pd

def get_graph(net_detail, inference_details) -> tuple[Data, Data]:
    '''Generate network graph and inference graph
    return: net_graph, inference_graph
    '''
    # Network node features
    net_node_feature = list(net_detail['net_node_feature']) # This is GFlops
    net_node_feature = torch.tensor(net_node_feature, dtype=torch.float)

    # Network adjacency matrix
    net_edge_index = list(net_detail['net_node_edge_index'])
    net_edge_index = torch.tensor(net_edge_index, dtype=torch.long)
    net_edge_index = net_edge_index.t().contiguous()

    # Network edge features
    net_edge_features = list(net_detail['net_node_edge_features']) # -> 
    net_edge_features = torch.tensor(net_edge_features, dtype=torch.float)

    net_graph = Data(x=net_node_feature, edge_index=net_edge_index, edge_attr = net_edge_features) # Network graph

    '''Above is net_graph, below is inference_graph'''

    # Inference model node features
    infe_node_feature = list(inference_details['comp_Mflops'])
    infe_node_feature = torch.tensor(infe_node_feature, dtype=torch.float)

    # Inference model adjacency matrix
    infe_edge_index = list(inference_details['inference_edge'])
    infe_edge_index = torch.tensor(infe_edge_index, dtype=torch.long)
    infe_edge_index = infe_edge_index.t().contiguous()

    # Inference model edge features
    infe_edge_features = list(inference_details['commu_kbit'])
    infe_edge_features = torch.tensor(infe_edge_features, dtype=torch.float)

    inference_graph = Data(x=infe_node_feature, edge_index=infe_edge_index, edge_attr = infe_edge_features) # Inference graph

    return net_graph, inference_graph


# if __name__ == '__main__':

#     pass