import os
import torch
import pickle
import pandas as pd
import networkx as nx
import numpy as np
from typing import List
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from sklearn.manifold import TSNE
from torch.nn import Linear, ReLU, MSELoss
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.nn import global_mean_pool
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

class CustomGraphDataset(Dataset):
    def __init__(self, graph_list: nx.Graph,
                 static_feature_list: List[List[float or int]], 
                 label: List[float]) -> None:
        self.data_list = []
        self.static_features = []
        for i, graph in enumerate(graph_list):
            # Extract the node features and graph label from the networkx graph
            x = self._get_node_features(graph)
            y = torch.tensor(label[i])
            
            # Create a PyTorch Geometric Data object from the node features and 
            # edge list
            edge_index = torch.tensor(list(graph.edges)).t().contiguous()
            data = Data(x=x, edge_index=edge_index, y=y)
            self.data_list.append(data)
            
            # Convert the static feature to torch tensor
            self.static_features.append(torch.tensor(static_feature_list[i], \
                dtype=torch.float))
    
    def _get_node_features(self, graph:nx.Graph):
        all_node_features = []
        for node in graph.nodes():
            node_features = []
            node_features.append(graph.nodes[node]['flop_count'])
            node_features.append(graph.nodes[node]['input_count'])
            node_features.append(graph.nodes[node]['output_count'])
            node_features.append(graph.nodes[node]['avg_input_bits'])
            node_features.append(graph.nodes[node]['avg_output_bits'])
            node_features.append(graph.nodes[node]['num_logic_gates'])
            node_features.append(graph.nodes[node]['avg_logic_bits'])
            node_features.append(graph.nodes[node]['macro_count'])
            all_node_features.append(node_features)
        return torch.tensor(all_node_features, dtype=torch.float)

    def __getitem__(self, index):
        data = self.data_list[index]
        static_feature = self.static_features[index]
        return data, static_feature

    def __len__(self):
        return len(self.data_list)

class MAPELoss(torch.nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100

class GCN(torch.nn.Module):
    def __init__(self, node_feature_count: int, hidden_channels: int, 
                 static_feature_count: int):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(node_feature_count, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 9)
        self.relu1 = ReLU()
        self.fc1 = Linear(9+static_feature_count, 32)
        self.relu2 = ReLU()
        self.fc2 = Linear(32, 64)
        self.relu3 = ReLU()
        self.fc3 = Linear(64, 32)
        self.relu4 = ReLU()
        self.fc4 = Linear(32, 16)
        self.relu5 = ReLU()
        self.fc5 = Linear(16, 1)

    def forward(self, x, edge_index, batch, static_features):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        x = torch.cat([x, static_features], dim=-1)
        x = self.relu1(x)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        x = self.relu3(x)
        x = self.fc3(x)
        x = self.relu4(x)
        x = self.fc4(x)
        x = self.relu5(x)
        x = self.fc5(x)
        
        return x.squeeze()

