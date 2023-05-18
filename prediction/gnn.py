from typing import Any, List, Union, Optional, Dict, Tuple
import networkx as nx
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import pickle
import os
import time

## Used for reproducibility of pytorch gpu results
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
torch.use_deterministic_algorithms(True)
from gen_graph import gen_graph_from_netlist
from torch.nn import Linear, ReLU
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, GCNConv, GATConv, \
    TransformerConv, global_mean_pool, global_max_pool, global_add_pool

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, AsyncHyperBandScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch


def get_node_config(input:int, hLayerCount:int, minP:int = 2, 
                    maxP:int = 7)-> List[int]:
    '''
    This function generates configuration for the hidden layers of the FC layer  
    input: number of input features  
    hLayerCount: number of hidden layers  
    minP: minimum number of nodes in a hidden layer  
    maxP: maximum number of nodes in a hidden layer  
    '''
    # Find the number which is power of 2 and is the closest to the input
    P = int(np.ceil(np.log2(input)))
    maxLayerNodeP = min(int((hLayerCount + minP + P) / 2), maxP)
    if maxLayerNodeP <= P:
        maxLayerNodeP = P + 1
    
    ## Number of times P will increase
    incrP = maxLayerNodeP - P
    
    ## Number of times P will decrease
    dcrP = min(maxLayerNodeP - minP + 1, hLayerCount - incrP)
    
    ## Number of times P will remain same
    sameP = 0
    if hLayerCount > incrP + dcrP:
        sameP = hLayerCount - incrP - dcrP
    
    layer = []
    for _ in range(incrP):
        layer.append(2**P)
        P += 1
    
    for _ in range(sameP):
        layer.append(2**P)
    
    for _ in range(dcrP):
        layer.append(2**P)
        P -= 1

    return layer

class MAPELoss(torch.nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100

class GCN(torch.nn.Module):
    def __init__(self, node_feature_count:int, 
                 hidden_channels:int,
                 conv_type:int = 0,
                 conv_layer_count:int = 3,
                 pool_type:int = 0,
                 fc_layer_count:int = 5,
                 fc_layer_activation:int = 0):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.gcn_layers = torch.nn.ModuleList()
        self.fc_layers = torch.nn.ModuleList()
        self.pool_type = pool_type
        # Now we are not using this data
        self.fc_layer_activation = fc_layer_activation
        
        if conv_type == 0:
            self.gcn_layers.append(GCNConv(node_feature_count, 
                                           hidden_channels))
            if conv_layer_count > 1:
                for _ in range(conv_layer_count - 1):
                    self.gcn_layers.append(GCNConv(hidden_channels,
                                                   hidden_channels))
        elif conv_type == 1:
            num_heads = 2
            self.gcn_layers.append(GATConv(node_feature_count, 
                                           hidden_channels,
                                           heads=num_heads))
            if conv_layer_count > 2:
                for _ in range(conv_layer_count - 2):
                    self.gcn_layers.append(GATConv(hidden_channels*num_heads, 
                                                   hidden_channels,
                                                   heads=num_heads))
            
            self.gcn_layers.append(GATConv(hidden_channels*num_heads, 
                                           hidden_channels, heads=1))
        
        elif conv_type == 2:
            self.gcn_layers.append(GraphConv(node_feature_count, 
                                             hidden_channels))
            
            if conv_layer_count > 1:
                for _ in range(conv_layer_count - 1):
                    self.gcn_layers.append(GraphConv(hidden_channels, 
                                                     hidden_channels))
        
        elif conv_type == 3:
            self.gcn_layers.append(TransformerConv(node_feature_count, 
                                                   hidden_channels))
            
            if conv_layer_count > 1:
                for _ in range(conv_layer_count - 1):
                    self.gcn_layers.append(TransformerConv(hidden_channels, 
                                                           hidden_channels))
        
        layer_config = get_node_config(hidden_channels, fc_layer_count)
        prev_layer_node_count = hidden_channels
        for node_count in layer_config:
            self.fc_layers.append(Linear(prev_layer_node_count, 
                                         node_count))
            prev_layer_node_count = node_count
        
        self.fc_layers.append(Linear(prev_layer_node_count, 1))
        self.fc_layers_count = len(self.fc_layers)
        

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # GCN Layers
        for layer in self.gcn_layers:
            x = layer(x, edge_index)
            x = x.relu()
        
        # Pooling Layer
        if self.pool_type == 0:
            x = global_mean_pool(x, batch)
        elif self.pool_type == 1:
            x = global_max_pool(x, batch)
        elif self.pool_type == 2:
            x = global_add_pool(x, batch)
        
        # FC Layers
        for i in range(self.fc_layers_count - 1):
            x = self.fc_layers[i](x)
            x = x.relu()
        
        x = self.fc_layers[-1](x)
        
        return x.squeeze()

class CustomGraphDataset(Dataset):
    def __init__(self, graph_list: List[nx.Graph],
                 static_feature_list: List[List[Union[int, float]]],
                 label: List[float]):
        self.data_list = []
        for i, graph in enumerate(graph_list):
            # Extract the node features and graph label from the networkx graph
            # Also embed the static features to node featuers
            x = self._get_node_features(graph, static_feature_list[i])
            y = torch.tensor(label[i])
            
            # Create a PyTorch Geometric Data object from the node features and edge list
            edge_index = torch.tensor(list(graph.edges)).t().contiguous()
            data = Data(x=x, edge_index=edge_index, y=y)
            self.data_list.append(data)
    
    def _get_node_features(self, graph:nx.Graph, 
                           static_features:List[Union[int, float]]):
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
            
            ## Including the static feature as node feature
            for feat in static_features:
                node_features.append(feat)
            all_node_features.append(node_features)
        
        return torch.tensor(all_node_features, dtype=torch.float)
            
    def __getitem__(self, index:int):
        data = self.data_list[index]
        return data

    def __len__(self):
        return len(self.data_list)

def model_train(model,
          train_loader:DataLoader, 
          optimizer, 
          loss_fn, 
          epoch:int, 
          device,
          scheduler:Optional[ReduceLROnPlateau] = None,
          validation_loader:Optional[DataLoader] = None,
          isPrint:bool = False) -> float:
    
    # model = model.to(device)
    train_error = 0.0
    best_valid_error = 100.0
    best_model = None
    
    for e in range(epoch):
        model.train()
        running_loss = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, data.y)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()

        train_error = running_loss / len(train_loader)
        
        model.eval()
        val_error = 0.0
        if validation_loader is not None:
            running_loss = 0.0
            for data in validation_loader:
                data = data.to(device)
                output = model(data)
                loss = loss_fn(output, data.y)
                running_loss += loss.item()
            val_error = running_loss / len(validation_loader)
            if val_error < best_valid_error:
                best_valid_error = val_error
                best_model = model.state_dict()
        
        if scheduler is not None:
            scheduler.step(val_error)
        
        if isPrint:
            print('Epoch: {}, Train Loss: {}, Validation Loss:{}'\
                .format(e+1, train_error, val_error))
    
    if best_model is not None:
        model.load_state_dict(best_model)
        
    return train_error

def model_eval(model, loss_fn, data_loader:DataLoader, 
               device) -> Tuple[float, ...]:
    model.eval()
    outs = []
    labels = []
    running_loss = 0.0
    for data in data_loader:
        data = data.to(device)
        output = model(data)
        outs.append(output.cpu().detach().numpy())
        labels.append(data.y.cpu().numpy())
        loss = loss_fn(output, data.y)
        running_loss += loss.item()

    outs_np:NDArray[np.float64] = np.concatenate(outs)
    labels_np:NDArray[np.float64] = np.concatenate(labels)
    
    error_np = 100*np.abs(outs_np - labels_np)/labels_np
    
    mean_ape = np.mean(error_np)
    max_ape = np.max(error_np)
    return running_loss / len(data_loader), mean_ape, max_ape

def get_graph_list(df:pd.DataFrame, metrics:str,
                file_to_graph_map:Dict[str, nx.Graph], 
                netlist_dir:str = '../generic_netlist') \
                -> tuple[list[nx.Graph], list[list[Union[int, float]]], \
                    list[float]]:
    train_graph_list = []
    train_static_features = []
    y = []
    
    for _, row in df.iterrows():
        design = row['benchmark']
        size = row['size']
        util = row['util']
        num_cycle = row['num_cycle']
        num_unit = row['num_unit']
        energy = row[metrics]
        bit = row['bit_width']
        ip_bit = int(bit/2)
        tcp = row['target_clock_frequency(GHz)']
        netlist = f'{netlist_dir}/accelerator_{design}_{size}_{num_cycle}_'\
            f'{ip_bit}_{num_unit}.v'
        
        if not os.path.isabs(netlist):
            netlist = os.path.abspath(netlist)
            
        if netlist not in file_to_graph_map:
            # print(netlist)
            G, _ = gen_graph_from_netlist(netlist)
            file_to_graph_map[netlist] = G
        
        train_graph_list.append(file_to_graph_map[netlist])
        train_static_features.append([tcp, util, bit, size, num_cycle])
        y.append(energy)
    return train_graph_list, train_static_features, y

def load_data(train_csv:str, 
              metric:str, 
              netlist_dir:str, 
              file_to_graph_map_file:str) -> tuple[list[nx.Graph], 
                                                list[list[Union[int, float]]], 
                                                list[float]]:
    file_to_graph_map:Dict[str, nx.Graph] = {}
    
    if os.path.exists(file_to_graph_map_file):
        with open(file_to_graph_map_file, 'rb') as f:
            file_to_graph_map = pickle.load(f)
    
    train_df = pd.read_csv(train_csv)
    
    train_graph_list, train_static_features, train_y = get_graph_list(train_df,\
                                metric, file_to_graph_map, netlist_dir)
    
    if not os.path.exists(file_to_graph_map_file):
        if not os.path.exists(os.path.dirname(file_to_graph_map_file)):
            os.makedirs(os.path.dirname(file_to_graph_map_file))
        
        with open(file_to_graph_map_file, 'wb') as f:
            pickle.dump(file_to_graph_map, f)
    
    return train_graph_list, train_static_features, train_y


def train_svm(train_csv:str, test_csv:str, metric:str, 
              train_graph_file:str, test_graph_file:str,
              netlist_dir:str, config:dict, isPrint:bool = False):
    
    train_graph_list, train_static_features, train_y = \
        load_data(train_csv, metric, netlist_dir, train_graph_file)
    
    test_graph_list, test_static_features, test_y = \
        load_data(test_csv, metric, netlist_dir, test_graph_file)
    
    lr = config['lr']
    batch_size = config['batch_size']
    epoch = config['epoch']
    
    train_data = CustomGraphDataset(train_graph_list, train_static_features,
                                    train_y)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    test_data = CustomGraphDataset(test_graph_list, test_static_features,
                                   test_y)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model = GCN(train_data[0].num_node_features, batch_size, 
                conv_layer_count=config['conv_layer_count'],
                conv_type=config['conv_type'],
                fc_layer_count=config['fc_layer_count']).to(device)
    
    loss_fn = MAPELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                        factor=0.7, patience=5, min_lr=1e-6)
    
    _ = model_train(model, train_loader, optimizer, loss_fn, epoch,
                             device, scheduler, test_loader, isPrint)
    
    _, meanAPE, maxAPE = model_eval(model, loss_fn, test_loader, device)
    
    if isPrint:
        print(f"Validation meanAPE:{meanAPE}, maxAPE:{maxAPE}")
    
    loss = meanAPE/3.0 + maxAPE/10.0
    
    # if checkpoint_dir is not None:
    #     with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
    #         path = os.path.join(checkpoint_dir, "checkpoint")
    #         torch.save((model.state_dict(), optimizer.state_dict()), path)
    
    return loss, meanAPE, maxAPE

class tuneGCN:
    def __init__(self, train_csv:str, test_csv:str, metric:str, 
                train_graph_file:str, test_graph_file:str,
                netlist_dir:str):
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.metric = metric
        self.train_graph_file = train_graph_file
        self.test_graph_file = test_graph_file
        self.netlist_dir = netlist_dir
        self.local_dir = './log'
        
        if not os.path.exists(self.local_dir):
            os.makedirs(self.local_dir)
        
        self.config = {
            "lr": tune.loguniform(1e-5, 1e-1),
            "batch_size": tune.choice([16, 32, 64]),
            "epoch": tune.choice([100, 200, 300, 400]),
            "conv_layer_count": tune.choice([1, 2, 3, 4, 5, 6]),
            "conv_type": tune.choice([0, 1, 2]),
            "fc_layer_count": tune.choice([2, 3, 4, 5, 6, 7, 8, 9])
        }
        
        self.initla_config = [
            {"lr": 0.01,
            "batch_size": 32,
            "epoch": 300,
            "conv_layer_count": 3,
            "conv_type": 0,
            "fc_layer_count": 5},
            {"lr": 0.001,
            "batch_size": 32,
            "epoch": 200,
            "conv_layer_count": 5,
            "conv_type": 0,
            "fc_layer_count": 5}]
        
        self.algo = HyperOptSearch(points_to_evaluate=self.initla_config)
        self.algo = ConcurrencyLimiter(self.algo, max_concurrent=8)
        self.scheduler = AsyncHyperBandScheduler()
        self.local_dir = os.path.join(os.getcwd(), "tune_results")
        self.num_samples = 50
    
    def autotuneObjective(self, config):
        val_loss, meanAPE, maxAPE = train_svm(self.train_csv, self.test_csv, 
                                            self.metric, self.train_graph_file, 
                                            self.test_graph_file, 
                                            self.netlist_dir, config)
        
        tune.report(loss = val_loss, meanAPE = meanAPE, maxAPE = maxAPE)
    
    def __call__(self):
        start = time.time()
        analysis = tune.run(self.autotuneObjective,
                            metric="loss",
                            mode="min",
                            resources_per_trial={"cpu": 4, "gpu": 1},
                            search_alg = self.algo,
                            scheduler = self.scheduler,
                            num_samples = self.num_samples,
                            config = self.config,
                            local_dir = self.local_dir)
        end = time.time()
        print(f"Total time taken: {end-start} seconds")
        
        best_cost = analysis.best_result['loss']
        best_config = analysis.best_config
        print(f"Best cost: {best_cost} meanAPE:{analysis.best_result['meanAPE']}, maxAPE:{analysis.best_result['maxAPE']}")
        print(f"Best config: {best_config}")
        return


if __name__ == '__main__':    
    train_csv = '/mnt/mwoo/sakundu/RTML/data/axiline_svm_lhs_util_20230209_train.csv'
    test_csv = '/mnt/mwoo/sakundu/RTML/data/axiline_svm_lhs_util_20230209_test.csv'
    metric = 'energy(uJ)'
    # metric = 'total_power(mW)'
    # metric = 'corea_area(um^2)'
    # metric = 'runtime(ms)'

    train_graph_file = '/mnt/mwoo/sakundu/RTML/generic_graph/gnn_model_tune/graphs/train_graph_list.pkl'
    test_graph_file = '/mnt/mwoo/sakundu/RTML/generic_graph/gnn_model_tune/graphs/test_graph_list.pkl'
    netlist_dir = '/mnt/mwoo/sakundu/RTML/generic_netlist'
    
    tune_gcn = tuneGCN(train_csv, test_csv, metric, train_graph_file, 
                       test_graph_file, netlist_dir)
    
    tune_gcn()
