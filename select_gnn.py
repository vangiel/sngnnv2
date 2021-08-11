import torch
import torch.nn as nn
from nets.rgcnDGL import RGCN
from nets.gat import GAT
from nets.mpnn_dgl import MPNN
import dgl

class SELECT_GNN(nn.Module):
    def __init__(self, num_features, num_edge_feats, n_classes, num_hidden, gnn_layers, dropout,
                 activation, final_activation, num_channels, gnn_type, K, num_heads, num_rels, num_bases, g, residual,
                 aggregator_type, attn_drop, num_hidden_layers_rgcn, num_hidden_layers_gat, num_hidden_layer_pairs,
                 improved=True, concat=True, neg_slope=0.2, bias=True, norm=None, alpha=0.12, grid_nodes = 0):
        super(SELECT_GNN, self).__init__()

        self.activation = activation
        if final_activation == 'relu':
            self.final_activation = torch.nn.ReLU()
        elif final_activation == 'tanh':
            self.final_activation = torch.nn.Tanh()
        elif final_activation == 'sigmoid':
            self.final_activation = torch.nn.Sigmoid()
        else:
            self.final_activation = None
        self.num_hidden_layer_pairs = num_hidden_layer_pairs
        self.attn_drop = attn_drop
        self.num_hidden_layers_rgcn = num_hidden_layers_rgcn
        self.num_hidden_layers_gat = num_hidden_layers_gat
        self.num_rels = num_rels
        self.residual = residual
        self.aggregator = aggregator_type
        self.num_bases = num_bases
        self.num_channels = num_channels
        self.n_classes = n_classes
        self.num_hidden = num_hidden
        self.gnn_layers = gnn_layers
        self.num_features = num_features
        self.num_edge_feats = num_edge_feats
        self.dropout = dropout
        self.bias = bias
        self.norm = norm
        self.improved = improved
        self.K = K
        self.g = g
        self.num_heads = num_heads
        self.concat = concat
        self.neg_slope = neg_slope
        self.dropout1 = dropout
        self.alpha = alpha
        self.gnn_type = gnn_type
        self.robot_node_indexes = []
        self.grid_nodes = grid_nodes


        if self.dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Dropout(p=0.)
        #
        if self.gnn_type == 'rgcn':
            print("GNN being used is RGCN")
            self.gnn_object = self.rgcn()
        elif self.gnn_type == 'gat':
            print("GNN being used is GAT")
            self.gnn_object = self.gat()
        elif self.gnn_type == 'mpnn':
            self.gnn_object = self.mpnn()

    def rgcn(self):
        return RGCN(self.g, self.gnn_layers, self.num_features, self.n_classes, self.num_hidden, self.num_rels, self.activation, self.final_activation, self.dropout1, self.num_bases)

    def gat(self):
        return GAT(self.g, self.gnn_layers, self.num_features, self.n_classes, self.num_hidden, self.num_heads, self.activation, self.final_activation,  self.dropout1, self.attn_drop, self.alpha, self.residual)

    def mpnn(self):
        return MPNN(self.num_features, self.n_classes, self.num_hidden, self.num_edge_feats, self.final_activation, self.aggregator, self.bias, self.residual, self.norm, self.activation)

    def forward(self, data, g, efeat):
        if self.gnn_type in ['gatmc', 'prgat2', 'prgat3']:
            x = self.gnn_object(data)
        elif self.gnn_type == 'mpnn':
            x = self.gnn_object(g, data, efeat)
        else:
            x = self.gnn_object(data, g)

        if not self.robot_node_indexes:
            indexes = []
            n_nodes = 0
            unbatched = dgl.unbatch(self.g)
            for g in unbatched:
                indexes.append(n_nodes+self.grid_nodes)
                n_nodes += g.number_of_nodes()
        else:
            indexes = self.robot_node_indexes

        logits = torch.squeeze(x, 1).to(device=data.device)
        output = logits[indexes].to(device=data.device)

        # logits = x

        # base_index = 0
        # batch_number = 0
        # unbatched = dgl.unbatch(self.g)
        # output = torch.Tensor(size=(len(unbatched), 2))
        # for g in unbatched:
        #     num_nodes = g.number_of_nodes()
        #     output[batch_number, :] = logits[base_index, :]  # Output is just the room's node
        #     # output[batch_number, :] = logits[base_index:base_index+num_nodes, :].mean(dim=0) # Output is the average of all nodes
        #     base_index += num_nodes
        #     batch_number += 1
        return output