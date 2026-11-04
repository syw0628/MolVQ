import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import dgl.nn.pytorch as dglnn
from dgl.nn import GINConv, GraphConv 
from dgl.nn.pytorch.glob import AvgPooling, SumPooling, MaxPooling

# GIN
class GIN_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim))
        self.linears.append(nn.Linear(hidden_dim, output_dim))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))
    
    def forward(self, x):
        h = F.relu(self.batch_norm(self.linears[0](x)))
        return self.linears[1](h)

class GINConvWithEdge(nn.Module):
    def __init__(self, mlp, learn_eps=False):
        super().__init__()
        self.mlp = mlp
        if learn_eps:
            self.eps = nn.Parameter(torch.Tensor([0]))
        else:
            self.register_buffer('eps', torch.FloatTensor([0]))
    
    def forward(self, g, node_feat, edge_feat):
        g = g.local_var()
        g.ndata['f'] = node_feat
        g.edata['f'] = edge_feat

        g.update_all(fn.u_add_e('f', 'f', 'm'), fn.sum('m', 'f'))
        h = self.mlp(g.ndata.pop('f'))
        return h     

class GIN(nn.Module):
    def __init__(self, num_layer, node_dim, edge_dim, emb_dim, out_dim, dropout=0., JK='last'):
        super().__init__() 
        self.num_layers = num_layer
        self.JK = JK
        
        self.gnn_layers = nn.ModuleList()
        
        # Node and edge embedding layers
        self.node_emb_layer = nn.Linear(node_dim, emb_dim)
        self.edge_emb_layer = nn.Linear(edge_dim, emb_dim)

        # GIN layers
        for layer in range(self.num_layers):
            mlp = GIN_MLP(emb_dim, emb_dim, emb_dim)
            self.gnn_layers.append(
                GINConvWithEdge(mlp)
            )
      
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU() 
        
        self.reset_parameters()

    # Initialize the node and edge embedding layers 
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.node_emb_layer.weight)
        nn.init.xavier_uniform_(self.edge_emb_layer.weight)
    
    # Prepare self-loop edge features
    def add_self_loops(self, g, edge_attr):
        g = dgl.add_self_loop(g)
        self_loop_attr = torch.tensor([0., 0., 0., 0., 1., 1., 0., 0., 0., 0.], device=edge_attr.device)   
        self_loop_attrs = self_loop_attr.unsqueeze(0).repeat(g.num_nodes(), 1)
        edge_attr = torch.cat([edge_attr, self_loop_attrs], dim=0)
        return g, edge_attr
    
    def forward(self, g):
        # Add self-loops
        g, edge_attr = self.add_self_loops(g, g.edata['f'])
        x = g.ndata['f']
        
        h = self.node_emb_layer(x)
        edge_attr = self.edge_emb_layer(edge_attr)
        
        h_list = [h]
        for layer in range(self.num_layers):
            h = self.gnn_layers[layer](g, h, edge_attr)
            if layer == self.num_layers - 1:
                h = self.dropout(h)
            else: 
                h = self.act(h)
                h = self.dropout(h)
            h_list.append(h)

        if self.JK == 'concat':
            node_rep = torch.cat(h_list, dim=1)
        elif self.JK == 'last':
            node_rep = h_list[-1]
        elif self.JK == 'max':
            h_list = [h.unsqueeze(0) for h in h_list]
            node_rep = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == 'sum':
            h_list = [h.unsqueeze(0) for h in h_list]
            node_rep = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        else:
            return ValueError("Expect self.JK to be 'concat', 'last', "
                              "'max' or 'sum', got {}".format(self.JK))
        return node_rep


# MLP
class MLPPredictor(nn.Module):
    def __init__(self, emb_dim, out_dim):
        super(MLPPredictor, self).__init__()
        self.fc = nn.Linear(emb_dim, out_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, emb):
        emb = self.log_softmax(self.fc(emb))  
        return out 
        
# 2-layer MLP         
class MLPDecoder(nn.Module):
    def __init__(self, emb_dim, out_dim):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(emb_dim, emb_dim, bias=True)
        self.fc2 = nn.Linear(emb_dim, out_dim, bias=True)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))   
        out = self.fc2(x)
        return out 

# GCN baseline 
class GCN_baseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes, pool='mean'):
        super(GCN_baseline, self).__init__()
        self.conv1 = dglnn.GraphConv(input_dim, hidden_dim,allow_zero_in_degree=True)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)
        self.conv3 = GraphConv(hidden_dim, n_classes, allow_zero_in_degree=True)
        self.mlp = nn.Linear(hidden_dim, n_classes)
        
        if pool == 'mean':
            self.pool = AvgPooling()
        elif pool == 'sum':
            self.pool = SumPooling()
        elif pool == 'max':
            self.pool = MaxPooling()
        else:
            self.pool = None
            
    def forward(self, g):
        h = g.ndata['f']
        # Apply graph convolution and activation.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            if self.pool != None: 
                hg = self.pool(g, h)
                return self.mlp(hg)
            return hg

# GIN baseline 
class MLP(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_feats, out_feats)
        self.linear2 = nn.Linear(out_feats, out_feats)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)

class GIN_baseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes, num_layers=3, pool='mean'):
        super(GIN_baseline, self).__init__()
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            mlp = MLP(in_dim, hidden_dim)
            conv = GINConv(mlp, 'sum')
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        if pool == 'mean':
            self.pool = AvgPooling()
        elif pool == 'sum':
            self.pool = SumPooling()
        elif pool == 'max':
            self.pool = MaxPooling()
        else:
            self.pool = None
            
        self.mlp = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        h = g.ndata['f']
        for conv, bn in zip(self.convs, self.batch_norms):
            h = conv(g, h)
            h = bn(h)
            h = F.relu(h)
        with g.local_scope():
            g.ndata['h'] = h
            hg = self.pool(g, h)
        return self.mlp(hg)  
