import copy
import math
from functools import partial
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn

from .VQ import VQVAE


def reverse_edge(tensor):
    n = tensor.size(0)
    assert n%2 ==0
    delta = torch.ones(n).type(torch.long)
    delta[torch.arange(1,n,2)] = -1
    return tensor[delta+torch.tensor(range(n))]

def del_reverse_message(edge,field):
    """for g.apply_edges"""
    return {'m': edge.src[field]-edge.data['rev_h']}

def add_attn(node,field,attn):
        feat = node.data[field].unsqueeze(1)
        return {field: (attn(feat,node.mailbox['m'],node.mailbox['m'])+feat).squeeze(1)}

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

# GRU readout from Original PharmHGT model code 
class Node_GRU(nn.Module):
    """GRU for graph readout. Implemented with dgl graph"""
    def __init__(self, hid_dim, bidirectional=True, num_heads=6,dropout=0):#, args:Namespace=None):
        super(Node_GRU,self).__init__()
        self.hid_dim = hid_dim
        if bidirectional:
            self.direction = 2
        else:
            self.direction = 1
        self.att_mix = MultiHeadedAttention(num_heads, hid_dim, dropout=dropout)
        self.gru  = nn.GRU(hid_dim, hid_dim, batch_first=True, 
                           bidirectional=bidirectional)
            
    def split_batch(self, bg, ntype, field, device):
        hidden = bg.nodes[ntype].data[field]
        node_size = bg.batch_num_nodes(ntype)
        start_index = torch.cat([torch.tensor([0],device=device),torch.cumsum(node_size,0)[:-1]])
        max_num_node = max(node_size)
        # padding
        hidden_lst = []
        for i  in range(bg.batch_size):
            start, size = start_index[i],node_size[i]
            assert size != 0, size
            cur_hidden = hidden.narrow(0, start, size)
            cur_hidden = torch.nn.ZeroPad2d((0,0,0,max_num_node-cur_hidden.shape[0]))(cur_hidden)
            hidden_lst.append(cur_hidden.unsqueeze(0))

        hidden_lst = torch.cat(hidden_lst, 0)

        return hidden_lst
        
    def forward(self,bg,suffix='h'):
        """
        bg: dgl.Graph (batch)
        hidden states of nodes are supposed to be in field 'h'.
        """
        self.suffix = suffix
        device = bg.device
        
        p_pharmj = self.split_batch(bg,'p',f'f_{suffix}',device)
        a_pharmj = self.split_batch(bg,'a',f'f_{suffix}',device)
        
        mask = (a_pharmj!=0).type(torch.float32).matmul((p_pharmj.transpose(-1,-2)!=0).type(torch.float32))==0
        h = self.att_mix(a_pharmj, p_pharmj, p_pharmj, mask) + a_pharmj

        hidden = h.max(1)[0].unsqueeze(0).repeat(self.direction,1,1)
        h, hidden = self.gru(h, hidden)
        
        # unpadding and reduce (mean) h: batch * L * hid_dim
        graph_embed = []
        node_size = bg.batch_num_nodes('a')
        start_index = torch.cat([torch.tensor([0],device=device),torch.cumsum(node_size,0)[:-1]])
        for i  in range(bg.batch_size):
            start, size = start_index[i],node_size[i]
            graph_embed.append(h[i, :size].view(-1, self.direction*self.hid_dim).mean(0).unsqueeze(0))
        graph_embed = torch.cat(graph_embed, 0)
        return graph_embed
               
# Multi-view message passing
class MVMP(nn.Module):
    def __init__(self,msg_func=add_attn, hid_dim=300, depth=3, view='apj', suffix='h', act=nn.ReLU(), num_heads=6, dropout=0):
        """
        MultiViewMassagePassing
        view: a, ap, apj
        suffix: filed to save the nodes' hidden state in dgl.graph. 
                e.g. bg.nodes[ntype].data['f'+'_junc'(in ajp view)+suffix]
        """
        super(MVMP,self).__init__()
        self.view = view
        self.depth = depth
        self.suffix = suffix
        self.msg_func = msg_func
        self.act = act
        self.homo_etypes = [('a','b','a')]
        self.hetero_etypes = []
        self.node_types = ['a','p']

        if 'p' in view:
            self.homo_etypes.append(('p','r','p'))
        if 'j' in view:
            self.node_types.append('junc')
            self.hetero_etypes=[('a','j','p'),('p','j','a')] # don't have feature

        self.attn = nn.ModuleDict()
        for etype in self.homo_etypes + self.hetero_etypes:
            self.attn[''.join(etype)] = MultiHeadedAttention(num_heads, hid_dim, dropout=dropout)

        self.mp_list = nn.ModuleDict()
        for edge_type in self.homo_etypes:
            self.mp_list[''.join(edge_type)] = nn.ModuleList([nn.Linear(hid_dim, hid_dim) for i in range(depth - 1)])

        self.node_last_layer = nn.ModuleDict()
        for ntype in self.node_types:
            self.node_last_layer[ntype] = nn.Linear(3*hid_dim, hid_dim)

    def update_edge(self,edge,layer):
        return {'h':self.act(edge.data['f']+layer(edge.data['m']))}
    
    def update_node(self,node,field,layer):
        return {field:layer(torch.cat([node.mailbox['mail'].sum(dim=1),
                                       node.data[field],
                                       node.data['f']],1))}
    def init_node(self,node):
        return {f'f_{self.suffix}':node.data['f'].clone()}

    def init_edge(self,edge):
        return {'h':edge.data['f'].clone()}

    def forward(self,bg):
        suffix = self.suffix
        for ntype in self.node_types:
            if ntype != 'junc':
                bg.apply_nodes(self.init_node,ntype=ntype)
        for etype in self.homo_etypes:
            bg.apply_edges(self.init_edge,etype=etype)

        if 'j' in self.view:
            bg.nodes['a'].data[f'f_junc_{suffix}'] = bg.nodes['a'].data['f_junc'].clone()
            bg.nodes['p'].data[f'f_junc_{suffix}'] = bg.nodes['p'].data['f_junc'].clone()

        update_funcs = {e:(fn.copy_e('h','m'),partial(self.msg_func, attn=self.attn[''.join(e)], field=f'f_{suffix}')) for e in self.homo_etypes }
        update_funcs.update({e:(fn.copy_u(f'f_junc_{suffix}','m'),partial(self.msg_func, attn=self.attn[''.join(e)], field=f'f_junc_{suffix}')) for e in self.hetero_etypes})
        
        # Message passing
        for i in range(self.depth-1):
            bg.multi_update_all(update_funcs,cross_reducer='sum')
            for edge_type in self.homo_etypes:
                bg.edges[edge_type].data['rev_h']=reverse_edge(bg.edges[edge_type].data['h'])
                bg.apply_edges(partial(del_reverse_message,field=f'f_{suffix}'),etype=edge_type)
                bg.apply_edges(partial(self.update_edge,layer=self.mp_list[''.join(edge_type)][i]), etype=edge_type)

        # last update of node feature
        update_funcs = {e:(fn.copy_e('h','mail'),partial(self.update_node,field=f'f_{suffix}',layer=self.node_last_layer[e[0]])) for e in self.homo_etypes}
        bg.multi_update_all(update_funcs,cross_reducer='sum')

        # last update of junc feature
        bg.multi_update_all({e:(fn.copy_u(f'f_junc_{suffix}','mail'),
                                 partial(self.update_node,field=f'f_junc_{suffix}',layer=self.node_last_layer['junc'])) for e in self.hetero_etypes},
                                 cross_reducer='sum')

# Modified PharmHGT
class PharmHGT_modified(nn.Module):
    def __init__(self, args, vq_args, data_args):
        super(PharmHGT_modified, self).__init__()
        
        self.act =nn.ReLU()

        # Initialization
        # atom view
        self.w_atom = nn.Linear(vq_args.emb_dim, args.hidden_dim)
        self.w_bond = nn.Linear(data_args.bond_fdim, args.hidden_dim)
        # pharm view
        self.w_pharm = nn.Linear(vq_args.emb_dim, args.hidden_dim)
        self.w_reac = nn.Linear(data_args.bond_fdim, args.hidden_dim)
        # junction view
        self.w_junc = nn.Linear(vq_args.emb_dim, args.hidden_dim)

        ## Define the VQVAE and MVMP
        self.vq = VQVAE(vq_args, data_args)
        self.mp = MVMP(msg_func=add_attn, hid_dim=args.hidden_dim, depth=args.depth, view='apj', suffix='h', act=self.act, num_heads=args.num_att_head, dropout=args.mvmp_dropout)

        ## Readout
        self.readout_method = args.readout
        if self.readout_method == 'gap':
            self.a_gate_nn = nn.Linear(args.hidden_dim, 1) 
            self.p_gate_nn = nn.Linear(args.hidden_dim, 1)
        else: 
            self.readout = Node_GRU(args.hidden_dim, num_heads=args.num_att_head, dropout=args.mvmp_dropout)

        # MLP predictor    
        self.W_o = nn.Linear(args.hidden_dim*2, args.hidden_dim)
        self.dropout = nn.Dropout(p=args.dropout)
        self.predictor = nn.Sequential(nn.Linear(args.hidden_dim, args.out_dim))                
                
    def init_feature(self, bg):
        bg.nodes['a'].data['f'] = self.act(self.w_atom(bg.nodes['a'].data['f']))
        bg.edges[('a','b','a')].data['f'] = self.act(self.w_bond(bg.edges[('a','b','a')].data['f']))
        bg.nodes['p'].data['f'] = self.act(self.w_pharm(bg.nodes['p'].data['f']))
        bg.edges[('p','r','p')].data['f'] = self.act(self.w_reac(bg.edges[('p','r','p')].data['f']))
        bg.nodes['a'].data['f_junc'] = self.act(self.w_junc(bg.nodes['a'].data['f_junc']))
        bg.nodes['p'].data['f_junc'] = self.act(self.w_junc(bg.nodes['p'].data['f_junc']))
        
    def from_vq_pretrained(self, model_file):
        self.vq.from_pretrained(model_file)
        self.vq.eval()
        for param in self.vq.parameters():
            param.requires_grad = False
        
    def from_pretrained(self, model_file, filtered=True):
        state_dict = torch.load(model_file, weights_only=True, map_location=torch.device('cpu'))
        if filtered:
            filtered_state_dict = {
                k: v for k, v in state_dict.items() if not k.startswith('predictor')
            }
            self.load_state_dict(filtered_state_dict, strict=False)
        else: 
            self.load_state_dict(state_dict, strict=False)
            
    def forward(self, g):
        """
        Args:
            bg: a batch of graphs
        """
        aug_g, atom_g, motif_g = g

        with torch.no_grad():
            atom_vq, atom_vq_idx = self.vq.inference(atom_g, node_type='atom')
            frag_vq, frag_vq_idx = self.vq.inference(motif_g, node_type='motif')
            
        aug_g.nodes['a'].data['f'] = atom_vq
        aug_g.nodes['p'].data['f'] = frag_vq

        aug_g.nodes['a'].data['f_junc'] = aug_g.nodes['a'].data['f'].clone()
        aug_g.nodes['p'].data['f_junc'] = aug_g.nodes['p'].data['f'].clone()
        
        self.init_feature(aug_g)
        
        self.mp(aug_g)
        
        if self.readout_method == 'gap':
            a_score = self.a_gate_nn(aug_g.nodes['a'].data['f_h'])
            aug_g.nodes['a'].data['a_score'] = a_score
            aug_g.nodes['a'].data['attn'] = dgl.softmax_nodes(aug_g, 'a_score', ntype='a')
            
            p_score = self.p_gate_nn(aug_g.nodes['p'].data['f_h'])
            aug_g.nodes['p'].data['p_score'] = p_score
            aug_g.nodes['p'].data['attn'] = dgl.softmax_nodes(aug_g, 'p_score', ntype='p')
            
            embed_atom = dgl.readout_nodes(aug_g, feat='f_h', weight='attn', ntype='a')
            embed_pharm = dgl.readout_nodes(aug_g, feat='f_h', weight='attn', ntype='p')
            ro_embed = torch.cat([embed_atom, embed_pharm], dim=1)
        else:
            ro_embed = self.readout(aug_g, 'h')
        raw_embed = self.W_o(ro_embed)
        embed = self.dropout(self.act(raw_embed))
        pred = self.predictor(embed)
        return (ro_embed, raw_embed, embed), pred
