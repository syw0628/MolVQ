import warnings
warnings.filterwarnings('ignore')

import torch
from torch import nn
from torch_scatter import scatter_mean, scatter_add, scatter_max

from .Model import MLPDecoder, GIN
from vector_quantize_pytorch import VectorQuantize


# Loss define 
atom_criterion = nn.CrossEntropyLoss()
motif_criterion = nn.BCEWithLogitsLoss()

class VQVAE(nn.Module):
    def __init__(self, args, data_args):
        super(VQVAE, self).__init__()
        self.args = args
        self.data_args = data_args
        
        # Encoder
        self.atom_encoder = GIN(num_layer=args.num_layer[0], node_dim=data_args.atom_fdim, edge_dim=data_args.bond_fdim, 
                                emb_dim=args.emb_dim, out_dim=args.emb_dim)
        self.motif_encoder = GIN(num_layer=args.num_layer[1], node_dim=data_args.motif_fdim, edge_dim=data_args.bond_fdim, 
                                emb_dim=args.emb_dim, out_dim=args.emb_dim)        
        # Quantizer
        if args.vq_pool == 'max':
            self.pool_func = scatter_max
        elif args.vq_pool == 'sum':
            self.pool_func = scatter_add
        elif args.vq_pool == 'mean':
            self.pool_func = scatter_mean
        else: 
            raise ValueError("Unavailable VQ pooling type")
                  
        self.atom_vq = VectorQuantize(dim=args.emb_dim, codebook_dim=args.vq_emb_dim, codebook_size=args.num_codebook_emb[0], rotation_trick=True,
                                      commitment_weight=args.commitment_cost, decay=args.codebook_decay, threshold_ema_dead_code=args.threshold_dead_code)

        self.motif_vq = VectorQuantize(dim=args.emb_dim, codebook_dim=args.vq_emb_dim, codebook_size=args.num_codebook_emb[1], rotation_trick=True, 
                                       commitment_weight=args.commitment_cost, decay=args.codebook_decay, threshold_ema_dead_code=args.threshold_dead_code)
        # Decoder
        self.atom_type_decoder = MLPDecoder(args.emb_dim, data_args.num_atom_vocab)
        self.motif_type_decoder = MLPDecoder(args.emb_dim*2, data_args.num_motif_maccs)

    def forward(self, batch):
        # Encoding
        atom_g, motif_g, atom_map, motif_map = batch
        atom_map, motif_map = atom_map.long(), motif_map.long()
        atom_rep = self.atom_encoder(atom_g)
        motif_rep = self.motif_encoder(motif_g)

        # Quantize
        ae, ae_idx, ae_q_loss = self.atom_vq(atom_rep) # atom-level vq
        me, me_idx, me_q_loss = self.motif_vq(motif_rep) # motif-level vq
                
        # Decoding
        modified_ae = ae[atom_map]
        ae_pooled = self.pool_func(modified_ae, motif_map, dim=0)
        if self.pool_func == 'max':
            ae_pooled = ae_pooled[0]

        pred_atom = self.atom_type_decoder(ae) # atom_ssl task
        pred_motif = self.motif_type_decoder(torch.cat([me, ae_pooled.detach()], dim=1)) # motif_ssl task

        # Loss calculation
        atom_loss = atom_criterion(pred_atom, atom_g.ndata['atom_vocab_idx'].long())
        motif_loss = motif_criterion(pred_motif, motif_g.ndata['f'][:, -self.data_args.num_motif_maccs:])

        vq_loss_list, rec_loss_list = [ae_q_loss, me_q_loss], [atom_loss, motif_loss]

        rec_loss = sum(rec_loss_list)
        losses, loss_lists, preds = (sum(vq_loss_list), rec_loss), (vq_loss_list, rec_loss_list), (pred_atom, pred_motif)
        
        return losses, loss_lists, (ae_idx, me_idx), preds 
        
    def from_pretrained(self, model_file):
        self.load_state_dict(torch.load(model_file, weights_only=True, map_location=torch.device('cpu')))  
        
    def inference(self, batch, node_type='atom'):
        if node_type == 'atom':
            node_rep = self.atom_encoder(batch)
            quantizer = self.atom_vq
        elif node_type == 'motif':
            node_rep = self.motif_encoder(batch)
            quantizer = self.motif_vq
        else:
            raise "Unavailable node type is given"
        vq_e, vq_idx, _ = quantizer(node_rep)
        return vq_e, vq_idx
