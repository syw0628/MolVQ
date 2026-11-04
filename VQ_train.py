import os
import yaml
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score

from model.VQ import VQVAE
from utils.warmup_scheduler import NoamLR
from utils.splitter import random_split
from utils.prepare_dataset import MolGraphDataset, MolAugGraphDataset, custom_collate_fn


def fix_the_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    

LOSS_KEYS = ('Total_loss', 'Total_rec_loss', 'Total_vq_loss',
             'A_rec_loss', 'M_rec_loss', 'A_vq_loss', 'M_vq_loss')


def update_loss(loss_sum, node_count, loss_list, batch):
    """Accumulate detached losses weighted by atom or motif count."""
    vq_losses, rec_losses = loss_list
    n_atom, n_motif = batch[0].num_nodes(), batch[1].num_nodes()
    for key, value, count in zip(LOSS_KEYS[3:], (*rec_losses, *vq_losses),
                                 (n_atom, n_motif, n_atom, n_motif)):
        loss_sum[key] += value.detach().reshape(()).to(torch.float64) * count
    node_count['atom'] += n_atom
    node_count['motif'] += n_motif


def compute_epoch_loss(loss_sum, node_count):
    """Return node-weighted means and their reconstruction/VQ totals."""
    if not all(node_count.values()):
        raise ValueError('Cannot compute epoch loss without atom and motif nodes.')
    counts = (node_count['atom'], node_count['motif']) * 2
    means = torch.stack([loss_sum[key] / count
                         for key, count in zip(LOSS_KEYS[3:], counts)]).cpu().tolist()
    a_rec, m_rec, a_vq, m_vq = means
    rec, vq = a_rec + m_rec, a_vq + m_vq
    return dict(zip(LOSS_KEYS, (rec + vq, rec, vq, *means)))


def SSL_eval(test_loader, model, device):
    with torch.no_grad():
        model.eval()
        loss_sum = dict.fromkeys(LOSS_KEYS[3:], 0.0)
        node_count = {'atom': 0, 'motif': 0}
        atom_preds, motif_preds = [], []
        atom_y, motif_y = [], []
        valid_indices = []

        with tqdm(test_loader, desc="Validation") as epoch_iter:
            for step, batch in enumerate(epoch_iter):
                batch = (batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device))
                _, valid_loss_list, valid_idx_list, valid_preds = model(batch)
                update_loss(loss_sum, node_count, valid_loss_list, batch)
    
                atom_preds.append(valid_preds[0].detach().cpu())
                motif_preds.append(valid_preds[1].detach().cpu())
                atom_y.append(batch[0].ndata['atom_vocab_idx'].detach().cpu())
                motif_y.append(batch[1].ndata['f'][:, -166:].detach().cpu())
                valid_indices.extend(valid_idx_list)
        
        atom_probs, motif_probs = F.softmax(torch.cat(atom_preds, dim=0), dim=1), F.sigmoid(torch.cat(motif_preds, dim=0)) 
        atom_preds, motif_preds = atom_probs.argmax(dim=1).numpy(), motif_probs.round().numpy()
        atom_y, motif_y = torch.cat(atom_y, dim=0).numpy(), torch.cat(motif_y, dim=0).numpy()

        atom_macro_f1, atom_micro_f1 = f1_score(atom_y, atom_preds, average='macro'), f1_score(atom_y, atom_preds, average='micro')
        motif_macro_f1, motif_micro_f1 = f1_score(motif_y, motif_preds, average='macro'), f1_score(motif_y, motif_preds, average='micro')
        
        print("[Atom] Macro F1: {}, Micro F1: {}".format(round(atom_macro_f1, 4), round(atom_micro_f1, 4)))
        print("[Motif] Macro F1: {}, Micro F1: {}".format(round(motif_macro_f1, 4), round(motif_micro_f1, 4))) 
                                                                            
        return compute_epoch_loss(loss_sum, node_count), valid_indices


def main(args):    
    # Seed fixation 
    fix_the_seed(args.seed)
    
    # Device setting 
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    print(device)
    
    # Data preparation
    if args.load_data_path:                
        dataset = MolAugGraphDataset(data_path=args.dataset_path, frag_vocab_path=args.vocab_path, atom_vocab_path=args.atom_vocab_path,
                                smiles_col=args.smiles_col, standardize=False, cpus=args.num_workers, aug_data_path=args.load_data_path)
    else: 
        dataset = MolGraphDataset(data_path=args.dataset_path, frag_vocab_path=args.vocab_path, atom_vocab_path=atom_vocab_path, 
                                        smiles_col=args.smiles_col, standardize=False, cpus=args.num_workers)
            
    train_dataset, valid_dataset = random_split(dataset, frac_train=1-args.valid_ratio,
                                                frac_test=args.valid_ratio, frac_valid=None, seed=args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=custom_collate_fn, shuffle=False)    

    # VQ-VAE model preparation
    with open(args.config, 'r') as config_file:
         config = yaml.safe_load(config_file)
    data_args = SimpleNamespace(**config['data_args'])
    model = VQVAE(args, data_args).to(device)
    
    # Generate the folder to save the pretrained model 
    os.makedirs(args.save_model_path , exist_ok=True)
    
    #Set up optimizers & scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    if args.use_scheduler:
        print("--------- Use scheduler -----------")
        scheduler = NoamLR(optimizer = optimizer, warmup_epochs = [args.warmup], total_epochs = [args.epochs],
                           steps_per_epoch = len(train_loader), init_lr = [args.init_lr], max_lr = [args.lr], final_lr = [args.final_lr])
    else:
        scheduler = None

    # Training
    history, valid_history = {key: [] for key in LOSS_KEYS}, {key: [] for key in LOSS_KEYS}

    for epoch in range(1, args.epochs+1):
        print("====epoch {}====".format(str(epoch)))
             
        model.train()
        loss_sum = dict.fromkeys(LOSS_KEYS[3:], 0.0)
        node_count = {'atom': 0, 'motif': 0} 
        epoch_iter = tqdm(train_loader, desc="Train")
        ae_idx_list, me_idx_list = [], []
        
        for step, batch in enumerate(epoch_iter):
            atom_g, motif_g, atom_map, motif_map = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
            batch = (atom_g, motif_g, atom_map, motif_map)
            
            losses, loss_list, vq_idx_list, _ = model(batch)
            vq_loss, rec_loss = losses 
            _, rec_loss_list = loss_list
            loss = rec_loss + vq_loss 
           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()

            update_loss(loss_sum, node_count, loss_list, batch)
            epoch_iter.set_postfix({"train_loss": f"{loss.item():.4f}", "atom_rec": f"{rec_loss_list[0].item():.4f}", "motif_rec": f"{rec_loss_list[1].item():.4f}"})
            
            ae_idx_list.append(vq_idx_list[0].detach().cpu())
            me_idx_list.append(vq_idx_list[0].detach().cpu())
        
        # Validation 
        epoch_valid_loss_dict, valid_vq_idx_list = SSL_eval(valid_loader, model, device)
        epoch_loss_dict = compute_epoch_loss(loss_sum, node_count)

        ae_idx_list = torch.cat(ae_idx_list)
        me_idx_list = torch.cat(me_idx_list)

        # Report the loss value for training and validation
        print("---Train loss---")
        print(epoch_loss_dict)
        
        print("---Valid loss---")
        print(epoch_valid_loss_dict)
        
        print("---Train usage---")
        print(len(torch.unique(ae_idx_list)) / args.num_codebook_emb[0],
              len(torch.unique(me_idx_list)) / args.num_codebook_emb[1])
        
        print("---Valid usage---")
        print(len(torch.unique(valid_vq_idx_list[0].detach().cpu())) / args.num_codebook_emb[0],
              len(torch.unique(valid_vq_idx_list[1].detach().cpu())) / args.num_codebook_emb[1])
        
        for key in LOSS_KEYS:
            history[key].append(epoch_loss_dict[key])
            valid_history[key].append(epoch_valid_loss_dict[key])
        
        torch.save(model.state_dict(), '{}/VQ_ep{}.pth'.format(args.save_model_path, epoch))
        
        # Save the output of pretrained model
        columns = dict(zip(LOSS_KEYS, ('total_loss', 'rec_loss', 'total_VQ_loss',
                                      'A_rec_loss', 'M_rec_loss', 'A_vq_loss', 'M_vq_loss')))
        df = pd.DataFrame({f'{split}_{columns[key]}': values
                           for split, records in (('train', history), ('valid', valid_history))
                           for key, values in records.items()})
        df.to_csv('{}/VQ_log.csv'.format(args.save_model_path), index=False)
        
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training for graph neural networks & VQ-VAE')
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--num_workers', type=int, default = 16, help='number of workers for dataset loading')
    parser.add_argument('--smiles_col', type=str, default='st_smiles', help='name of column which contains smiles')
    parser.add_argument('--load_data_path', type=str, default='./constructed_data/guacamol_aug_dataset', 
                        help='dataset path to load')
    parser.add_argument('--dataset_path', type=str, default='./Dataset/pretraining/guacamol_smiles.csv',
                        help='dataset path to load')
    parser.add_argument('--atom_vocab_path', type=str, default='./vocab/guacamol_atom_vocab.pkl',
                        help='path of already extracted atom vocabulary')
    parser.add_argument('--vocab_path', type=str, default='./vocab/guacamol_relmole_vocab.txt',
                        help='path of already extracted fragment vocabulary')
    parser.add_argument('--config', type=str, default='./config.yaml')
    parser.add_argument('--save_model_path', type=str, default = './pretrained_model/VQ', help='filename to output the model') 
    
    parser.add_argument('--valid_ratio', type=float, default=0.1, help='ratio of validation set (default: 0.1)')
    parser.add_argument('--batch_size', type=int, default=1024, help='input batch size for training (default: 1024)')
    parser.add_argument('--epochs', type=int, default=80, help='number of epochs to train (default: 80)')
    parser.add_argument('--init_lr', type=float, default=1e-6, help='learning rate (default: 1e-6)')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate (default: 5e-4)')
    parser.add_argument('--final_lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--warmup', type=float, default=5)
    parser.add_argument('--decay', type=float, default=1e-6, help='weight decay (default: 1e-6)')
    parser.add_argument("--use_scheduler", type=bool, default=True)

    parser.add_argument('--num_layer', type=int, nargs=2, default=[3, 3], help='number of GNN message passing layers (default: 3).')
    parser.add_argument('--vq_emb_dim', type=int, default=300, help='embedding dimensions (default: 300)')  # VQ
    parser.add_argument('--emb_dim', type=int, default=300, help='embedding dimensions (default: 300)') # GNN 
    parser.add_argument('--num_codebook_emb', type=int, nargs=2, default=[2000, 2000], help='number of codebook embedding in VQ module')
    parser.add_argument('--vq_pool', type=str, default="mean")
    parser.add_argument('--commitment_cost', type = float, default = 0.5, help = 'commitment_cost')
    parser.add_argument('--codebook_decay', type=float, default=0.9, help='EMA decay for VQ codebooks')
    parser.add_argument('--threshold_dead_code', type=float, default=1.0, help='EMA threshold for replacing dead codes')   
    
    args = parser.parse_args()

    main(args)