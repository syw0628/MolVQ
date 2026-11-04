"""
Fine-tuning script for toxicity prediction
"""

import os
import yaml
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from types import SimpleNamespace

import torch
import torch.nn as nn
from dgl.dataloading import GraphDataLoader

from utils.splitter import scaffold_split
from utils.warmup_scheduler import CosineAnnealingWarmupRestarts
from utils.prepare_dataset import MolAugGraphDataset

from model.HeteroGNN import PharmHGT_modified
from sklearn.metrics import (
    precision_score, confusion_matrix, accuracy_score,
    balanced_accuracy_score, f1_score, average_precision_score, roc_auc_score
)


def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def fix_the_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_predictor(model, args, device, hidden_dim=300, dropout=0.0):
    if args.n_pred_layer == 3:
        model.predictor = nn.Sequential(
            nn.Linear(hidden_dim, args.pred_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(args.pred_hidden, args.pred_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(args.pred_hidden // 2, args.n_tasks)
            )
    elif args.n_pred_layer == 2:
        model.predictor = nn.Sequential(
            nn.Linear(hidden_dim, args.pred_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(args.pred_hidden, args.n_tasks)
            )
    elif args.n_pred_layer == 1:
        model.predictor = nn.Sequential(
            nn.Linear(hidden_dim, args.n_tasks)
            )
    else:
        raise ValueError(f"Unavailable number of predictor layers: {args.n_pred_layer}")
    return model.to(device)
    
def build_optimizer(parameters, lr, use_LLRD=False, weight_decay=0):
    if use_LLRD: 
        optimizer = torch.optim.Adam(parameters, weight_decay=weight_decay)
    else:
        trainable_params = [p for p in parameters if p.requires_grad]
        optimizer = torch.optim.Adam([{'params': trainable_params, 'lr': lr}], weight_decay=weight_decay)
    return optimizer

def build_scheduler(optimizer, scheduler_type: str, args):
    """
    Learning rate scheduler 생성
    Returns:
        scheduler or None
    """
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=args.first_cycle_steps,
            cycle_mult=args.cycle_mult,
            max_lr=args.lr,
            min_lr=args.min_lr,
            warmup_steps=args.warmup,
            gamma=args.gamma
        )
    else:
        scheduler = None
    
    return scheduler

def get_llrd_lr(model, init_lr, decay_coef):
    parameters, lr = [], 0
    
    for name, param in model.named_parameters():
        if 'vq.' in name or not param.requires_grad:
            continue       
        
        # 1. Predictor
        if ('predictor' in name) or ('W_o' in name):
            lr = init_lr
            
        # 2. Readout   
        elif ('readout' in name) or ('gate' in name): 
            lr = init_lr * (decay_coef ** 1)

        # 3. Update
        elif 'node_last_layer' in name:
            lr = init_lr * (decay_coef ** 2)

        # 5. 2nd MP
        elif ('mp_list' in name and '.1.' in name) or ('mp.attn' in name):
            lr = init_lr * (decay_coef ** 3)
            
        # 6. 1st MP 
        elif ('mp_list' in name and '.0.' in name):
            lr = init_lr * (decay_coef ** 4)
            
        # 7. Init 
        elif any(key in name for key in ['w_atom', 'w_bond', 'w_pharm', 'w_reac']):
            lr = init_lr * (decay_coef ** 5)
            
        # 8. Others
        if lr == 0:
            lr = init_lr * (decay_coef ** 6)
            
        parameters += [{'params': [param], 'lr': lr}]
        
    return parameters


def setup_task_config(task: str, config):
    task_config = config['num_task']
    if task in task_config['single_task']:
        return task, 1
    elif task in task_config['multi_task']:
        return None, task_config['multi_task'][task]
    else:
        raise ValueError(f"Unknown task dataset: {task}")


def get_dataloader(config, args):
    if args.split_method == 'scaffold_split': 
        data_path = args.data_dir+f'{args.dataset}.csv'
        ft_dataset = MolAugGraphDataset(data_path=data_path,
                                        frag_vocab_path=config['paths']['vocab_path'], atom_vocab_path=config['paths']['atom_vocab_path'],
                                        smiles_col='st_smiles', task_col=args.task_col, cpus=8)
        smiles_list = pd.read_csv(data_path)['st_smiles']
        train_dataset, valid_dataset, test_dataset = scaffold_split(ft_dataset, smiles_list, frac_train=1-args.test_ratio*2, 
                                                                    frac_valid=args.test_ratio, frac_test=args.test_ratio)
    else:            
        train_dataset = MolAugGraphDataset(data_path=args.data_dir+f'{args.dataset}/{args.dataset}_train.csv',
                                        frag_vocab_path=config['paths']['vocab_path'], atom_vocab_path=config['paths']['atom_vocab_path'],
                                        smiles_col='st_smiles', task_col=args.task_col, cpus=8)
        valid_dataset = MolAugGraphDataset(data_path=args.data_dir+f'{args.dataset}/{args.dataset}_valid.csv',
                                        frag_vocab_path=config['paths']['vocab_path'], atom_vocab_path=config['paths']['atom_vocab_path'],
                                        smiles_col='st_smiles', task_col=args.task_col, cpus=8)
        test_dataset = MolAugGraphDataset(data_path=args.data_dir+f'{args.dataset}/{args.dataset}_test.csv',
                                        frag_vocab_path=config['paths']['vocab_path'], atom_vocab_path=config['paths']['atom_vocab_path'],
                                        smiles_col='st_smiles', task_col=args.task_col, cpus=8)
    train_dataset.process()
    valid_dataset.process()
    test_dataset.process()
    
    train_loader = GraphDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = GraphDataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = GraphDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader


def train(train_loader, valid_loader, test_loader, args, config, device, save_path=None):
    heteroGNN_args = SimpleNamespace(**config['heteroGNN_args'])
    data_args = SimpleNamespace(**config['data_args'])
    vq_args = SimpleNamespace(**config['vq_args'])
    
    # Model preparation 
    model = PharmHGT_modified(heteroGNN_args, vq_args, data_args).to(device)
    model.from_vq_pretrained(config['paths']['vq_model_path'])
    model.from_pretrained(config['paths']['model_path'], filtered=True)

    for param in model.vq.parameters():
        param.requires_grad = False
    
    model = build_predictor(model, args, device, heteroGNN_args.hidden_dim, heteroGNN_args.dropout)
    
    # LLRD, Scheduler setting 
    if args.LLRD:
        parameters = get_llrd_lr(model, args.lr, decay_coef=args.LLRD_decay_coef)
        optimizer = build_optimizer(parameters, args.lr, args.LLRD, args.weight_decay)
    else: 
        optimizer = build_optimizer(model.parameters(), args.lr, args.LLRD, args.weight_decay)
        
    scheduler = build_scheduler(optimizer, args.scheduler, args)
    
    criterion = nn.BCEWithLogitsLoss()
    
    train_history, valid_history, test_history = [], [], []

    best_valid_score = float('-inf')
     
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        epoch_iter = tqdm(train_loader, desc="Iteration")
        
        for step, batch in enumerate(epoch_iter):
            hetero_g, atom_g, motif_g, label = batch 
            batch = (hetero_g, atom_g, motif_g)
            batch = tuple(map(lambda x: x.to(device), batch))
            label = label.to(device).float()

            _, predictions = model(batch)

            mask = ~torch.isnan(label.cpu())
            loss = 0
            for i in range(args.n_tasks):
                task_loss = criterion(
                    predictions[:, i][mask[:, i]],
                    label[:, i][mask[:, i]]
                )
                loss += task_loss
            loss = loss / args.n_tasks
            train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_iter.set_description(f"Epoch: {epoch} tloss: {loss:.4f}")

        # Scheduler step: Cosine
        if args.scheduler == 'cosine':
            scheduler.step()       
        
        _, train_res = evaluate(model, train_loader, args, device)
        valid_loss, valid_res = evaluate(model, valid_loader, args, device)
        _, test_res = evaluate(model, test_loader, args, device)

        if epoch % 5 == 0:
            print(f"Score(train, valid, test): {(train_res[-1]+train_res[-2])/2:.3f} "
                  f"{(valid_res[-1]+valid_res[-2])/2:.3f} {(test_res[-1]+test_res[-2])/2:.3f}")
        
        train_history.append([*train_res, train_loss / len(train_loader)])
        valid_history.append(valid_res)
        test_history.append(test_res)

        valid_score = (valid_res[-1] + valid_res[-2]) / 2
        if best_valid_score < valid_score:
            best_valid_score = valid_score
            best_epoch = epoch

            if args.save_model:
                torch.save(model.state_dict(), '{}/state_dict_{}_{}.pth'.format(save_path, best_epoch, args.seed))
        
    
    return train_history, valid_history, test_history


def evaluate(model, loader, args, device):
    """
    **Evaluation func**
    Returns:
        tuple: (test_loss, metrics_list)
            metrics_list = [precision, sensitivity, specificity, accuracy, 
                          balanced_accuracy, f1, aupr, auc]
    """
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        model.eval()
        
        y_true, y_scores = [], []
        test_loss = 0
        
        for step, batch in enumerate(loader):
            hetero_g, atom_g, motif_g, label = batch 
            batch = (hetero_g, atom_g, motif_g)
            batch = tuple(map(lambda x: x.to(device), batch))
            label = label.to(device).float()

            _, predictions = model(batch)
                    
            mask = ~torch.isnan(label.cpu())
            loss = 0
            for i in range(args.n_tasks):
                task_loss = criterion(
                    predictions[:, i][mask[:, i]],
                    label[:, i][mask[:, i]]
                )
                loss += task_loss
            loss = loss / args.n_tasks
            test_loss += loss.item()
            
            prediction = predictions.sigmoid()
            
            y_true.append(label)
            y_scores.append(prediction)
    
    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()
    
    pre_list, acc_list, bac_list = [], [], []
    f1_list, aupr_list, roc_list = [], [], []
    sen_list, spe_list = [], []
    
    for i in range(args.n_tasks):
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_valid = y_true[:, i]**2 >= 0
            
            pre_list.append(precision_score(y_true[is_valid, i], y_scores[is_valid, i].round()))
            acc_list.append(accuracy_score(y_true[is_valid, i], y_scores[is_valid, i].round()))
            bac_list.append(balanced_accuracy_score(y_true[is_valid, i], y_scores[is_valid, i].round()))
            f1_list.append(f1_score(y_true[is_valid, i], y_scores[is_valid, i].round()))
            
            cof_m = confusion_matrix(y_true[is_valid, i], y_scores[is_valid, i].round())
            TN, FP, FN, TP = cof_m.ravel()
            sen_list.append(TP / (TP + FN))
            spe_list.append(TN / (TN + FP))
            aupr_list.append(average_precision_score(y_true[is_valid, i], y_scores[is_valid, i].flatten()))
            roc_list.append(roc_auc_score(y_true[is_valid, i], y_scores[is_valid, i].flatten()))
    
    metrics = [
        sum(pre_list) / len(pre_list), sum(sen_list) / len(sen_list), sum(spe_list) / len(spe_list),
        sum(acc_list) / len(acc_list), sum(bac_list) / len(bac_list), sum(f1_list) / len(f1_list),
        sum(aupr_list) / len(aupr_list), sum(roc_list) / len(roc_list)
                ]    
    return test_loss / len(loader), metrics

  
def main():
    parser = argparse.ArgumentParser(description='Fine-tuning for molecular property prediction')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration YAML file')
    parser.add_argument('--data_dir', type=str, default='./Dataset/fine-tuning/MPP/')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., AMES, Tox21, Clintox)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', type=int, default=0, help='GPU device number')
    parser.add_argument('--split_method', type=str, default='already_split',
                       help='Split method: scaffold, already_split')    
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Split ratio')
    parser.add_argument('--output_dir', type=str, default='./output/MPP/',
                       help='Output directory for results')
    parser.add_argument('--save_model', action='store_true',
                       help='Save the model or not ')
    
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for optimizer')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--n_pred_layer', type=int, default=1, choices=[1, 2, 3], help='Number of predictor layers')
    parser.add_argument('--pred_hidden', type=int, default=128, help='Hidden dimension of predictor')
    parser.add_argument('--dropout', type=float, default=0.3, help='Number of training epochs')
    
    # LLRD 
    parser.add_argument('--LLRD', type=bool, default=True, help='Use LLRD')
    parser.add_argument('--LLRD_decay_coef', type=float, default=0.85, help='LLRD decay coefficient')

    # Scheduler setting
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'none'],
                       help='Learning rate scheduler type')
    parser.add_argument('--warmup', type=int, default=10,
                       help='Warmup epochs (for cosine scheduler)')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                       help='Minimum learning rate (for cosine scheduler)')
    parser.add_argument('--first_cycle_steps', type=int, default=70, 
                       help='First cycle steps (for cosine scheduler)')
    parser.add_argument('--cycle_mult', type=float, default=1,
                       help='Cycle multiplier (for cosine scheduler)')
    parser.add_argument('--gamma', type=float, default=0.7,
                       help='Gamma (for cosine scheduler)')
    
    args = parser.parse_args()
    
    # Load the config.yaml file
    config = load_config(args.config)

    # Seed fixation
    fix_the_seed(args.seed)
    
    # Seed fixation
    fix_the_seed(args.seed)

    # Device setting
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    device = torch.device(f"cuda:0")
    
    # Task setting
    args.task_col, args.n_tasks = setup_task_config(args.dataset, config)
    config['heteroGNN_args']['out_dim'] = args.n_tasks

    df = pd.read_csv(args.data_dir+f'{args.dataset}/{args.dataset}.csv')
    if args.task_col is None:  # multi-task
        args.task_col = list(df.columns[-args.n_tasks:])
         
    # Generate directory for saving the output file
    output_path = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_path, exist_ok=True)
    
    # Data loader preparation
    train_loader, valid_loader, test_loader = get_dataloader(config, args)

    # Training
    print(f"\n{'='*50}")
    print(f"Training {args.dataset} with seed {args.seed}")
    print(f"{'='*50}\n")
    
    res = train(train_loader, valid_loader, test_loader, args, config, device, save_path=output_path)
    
    # Save the output performances   
    eval_cols = ['pre', 'sen', 'spe', 'acc', 'bac', 'f1', 'aupr', 'auc']
    
    train_df = pd.DataFrame(res[0], columns=eval_cols + ['train_loss'])
    train_df.to_csv(f'{output_path}/train_seed{args.seed}.csv', index=False)
    
    pd.DataFrame(res[1], columns=eval_cols).to_csv(
        f'{output_path}/valid_seed{args.seed}.csv', index=False
    )
    pd.DataFrame(res[2], columns=eval_cols).to_csv(
        f'{output_path}/test_seed{args.seed}.csv', index=False
    )
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()