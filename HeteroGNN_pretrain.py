import os 
import sys
sys.path.append("./utils")
import torch 
import random
import argparse
import yaml
import numpy as np 
import pandas as pd 
from types import SimpleNamespace

from tqdm import tqdm
from torch import nn 
import torch.optim as optim
from dgl.dataloading import GraphDataLoader

from sklearn.metrics import r2_score

from model.HeteroGNN import PharmHGT_modified
from utils.splitter import random_split
from utils.warmup_scheduler import NoamLR
from utils.build_grover_vocab import MolVocab
from utils.prepare_dataset import MolGraphDataset, MolAugGraphDataset

reg_criterion = nn.MSELoss()

def model_eval(model, test_loader, device):
    with torch.no_grad():
        model.eval()
        test_loss = 0 
        epoch_iter = tqdm(test_loader, desc="Iteration", disable=not sys.stdout.isatty())
        label_list, pred_list = torch.tensor([]), torch.tensor([])
        
        for step, batch in enumerate(epoch_iter):
            aug_g, atom_g, motif_g, labels = batch
            batch_g = (aug_g.to(device), atom_g.to(device), motif_g.to(device))
            labels = labels.to(device)
            _, logits = model(batch_g)
                        
            loss = reg_criterion(logits, labels).mean()
            
            pred_list = torch.cat([pred_list, logits.cpu()], dim=0)
            label_list = torch.cat([label_list, labels.cpu()], dim=0)
            test_loss += loss.item() 
            
        test_loss = test_loss / len(epoch_iter)
        return test_loss, label_list, pred_list 

def fix_the_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def main(args):
    with open(args.config, 'r') as config_file:
        config = yaml.safe_load(config_file)
    data_args = SimpleNamespace(**config['data_args'])
    vq_args = SimpleNamespace(**config['vq_args'])
    
    # Seed fixation 
    fix_the_seed(args.seed)
    
    # Device setting
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    # Load dataset
    if args.load_data_path: 
        print("Load the aug dataset...")
        aug_dataset = MolAugGraphDataset(data_path=None, aug_data_path=args.load_data_path, frag_vocab_path=args.vocab_path, 
                                         smiles_col=None, task_col=args.task, cpus=args.num_workers)
    else: 
        atom_vocab = MolVocab.load_vocab(args.atom_vocab_path)       
        print("Construct the aug dataset...")
        aug_dataset = MolAugGraphDataset(data_path=None, frag_vocab_path=args.vocab_path, smiles_col=None, 
                                         task_col=args.task, cpus=args.num_workers) 
        aug_dataset.process()

    # Dataset split 
    train_dataset, valid_dataset = random_split(aug_dataset, frac_train=1-args.valid_ratio,
                                                frac_test=args.valid_ratio, frac_valid=None, seed=args.seed)
    train_loader = GraphDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = GraphDataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Build VQVAE from YAML settings and load the pretrained weights.
    model = PharmHGT_modified(args, vq_args, data_args).to(device)
    model.from_vq_pretrained(args.vq_model_path)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    scheduler = NoamLR(optimizer = optimizer, warmup_epochs = [args.warmup], total_epochs = [args.epochs],
                       steps_per_epoch = len(train_loader), init_lr = [args.init_lr], max_lr = [args.lr], final_lr = [args.final_lr])

    # Training 
    train_history, valid_history, r2_history = [], [], []
   
    os.makedirs(args.save_model_path , exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        train_loss = 0 
        print("====epoch {}====".format(str(epoch)))
        epoch_iter = tqdm(train_loader, desc="Iteration", disable=not sys.stdout.isatty())
        model.train()
        #grad_flow_history, layers_ref = [], None
        for step, batch in enumerate(epoch_iter):
            aug_g, atom_g, motif_g, labels = batch
            batch_g = (aug_g.to(device), atom_g.to(device), motif_g.to(device))
            labels = labels.to(device)
            
            emb, logits = model(batch_g)
            loss = reg_criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            epoch_iter.set_postfix({"train_loss": f"{loss.item():.4f}"})

            if not sys.stdout.isatty():
                if step % 1000 == 0:
                    print(f"[Iter {step}] loss={loss:.4f}")
        
        print("Train Loss: ", train_loss/len(epoch_iter))
        train_history.append(train_loss/len(epoch_iter))
        
        valid_loss, labels, preds = model_eval(model, valid_loader, device)
        valid_r2 = r2_score(labels, preds)
        print("Valid Loss / R2: {} / {}".format(valid_loss, valid_r2))

        valid_history.append(valid_loss)
        r2_history.append(valid_r2)
        
        torch.save(model.state_dict(), '{}/state_dict_{}.pth'.format(args.save_model_path, epoch))
        pd.DataFrame({'train_loss':train_history, 'valid_loss':valid_history, 'valid_r2':r2_history}).to_csv('{}/log.csv'.format(args.save_model_path), index=False)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HeteroGNN pretraining for normalized molecule descriptor prediction')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=24)
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
        
    parser.add_argument('--task', type=str, default='rdkit_des')
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('--data_path', type=str, default='./Dataset/pretraining/guacamol_smiles.csv', 
                        help='source CSV used when dataset_path is empty')
    parser.add_argument('--atom_vocab_path', type=str, default='./vocab/guacamol_atom_vocab.pkl')
    parser.add_argument('--vocab_path', type=str, default='./vocab/guacamol_relmole_vocab.txt')
    parser.add_argument('--load_data_path', type=str, default='./constructed_data/guacamol_aug_dataset',
                        help='saved atom/motif graph dataset directory')
    parser.add_argument('--save_model_path', type=str, default='./pretrained_model/HeteroGNN')
    parser.add_argument('--config', type=str, default='./config.yaml')
    parser.add_argument('--vq_model_path', type=str, default="./pretrained_model/VQ.pth")
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--num_att_head', type=int, default=6)
    parser.add_argument('--readout', type=str, default='gru')
    parser.add_argument('--mvmp_dropout', type=float, default=0.)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--out_dim', type=float, default=200)

    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--init_lr', type=float, default=1e-6)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--final_lr', type=float, default=3e-4)
    parser.add_argument('--warmup', type=float, default=5)
    parser.add_argument('--decay', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)

    args = parser.parse_args()

    main(args)
