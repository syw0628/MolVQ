import os
import math
import glob
import gzip
import shutil
import tempfile
import warnings

import torch
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from torch.utils.data import Dataset
from joblib import Parallel, delayed

import dgl

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize.rdMolStandardize import StandardizeSmiles

from descriptastorus.descriptors import rdNormalizedDescriptors

from utils.chem_utils import fg_featurizer
from utils.generate_subgraph import SubgraphGenerator
from utils.build_grover_vocab import MolVocab, atom_to_vocab, bond_to_vocab
from utils.featurizer import atom_featurizer, bond_featurizer, motif_featurizer, motif_bond_featurizer

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

def Smi2AtomGraph(smi, atom_vocab=False, bond_vocab=False):
    mol = Chem.MolFromSmiles(smi)
    if mol == None: 
        print('{} cannot convert to rdkit mol'.format(smi))
        return None
    
    # Define connection between edges
    src = []
    dst = []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx() 
        v = bond.GetEndAtomIdx()
        src.extend([u, v])
        dst.extend([v, u])  # undirectional
        
    g = dgl.graph((src, dst), num_nodes=mol.GetNumAtoms())
    g.smiles = smi

    # Node featurize 
    atom_feats = []
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        atom_feats.append(atom_featurizer(atom))
    g.ndata['f'] = torch.tensor(atom_feats)

    # Edge featurize: reuse each bond feature for both directions
    bond_feats = []
    for bond in mol.GetBonds():
        bond_feat = bond_featurizer(bond)
        bond_feats.extend([bond_feat, bond_feat])
    g.edata['f'] = torch.tensor(bond_feats)
    
    if atom_vocab:            
        atom_idx = []
        for atom in mol.GetAtoms():
            atom_idx.append(atom_vocab.stoi.get(atom_to_vocab(mol, atom), atom_vocab.other_index))
        g.ndata['atom_vocab_idx'] = torch.tensor(atom_idx)
    
    if bond_vocab:
        bond_idx = []
        nm_atoms = mol.GetNumAtoms()
        for i in range(nm_atoms):
            for j in range(i+1, nm_atoms):
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond is None:
                    continue
                bond_idx.append(bond_vocab.stoi.get(bond_to_vocab(mol, bond), bond_vocab.other_index))
        g.edata['bond_vocab_idx'] = torch.tensor(bond_idx)
    return g


def Tree2MotifGraph(smi, tree, vocab_dict):
    # Define connection between edges 
    src, dst = [], []
    fe_feats = []
    for u, v in tree.edges:
        src.extend([u, v])
        dst.extend([v, u])
        fe_feat = motif_bond_featurizer(tree.get_edge(u, v).edges[0][2:])
        fe_feats.append(fe_feat)
        fe_feats.append(fe_feat)
        
    # Exception: only self-loop exists
    if (len(fe_feats) == 0) or (len(tree.nodes) == 1):
        src.extend([0., 0.])
        dst.extend([0., 0.])
        fe_feats.append([0., 0., 0., 0., 1., 1., 0., 0., 0., 0.])
        fe_feats.append([0., 0., 0., 0., 1., 1., 0., 0., 0., 0.])
    
    if len(tree.nodes) == 0.:
        mol = Chem.MolFromSmiles(smi)
        smi = Chem.MolToSmiles(mol)
        g = dgl.graph((src, dst), num_nodes=1)
        g.smiles = smi
        g.edata['f'] = torch.tensor(fe_feats)

        motif_feat = motif_featurizer(smi, vocab_dict)
        g.ndata['f'] = torch.tensor([motif_feat[1:]])
        g.ndata['motif_vocab_idx'] = torch.tensor([motif_feat[0]])
        
        g.gdata = {'atom_map': torch.tensor(list(range(mol.GetNumAtoms()))), 'motif_map' : torch.tensor([0]*mol.GetNumAtoms())}
        return g

    g = dgl.graph((src, dst), num_nodes=len(tree.nodes))
    g.smiles = smi
    g.edata['f'] = torch.tensor(fe_feats)
        
    atom_map, motif_map, motif_idx, motif_feats = [], [], [], [] 
    for node_i in tree.nodes:
        node = tree.get_node(node_i)
        atom_map.extend(node.atom_mapping.keys())
        motif_map.extend([node_i]*len(node.atom_mapping.keys()))
        motif_feat = motif_featurizer(node.smiles, vocab_dict)
        motif_feats.append(motif_feat[1:])
        motif_idx.append(motif_feat[0])
        
    g.ndata['f'] = torch.tensor(motif_feats)
    g.ndata['motif_vocab_idx'] = torch.tensor(motif_idx) 
    
    g.gdata = {}
    g.gdata['atom_map'], g.gdata['motif_map'] = torch.tensor(atom_map), torch.tensor(motif_map)
    return g


class MolGraphDataset(Dataset):
    def __init__(self, data_path, frag_vocab_path, atom_vocab_path=False, bond_vocab_path=False, smiles_col='st_smiles', 
                 task_col=None, standardize=True, cpus=1, method=None):
              
        df = pd.read_csv(data_path)

        if smiles_col not in df.columns:
            raise ValueError("The specified SMILES column name is not found in the data file.")

        self.task_col = task_col
        if type(task_col) is list:
            self.tasks = task_col
        elif type(task_col) is str:
            self.tasks = [task_col]
        elif task_col == None:
            self.tasks = None
        else: 
            self.labels = [column for column in df.columns if column != smiles_col]
        
        if self.tasks != None: 
            self.labels = df[self.tasks].values
            
        self.smiles_list = df[smiles_col].values.tolist()
        if standardize:
            st_smiles_list = []
            for i, smi in enumerate(self.smiles_list):
                try:
                    st_smiles = StandardizeSmiles(smi)
                    if not st_smiles:
                        raise ValueError('Standardized SMILES is empty.')
                except Exception as error:
                    raise ValueError('Failed to standardize SMILES at row {}: {!r}'.format(i, smi)) from error
                st_smiles_list.append(st_smiles)
            self.smiles_list = st_smiles_list
        if atom_vocab_path:
            self.atom_vocab = MolVocab.load_vocab(atom_vocab_path)
        else: 
            self.bond_path = False
            
        if bond_vocab_path:
            self.bond_vocab = MolVocab.load_vocab(bond_vocab_path)
        else: 
            self.bond_vocab = False
        
        self.frag_vocab_path = frag_vocab_path

        if method == None: 
            self.method = self.frag_vocab_path.split('/')[-1].split('_')[0]
        else: 
            self.method = method
        
        self.cpus = cpus
        self.process()

    def process(self):
        self.atom_graphs, self.motif_graphs = [], []
       
        tree_generator = SubgraphGenerator(self.frag_vocab_path, self.method)
        self.vocab_dict = {smiles:i for i,smiles in enumerate(tree_generator.vocab_dict.keys())}
        
        if self.cpus <= 1:
            valid_indices = []
            for i in tqdm(range(len(self.smiles_list))):
                smi = self.smiles_list[i]
                mol = Chem.MolFromSmiles(smi)
                if mol == None: 
                    print('{} cannot convert to rdkit mol'.format(smi))
                    continue
                # atom-level graph 
                atom_g = Smi2AtomGraph(smi, atom_vocab=self.atom_vocab, bond_vocab=self.bond_vocab)

                tree = tree_generator.generate(mol)
                motif_g = Tree2MotifGraph(smi, tree, self.vocab_dict)
                if motif_g != None:
                    self.atom_graphs.append(atom_g)
                    self.motif_graphs.append(motif_g)
                    valid_indices.append(i)
            if self.task_col != None:            
                self.smiles_list = [self.smiles_list[i] for i in valid_indices]   
        else: 
            valid_smiles = []
            for smi in tqdm(self.smiles_list):
                mol = Chem.MolFromSmiles(smi)
                if mol == None: 
                    print('{} cannot convert to rdkit mol'.format(smi))
                    continue
                valid_smiles.append(smi)
            self.smiles_list = valid_smiles

            # atom-level graphs
            atom_tasks = (
                delayed(Smi2AtomGraph)(smi, atom_vocab=self.atom_vocab, bond_vocab=self.bond_vocab)
                for smi in self.smiles_list
            )
            self.atom_graphs = Parallel(n_jobs=self.cpus)(
                tqdm(atom_tasks, total=len(self.smiles_list))
            )

            # fragment-level graphs
            tasks = [delayed(process_motif_parallel)(smi, tree_generator, self.vocab_dict) for smi in self.smiles_list]
            self.motif_graphs = Parallel(n_jobs=self.cpus)(tqdm(tasks, total=len(tasks)))

            if self.task_col != None:
                zipped = [(s, a, m, l) for s, a, m, l in zip(self.smiles_list, self.atom_graphs, self.motif_graphs, self.labels) if m is not None]
                if zipped:
                    self.smiles_list, self.atom_graphs, self.motif_graphs, self.labels = map(list, zip(*zipped))
            else: 
                zipped = [(s, a, m) for s, a, m in zip(self.smiles_list, self.atom_graphs, self.motif_graphs) if m is not None]
                if zipped:  
                    self.smiles_list, self.atom_graphs, self.motif_graphs = map(list, zip(*zipped))


    def save_dataset(self, n_split, save_dir, dataset_name): 
        if not isinstance(n_split, int) or not 1 <= n_split <= 10:
            raise Exception('n_split must be a positive integer; only 10 or fewer chunks are allowed.')
        os.makedirs('{}/{}'.format(save_dir, dataset_name), exist_ok=True)
        save_dir = '{}/{}'.format(save_dir, dataset_name)

        meta_path = os.path.join(save_dir, "meta_data.joblib")
        meta_data = {'smiles': self.smiles_list, 'atom_map': [g.gdata['atom_map'] for g in self.motif_graphs], 'motif_map': [g.gdata['motif_map'] for g in self.motif_graphs]}
        meta_data['label'] = self.labels
        joblib.dump(meta_data, meta_path)

        total = len(self.atom_graphs)
        chunk_size = math.ceil(total / n_split)

        for i in tqdm(range(n_split)):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, total)

            atom_g_chunk = self.atom_graphs[start:end]
            motif_g_chunk = self.motif_graphs[start:end]

            # Save
            atom_g_path = os.path.join(save_dir, "atom_g_{}.bin".format(i))
            motif_g_path = os.path.join(save_dir, "motif_g_{}.bin".format(i))

            dgl.save_graphs(atom_g_path, atom_g_chunk)
            dgl.save_graphs(motif_g_path, motif_g_chunk)
        print("{} chunks of dataset are saved in '{}'.".format(n_split, save_dir))

    def load_dataset(self, load_dir):
        atom_g_files = sorted(glob.glob(os.path.join(load_dir, "atom_g_*.bin")))
        motif_g_files = sorted(glob.glob(os.path.join(load_dir, "motif_g_*.bin")))

        meta_path = os.path.join(load_dir, "meta_data.joblib")
        meta_data = joblib.load(meta_path)
        self.smiles_list = meta_data['smiles']
        if 'label' in meta_data.keys():
            self.labels = meta_data['label']

        atom_g_list, motif_g_list = [], [] 
        for atom_g_file, motif_g_file in tqdm(zip(atom_g_files, motif_g_files)):
            atom_g, _ = dgl.load_graphs(atom_g_file)
            motif_g, _ = dgl.load_graphs(motif_g_file)
            
            atom_g_list.extend(atom_g)
            motif_g_list.extend(motif_g)

        for g, am, mm in zip(motif_g_list, meta_data['atom_map'], meta_data['motif_map']):
            g.gdata = {'atom_map': am, 'motif_map': mm}
        self.atom_graphs = atom_g_list
        self.motif_graphs = motif_g_list
        
        print("Dataset is loaded from '{}'.".format(load_dir))
    
    def __len__(self):
        return len(self.atom_graphs)
    
    def __getitem__(self, idx):
        if self.task_col != None:
            if hasattr(idx, "__iter__"):
                return [(self.smiles_list[i], self.atom_graphs[i], self.motif_graphs[i], self.labels[i]) for i in idx]
            return (self.smiles_list[idx], self.atom_graphs[idx], self.motif_graphs[idx], self.labels[idx])
        else: 
            if hasattr(idx, "__iter__"):
                return [(self.smiles_list[i], self.atom_graphs[i], self.motif_graphs[i]) for i in idx]
            return (self.smiles_list[idx], self.atom_graphs[idx], self.motif_graphs[idx])

def process_motif_parallel(smi, generator, vocab_dict):
    mol = Chem.MolFromSmiles(smi)
    try:
        tree = generator.generate(mol)
        return Tree2MotifGraph(smi, tree, vocab_dict)
    except:
        print(smi)
        return None

def _to_homo_preserve_gdata(g):
    gdata = getattr(g, "gdata", None)

    u, v = g.edges()
    new_g = dgl.graph((u, v), num_nodes=g.num_nodes())
    
    if len(g.ntypes) > 1:
        g_temp = dgl.to_homogeneous(g)
        new_g.ndata.update(g_temp.ndata)
        new_g.edata.update(g_temp.edata)
    else:
        new_g.ndata.update(g.ndata)
        new_g.edata.update(g.edata)
    
    if gdata is not None:
        new_g.gdata = gdata
        
    return new_g

def custom_collate_fn(batch):
    smi_list, atom_g_list, motif_g_list = zip(*batch)
    
    motif_g_list = tuple(_to_homo_preserve_gdata(g) for g in motif_g_list)
    
    valid_indices = []
    for i, (atom_g, motif_g) in enumerate(zip(atom_g_list, motif_g_list)):
        if atom_g.num_nodes() > 0 and motif_g.num_nodes() > 0:
            valid_indices.append(i)
        else:
            print(f"Warning: Skipping empty graph at batch index {i}")
    
    if len(valid_indices) == 0:
        return None
    
    atom_g_list = [atom_g_list[i] for i in valid_indices]
    motif_g_list = [motif_g_list[i] for i in valid_indices]
    smi_list = [smi_list[i] for i in valid_indices]

    atom_g_batch = dgl.batch(atom_g_list)
    motif_g_batch = dgl.batch(motif_g_list)
    
    atom_maps = []
    motif_maps = []
    
    atom_offset = 0
    motif_offset = 0
    
    for atom_g, motif_g in zip(atom_g_list, motif_g_list):
        atom_map = motif_g.gdata.get('atom_map', torch.tensor([]))
        motif_map = motif_g.gdata.get('motif_map', torch.tensor([]))
        
        if len(atom_map) > 0:
            atom_maps.append(atom_map + atom_offset)
        if len(motif_map) > 0:
            motif_maps.append(motif_map + motif_offset)
        
        atom_offset += atom_g.num_nodes()
        motif_offset += motif_g.num_nodes()
    
    atom_map_batch = torch.cat(atom_maps) if atom_maps else torch.tensor([])
    motif_map_batch = torch.cat(motif_maps) if motif_maps else torch.tensor([])
    
    return atom_g_batch, motif_g_batch, atom_map_batch, motif_map_batch    

def extract_descriptor(smi):
    generator = rdNormalizedDescriptors.RDKit2DNormalized()
    return np.nan_to_num(np.array(generator.process(smi)[1:], dtype=np.float32), nan=0.0)
    
class MolAugGraphDataset(Dataset):
    def __init__(self, data_path=None, frag_vocab_path=None, atom_vocab_path=None, aug_data_path=None,
                 smiles_col='st_smiles', task_col=None, cpus=1, method=None, standardize=False):
        self.task_col, self.cpus = task_col, cpus
        if aug_data_path is not None:
            self.load_dataset(aug_data_path)
        else:
            if (self.task_col == 'functional_group') | (self.task_col == 'rdkit_des'):
                self.dataset = MolGraphDataset(data_path=data_path, frag_vocab_path=frag_vocab_path, atom_vocab_path=atom_vocab_path,
                                                smiles_col=smiles_col, task_col=None, cpus=cpus, method=method, standardize=standardize)
            else:
                self.dataset = MolGraphDataset(data_path=data_path, frag_vocab_path=frag_vocab_path, atom_vocab_path=atom_vocab_path,
                                                smiles_col=smiles_col, task_col=self.task_col, cpus=self.cpus, method=method, standardize=standardize)
            self.smiles_list = self.dataset.smiles_list
        
    def process(self):
        if self.task_col not in ('functional_group', 'rdkit_des'):
            self.label = self.dataset.labels
        else:
            self.label = []
            if self.cpus > 1: 
                featurizer = fg_featurizer if self.task_col == 'functional_group' else extract_descriptor
                tasks = [delayed(featurizer)(smi) for smi in self.dataset.smiles_list]
                self.label = Parallel(n_jobs=self.cpus)(tqdm(tasks, total=len(tasks)))               
        
        self.aug_g_list, self.atom_g_list, self.motif_g_list = [], [], []
        for i in tqdm(range(len(self.dataset))):
            _, atom_g, motif_g = self.dataset[i][:3]
            atom_map = motif_g.gdata['atom_map'].long()
            motif_map = motif_g.gdata['motif_map'].long()
            
            num_node_dict = {'a': atom_g.num_nodes(), 'p': motif_g.num_nodes()}
            edge_dict = {('a', 'b', 'a') : atom_g.edges(),
                         ('p', 'r', 'p') : motif_g.edges(),
                         ('a', 'j', 'p') : (atom_map, motif_map),
                         ('p', 'j', 'a') : (motif_map, atom_map)}
            
            hetero_g = dgl.heterograph(edge_dict, num_nodes_dict=num_node_dict)            
            
            self._copy_features_to_aug(hetero_g, atom_g, motif_g)

            self.aug_g_list.append(hetero_g)
            self.atom_g_list.append(atom_g)
            self.motif_g_list.append(motif_g)    

            if self.cpus <= 1 and self.task_col in ['functional_group', 'rdkit_des']:
                if self.task_col == 'functional_group':
                    self.label.append(fg_featurizer(self.dataset.smiles_list[i]))
                elif self.task_col == 'rdkit_des':
                    generator = rdNormalizedDescriptors.RDKit2DNormalized()
                    des_label = np.array(generator.process(self.dataset.smiles_list[i]))[1:]
                    des_label = np.nan_to_num(des_label, nan=0.0)
                    self.label.append(des_label)    
                   
    @staticmethod
    def _copy_features_to_aug(aug_g, atom_g, motif_g):
        # Keep all fields (including vocabulary indices), without dtype conversion.
        aug_g.nodes['a'].data.update(atom_g.ndata)
        aug_g.nodes['p'].data.update(motif_g.ndata)
        aug_g.edges[('a', 'b', 'a')].data.update(atom_g.edata)
        aug_g.edges[('p', 'r', 'p')].data.update(motif_g.edata)

    @staticmethod
    def _split_aug_graph(aug_g, feature_keys):
        graphs = []
        for ntype, etype, prefix in (
            ('a', ('a', 'b', 'a'), 'atom'),
            ('p', ('p', 'r', 'p'), 'motif'),
        ):
            # EID order keeps every edge feature aligned, including parallel edges.
            src, dst = aug_g.edges(etype=etype, order='eid')
            graph = dgl.graph((src, dst), num_nodes=aug_g.num_nodes(ntype),
                              idtype=feature_keys[prefix + '_idtype'], device=aug_g.device)
            for key in feature_keys[prefix + '_node']:
                graph.ndata[key] = aug_g.nodes[ntype].data[key]
            for key in feature_keys[prefix + '_edge']:
                graph.edata[key] = aug_g.edges[etype].data[key]
            graphs.append(graph)
        return graphs

    def save_dataset(self, n_split, save_dir, dataset_name):
        if not isinstance(n_split, int) or not 1 <= n_split <= 10:
            raise Exception('n_split must be a positive integer; only 10 or fewer chunks are allowed.')
        total = len(self.aug_g_list)
        if len(self.atom_g_list) != total or len(self.motif_g_list) != total:
            raise ValueError('Atom, motif and augmented graph counts must match.')
        if len(self.smiles_list) != total:
            raise ValueError('SMILES and graph counts must match.')
        save_dir = os.path.join(save_dir, dataset_name)
        os.makedirs(save_dir, exist_ok=True)

        # A manifest avoids loading stale chunks left by an earlier save.
        meta_data = {
            'format_version': 2,
            'smiles': self.smiles_list,
            'atom_map': [g.gdata['atom_map'] for g in self.motif_g_list],
            'motif_map': [g.gdata['motif_map'] for g in self.motif_g_list],
            'label': getattr(self, 'label', []),
            'feature_keys': [],
            'aug_files': [],
        }
        chunk_size = max(1, math.ceil(total / n_split))
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_path = os.path.join(temp_dir, 'graphs.bin')
            for chunk_idx, start in enumerate(tqdm(range(0, total, chunk_size))):
                end = min(start + chunk_size, total)
                aug_chunk = []
                for i in range(start, end):
                    atom_g, motif_g = self.atom_g_list[i], self.motif_g_list[i]
                    # Separate feature frames so saving does not mutate live graphs.
                    aug_g = self.aug_g_list[i].local_var()
                    self._copy_features_to_aug(aug_g, atom_g, motif_g)
                    aug_chunk.append(aug_g)
                    meta_data['feature_keys'].append({
                        'atom_idtype': atom_g.idtype,
                        'motif_idtype': motif_g.idtype,
                        'atom_node': list(atom_g.ndata.keys()),
                        'atom_edge': list(atom_g.edata.keys()),
                        'motif_node': list(motif_g.ndata.keys()),
                        'motif_edge': list(motif_g.edata.keys()),
                    })
                filename = 'aug_g_{}.bin.gz'.format(chunk_idx)
                dgl.save_graphs(raw_path, aug_chunk)
                with open(raw_path, 'rb') as source:
                    with gzip.open(os.path.join(save_dir, filename), 'wb', compresslevel=3) as dest:
                        shutil.copyfileobj(source, dest)
                meta_data['aug_files'].append(filename)

        joblib.dump(meta_data, os.path.join(save_dir, 'meta_data.joblib'), compress=3)
        print("{} chunks of dataset are saved in '{}'.".format(len(meta_data['aug_files']), save_dir))

    def load_dataset(self, load_dir):
        meta_data = joblib.load(os.path.join(load_dir, 'meta_data.joblib'))
        self.smiles_list = meta_data['smiles']
        self.label = meta_data['label']
        atom_g_list, motif_g_list, aug_g_list = [], [], []

        if meta_data.get('format_version') == 2:
            # DGL reads paths; decompress one chunk at a time, without a full
            # uncompressed copy in RAM or permanent extracted graph files.
            with tempfile.TemporaryDirectory() as temp_dir:
                raw_path = os.path.join(temp_dir, 'graphs.bin')
                for filename in tqdm(meta_data['aug_files']):
                    with gzip.open(os.path.join(load_dir, filename), 'rb') as source:
                        with open(raw_path, 'wb') as dest:
                            shutil.copyfileobj(source, dest)
                    aug_chunk, _ = dgl.load_graphs(raw_path)
                    for aug_g in aug_chunk:
                        idx = len(aug_g_list)
                        atom_g, motif_g = self._split_aug_graph(
                            aug_g, meta_data['feature_keys'][idx])
                        atom_g.smiles = self.smiles_list[idx]
                        motif_g.smiles = self.smiles_list[idx]
                        atom_g_list.append(atom_g)
                        motif_g_list.append(motif_g)
                        aug_g_list.append(aug_g)
            if len(aug_g_list) != len(self.smiles_list):
                raise ValueError('Saved graph and SMILES counts do not match.')
        else:
            # Backward compatibility with the original three-file format.
            def chunk_number(filename):
                return int(os.path.basename(filename).rsplit('_', 1)[1].split('.')[0])

            file_groups = [
                sorted(glob.glob(os.path.join(load_dir, prefix + '_g_*.bin')), key=chunk_number)
                for prefix in ('atom', 'motif', 'aug')]
            if len(set(map(len, file_groups))) != 1:
                raise ValueError('Saved atom, motif and augmented chunk counts must match.')
            for filenames in tqdm(zip(*file_groups)):
                for graphs, filename in zip((atom_g_list, motif_g_list, aug_g_list), filenames):
                    graphs.extend(dgl.load_graphs(filename)[0])
            for aug_g, atom_g, motif_g in zip(aug_g_list, atom_g_list, motif_g_list):
                self._copy_features_to_aug(aug_g, atom_g, motif_g)

        for g, am, mm in zip(motif_g_list, meta_data['atom_map'], meta_data['motif_map']):
            g.gdata = {'atom_map': am, 'motif_map': mm}
        self.atom_g_list = atom_g_list
        self.motif_g_list = motif_g_list
        self.aug_g_list = aug_g_list
        print("Dataset is loaded from '{}'.".format(load_dir))

    def __len__(self):
        return len(self.aug_g_list)
    
    def __getitem__(self, idx):
        columns = (self.aug_g_list, self.atom_g_list, self.motif_g_list)
        if self.task_col is not None:
            columns += (self.label,)
        if hasattr(idx, '__iter__'):
            return [tuple(column[i] for column in columns) for i in idx]
        return tuple(column[idx] for column in columns)
