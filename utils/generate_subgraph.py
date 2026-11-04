import sys
sys.path.append("../")

from copy import copy
from collections import defaultdict
from rdkit import Chem 
from rdkit import RDLogger

from rdkit.Chem.MolStandardize.rdMolStandardize import StandardizeSmiles
from utils.chem_utils import smi_to_mol, mol_to_smi, get_clique_mol
from utils.molecule import Molecule

RDLogger.DisableLog('rdApp.*') 

class MolInSubgraph:
    def __init__(self, smi, kekulize=False, vocab=False):
        '''
        self.subgraphs: 각 subgraph에 속해있는 원자들의 index를 보여준다 -> {subgraph_idx : [atom1_idx, atom2_idx, ...]}
        self.subgraphs_smis: 각 subgraph를 smiles로 바꿔준다 -> {subgraph_idx : subgraph_smiles}
        self.inversed_index: assign atom idx to pid: 각 aid의 atom이 몇번 인덱스의 subgraph에 속해있는지를 정리해주는 dict -> {atom_idx : subgraph_idx}
        '''
        self.smi = smi
        self.mol = Chem.MolFromSmiles(smi)
        self.kekulize = kekulize
        self.vocab = vocab 
        
        #Subgraph initialization
        self.subgraphs, self.subgraphs_smis, self.inversed_index = self.initialize_subgraph(self.mol, self.vocab)
        self.upid_cnt = len(self.subgraphs)
        self.dirty = True
        self.smi2pids = {} 
        
    def initialize_subgraph(self, mol, vocab=False):
        subgraphs_smi = {}
        inversed_index = defaultdict(list)
        subgraphs = []
        n_atoms = mol.GetNumAtoms()
        
        if n_atoms < 2:
            subgraphs = {0: [mol.GetAtomWithIdx(0).GetIdx()]} 
        else:
            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom().GetIdx()
                a2 = bond.GetEndAtom().GetIdx()
                if not bond.IsInRing():
                    subgraphs.append([a1, a2])
            rings = Chem.GetSymmSSSR(mol)
            subgraphs.extend([list(ring) for ring in rings])
            subgraphs = dict(enumerate(subgraphs))
            
            m_subgraphs = subgraphs.copy()
            if vocab:
                nei_list = [[] for i in range(n_atoms)]
                for i in range(len(m_subgraphs)):
                    for atom in m_subgraphs[i]:
                        nei_list[atom].append(i)
                
                for i in range(len(m_subgraphs)):
                    if len(m_subgraphs[i]) <= 2: continue
                    for atom in m_subgraphs[i]:
                        for j in nei_list[atom]:
                            if i >= j or len(m_subgraphs[j]) <= 2: continue
                            inter = set(m_subgraphs[i]) & set(m_subgraphs[j])
                            if len(inter) > 2:
                                m_subgraphs[i].extend(m_subgraphs[j])
                                m_subgraphs[i] = list(set(m_subgraphs[i]))
                                m_subgraphs[j] = []
                
                m_subgraphs = {k: v for k, v in m_subgraphs.items() if v}
                
                if all([StandardizeSmiles(Chem.MolToSmiles(get_clique_mol(mol, m_subgraphs[k]))) in vocab for k in m_subgraphs]):
                    subgraphs = m_subgraphs
    
        for i in subgraphs.keys():
            sub_idx = subgraphs[i]
            if len(sub_idx) == 1:
                subgraphs_smi[i] = mol.GetAtomWithIdx(sub_idx[0]).GetSymbol()
                subgraphs[i] = {sub_idx[0]}
            else:
                subgraphs_smi[i] = StandardizeSmiles(Chem.MolToSmiles(get_clique_mol(mol, subgraphs[i])))
                subgraphs[i] = {*subgraphs[i]}

            for idx in subgraphs[i]:
                inversed_index[idx].append(i)
    
        return subgraphs, subgraphs_smi, inversed_index

    def get_nei_subgraphs(self):
        nei_subgraphs, merge_pids = [], []
        processed_pairs = set()
        
        for key in self.subgraphs:
            subgraph = self.subgraphs[key]
            local_nei_pid = []
            for aid in subgraph:
                atom = self.mol.GetAtomWithIdx(aid)
                for nei in atom.GetNeighbors():
                    nei_idx = nei.GetIdx()
                    if nei_idx in subgraph:
                        continue
                    for nei_pid in self.inversed_index[nei_idx]:
                        if nei_pid in self.subgraphs and nei_pid != key:
                            local_nei_pid.append(nei_pid)
                    
            local_nei_pid = set(local_nei_pid)
            for nei_pid in local_nei_pid:
                pair = tuple(sorted([key, nei_pid]))
                if pair in processed_pairs:
                    continue
                processed_pairs.add(pair)
                
                new_subgraph = copy(subgraph) 
                new_subgraph.update(self.subgraphs[nei_pid])
                nei_subgraphs.append(new_subgraph)
                merge_pids.append((key, nei_pid))
        return nei_subgraphs, merge_pids
        
    def get_nei_smis(self):
        if self.dirty:
            nei_subgraphs, merge_pids = self.get_nei_subgraphs()
            nei_smis, self.smi2pids = [], {}
            for i, subgraph in enumerate(nei_subgraphs):
                submol = get_clique_mol(self.mol, list(subgraph)) ####
                smi = StandardizeSmiles(mol_to_smi(submol))
                nei_smis.append(smi)
                self.smi2pids.setdefault(smi, [])
                self.smi2pids[smi].append(merge_pids[i])
            self.dirty = False
        else:
            nei_smis = list(self.smi2pids.keys())
        return nei_smis
   
    def merge(self, smi):
        if self.dirty:
            self.get_nei_smis()
        if smi in self.smi2pids:
            for pid1, pid2 in self.smi2pids[smi]:
                if pid1 in self.subgraphs and pid2 in self.subgraphs:
                    new_pid = self.upid_cnt
                    merged_atoms = self.subgraphs[pid1] | self.subgraphs[pid2]
                    self.subgraphs[new_pid] = merged_atoms
                    self.subgraphs_smis[new_pid] = smi
    
                    for aid in merged_atoms:
                        pids = self.inversed_index[aid]
                        pids = [p for p in pids if p not in (pid1, pid2)]
                        pids.append(new_pid)
                        self.inversed_index[aid] = pids
                    
                    del self.subgraphs[pid1]
                    del self.subgraphs[pid2]
                    del self.subgraphs_smis[pid1]
                    del self.subgraphs_smis[pid2]
                    self.upid_cnt += 1
        self.dirty = True
    
    def get_smis_subgraphs(self):
        res = []
        for pid in self.subgraphs_smis:
            smi = self.subgraphs_smis[pid]
            indices = list(self.subgraphs[pid])
            res.append((smi, indices))
        return res

def parallel_merge(mol_smi):
    mol, merge_smi = mol_smi
    mol.merge(merge_smi)
    return mol

def freq_cnt(mol):
    freqs = {}
    nei_smis = mol.get_nei_smis()
    for smi in nei_smis:
        freqs.setdefault(smi, 0)
        freqs[smi] += 1
    return freqs, mol

def init_decomp(mol, vocab, method='relmole'):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]], [mol_to_smi(get_clique_mol(mol, [0]))]

    if method == 'jtvae':
        raw_cliques = []
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            if not bond.IsInRing():
                raw_cliques.append([a1, a2])
        
        rings = [list(x) for x in Chem.GetSymmSSSR(mol)]
        raw_cliques.extend(rings)
        cliques = raw_cliques.copy()

        nei_list = [[] for _ in range(n_atoms)]
        for i in range(len(cliques)):
            for atom in cliques[i]:
                nei_list[atom].append(i)

        for i in range(len(cliques)):
            if len(cliques[i]) <= 2: continue
            for atom in cliques[i]:
                for j in nei_list[atom]:
                    if i >= j or len(cliques[j]) <= 2: continue
                    inter = set(cliques[i]) & set(cliques[j])
                    if len(inter) > 2:
                        cliques[i].extend(cliques[j])
                        cliques[i] = list(set(cliques[i]))
                        cliques[j] = []     

        cliques = [c for c in cliques if len(c) > 0]
        clique_smis = [StandardizeSmiles(mol_to_smi(get_clique_mol(mol, clique))) for clique in cliques]

        if all([sub_smi in vocab for sub_smi in clique_smis]):
            return cliques, clique_smis

        clique_smis = [StandardizeSmiles(mol_to_smi(get_clique_mol(mol, clique))) for clique in raw_cliques]
        return raw_cliques, clique_smis

    elif method == 'relmole':
        PATT = {
            'HETEROATOM': '[!#6]',
            'DOUBLE_TRIPLE_BOND': '*=,#*',
            'ACETAL': '[CX4]([O,N,S])[O,N,S]'
        }
        PATT = {k: Chem.MolFromSmarts(v) for k, v in PATT.items()}

        raw_rings = [set(x) for x in Chem.GetSymmSSSR(mol)]
        rings = raw_rings.copy()
        
        flag = True
        while flag:
            flag = False
            for i in range(len(rings)):
                if len(rings[i]) == 0:
                    continue
                for j in range(i + 1, len(rings)):
                    shared_atoms = rings[i] & rings[j]
                    if len(shared_atoms) > 2:
                        merged_candidate = list(rings[i] | rings[j])
                        rings[i].update(rings[j])
                        rings[j] = set()
                        flag = True
        rings = [r for r in rings if len(r) > 0]
        ring_smis = [StandardizeSmiles(mol_to_smi(get_clique_mol(mol, ring))) for ring in rings]

        if not all(sub_smi in vocab for sub_smi in ring_smis):
            rings = raw_rings

        fgs = []
        marks = set()
        for patt in PATT.values():
            for sub in mol.GetSubstructMatches(patt):
                marks.update(sub)
        atom2fg = [[] for _ in range(n_atoms)]
        for atom in marks:
            fgs.append({atom})
            atom2fg[atom] = [len(fgs) - 1]
        for bond in mol.GetBonds():
            if bond.IsInRing():
                continue
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a1 in marks and a2 in marks:
                assert a1 != a2
                assert len(atom2fg[a1]) == 1 and len(atom2fg[a2]) == 1
                fgs[atom2fg[a1][0]].update(fgs[atom2fg[a2][0]])
                fgs[atom2fg[a2][0]] = set()
                atom2fg[a2] = atom2fg[a1]
            elif a1 in marks:
                fgs[atom2fg[a1][0]].add(a2)
                atom2fg[a2].extend(atom2fg[a1])
            elif a2 in marks:
                fgs[atom2fg[a2][0]].add(a1)
                atom2fg[a1].extend(atom2fg[a2])
            else:
                fgs.append({a1, a2})
                atom2fg[a1].append(len(fgs) - 1)
                atom2fg[a2].append(len(fgs) - 1)

        tmp = []
        for fg in fgs:
            if len(fg) == 0:
                continue
            if len(fg) == 1 and mol.GetAtomWithIdx(list(fg)[0]).IsInRing():
                continue
            tmp.append(fg)
        fgs = tmp
        fgs.extend(rings)

        fgs = [list(fg) for fg in fgs]
        fg_smis = [StandardizeSmiles(mol_to_smi(get_clique_mol(mol, fg))) for fg in fgs]
        return fgs, fg_smis

    else:
        raise ValueError(f"Unknown method: {method}")

class SubgraphGenerator:
    def __init__(self, vocab_path, method):
        self.kekulize = False
        self.idx2subgraph, self.subgraph2idx, self.vocab_dict = [], {}, {}
        with open(vocab_path, 'r') as fin:
            lines = fin.read().strip().split('\n')
            for line in lines:
                try:
                    smi, atom_num, freq, _ = line.strip().split('\t')
                except: 
                    smi, atom_num, freq = line.strip().split('\t')
                #smi, atom_num = line.strip().split('\t')
                self.vocab_dict[smi] = int(atom_num)
                self.subgraph2idx[smi] = len(self.idx2subgraph)
                self.idx2subgraph.append(smi)
        self.method = method
            
    def generate(self, mol):
        smiles = mol_to_smi(mol)
        mol = smi_to_mol(smiles)

        if (self.method == 'jtvae') | (self.method=='relmole'):
            subgraphs, subgraph_smis = init_decomp(mol, self.vocab_dict, method=self.method)
            res = [(sub_smi, sub_g) for sub_g, sub_smi in zip(subgraphs, subgraph_smis)]
    
        else: 
            sub_mol = MolInSubgraph(smiles, self.kekulize, self.vocab_dict)
            
            while True:
                nei_smis = sub_mol.get_nei_smis()
                valid_candidates = [
                    (smi, self.vocab_dict[smi]) for smi in nei_smis if smi in self.vocab_dict
                ]
    
                if not valid_candidates:
                    break
    
                merge_smi, _ = max(valid_candidates, key=lambda x: x[1])
                sub_mol.merge(merge_smi) 
            res = sub_mol.get_smis_subgraphs()  

        # construct reversed index
        aid2pids = [[] for _ in range(mol.GetNumAtoms())]
        for pid, subgraph in enumerate(res):
            _, aids = subgraph
            for aid in aids:
                aid2pids[aid].append(pid)
   
        # # construct adjacent matrix
        # ad_mat = [[0 for _ in res] for _ in res]
        # for i in range(len(res)):
        #     for j in range(i + 1, len(res)):
        #         set_i = set(res[i][1])
        #         set_j = set(res[j][1])
        #         if set_i & set_j: # Share at least one atom
        #             ad_mat[i][j] = ad_mat[j][i] = 1
        # #group_idxs = [x[1] for x in res]
        return Molecule(smiles, res, self.kekulize)

    def idx_to_subgraph(self, idx):
        return self.idx2subgraph[idx]
    
    def subgraph_to_idx(self, subgraph):
        return self.subgraph2idx[subgraph]

    def num_subgraph_type(self):
        return len(self.idx2subgraph)

    def __call__(self, mol):
        return self.generate(mol)
    
    def __len__(self):
        return len(self.idx2subgraph)
