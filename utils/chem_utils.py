import numpy as np
from rdkit import Chem
from rdkit.Chem import Fragments

MAX_VALENCE = {'B': 3, 'Br':1, 'C':4, 'Cl':1, 'F':1, 'I':1, 'N':3, 'O':2, 'P':5, 'S':6, 'Se':4, 'Si':4}
    
def smi_to_mol(smiles: str, kekulize=False, sanitize=True):
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if mol is None:
        return None
    if kekulize:
        Chem.Kekulize(mol, True)
    return mol


def mol_to_smi(mol):
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)#, kekulieSmiles=True)

def get_clique_mol(mol, atoms):
    Chem.Kekulize(mol, clearAromaticFlags=True)
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=True)
    Chem.Kekulize(new_mol, clearAromaticFlags=True)
    return new_mol


organic_major_ish = {'C', 'O', 'N', 'F', 'Cl', 'Br', 'I', 'S', 'P', 'B', 'H'}
def get_submol(mol, atom_indices, kekulize=False):
    #Chem.Kekulize(mol, clearAromaticFlags=True)
    if len(atom_indices) == 1:
        atom_symbol = mol.GetAtomWithIdx(atom_indices[0]).GetSymbol()
        if atom_symbol == 'Si':
            atom_symbol = '[Si]'
        elif atom_symbol == 'Na':
            atom_symbol = '[Na+]'
        elif atom_symbol == 'H':
            atom_symbol = '[H]'
        elif atom_symbol not in organic_major_ish:
            atom_symbol = '[' + atom_symbol + ']'
        return smi_to_mol(atom_symbol, kekulize)
    aid_dict = { i: True for i in atom_indices }
    edge_indices = []
    for i in range(mol.GetNumBonds()):
        bond = mol.GetBondWithIdx(i)
        begin_aid = bond.GetBeginAtomIdx()
        end_aid = bond.GetEndAtomIdx()
        if begin_aid in aid_dict and end_aid in aid_dict:
            edge_indices.append(i)
    mol = Chem.PathToSubmol(mol, edge_indices)
    return mol

def get_submol_atom_map(mol, submol, group, kekulize=False):
    if len(group) == 1:
        return { group[0]: 0 }
    # special with N+ and N-
    for atom in submol.GetAtoms():
        if atom.GetSymbol() != 'N':
            continue
        if (atom.GetExplicitValence() == 3 and atom.GetFormalCharge() == 1) or atom.GetExplicitValence() < 3:
            atom.SetNumRadicalElectrons(0)
            atom.SetNumExplicitHs(2)
            
    matches = mol.GetSubstructMatches(submol)
    if len(matches) < 1:
        Chem.Kekulize(mol, clearAromaticFlags=True)
        Chem.Kekulize(submol, clearAromaticFlags=True)
        matches = mol.GetSubstructMatches(submol)
    old2new = { i: 0 for i in group}  # old atom idx to new atom idx
    found = False
    for m in matches:
        hit = True
        for i, atom_idx in enumerate(m):
            if atom_idx not in old2new:
                hit = False
                break
            old2new[atom_idx] = i
        if hit:
            found = True
            break
    assert found
    return old2new

def has_rdkit_fg(smi):
    mol = Chem.MolFromSmiles(smi, sanitize=True)
    fragment_funcs = [f for f in dir(Fragments) if f.startswith("fr_")]
    features = []
    for func_name in fragment_funcs:
        func = getattr(Fragments, func_name)  
        try:
            features.append(func(mol))
        except Exception as e:
            features.append(0)
    features = np.array(features, dtype=float)
    features[features != 0.] = 1.    
    return features

def has_atom_smarts(smi, atom_nums):
    features = []
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
    except: 
        mol = Chem.MolFromSmiles('[{}]'.format(smi))
    for num in atom_nums:
        smarts = f"[#{num}]"
        query = Chem.MolFromSmarts(smarts) 
        try:
            features.append(int(mol.HasSubstructMatch(query)))
        except:
            features.append(0)
    features = np.array(features, dtype=float)
    features[features != 0.] = 1.
    return features

def fg_featurizer(smi, atom_nums=None):  # atom_nums=[5, 6, 7, 8, 9, 16, 17, 35]
    rdkit_feat = has_rdkit_fg(smi)
    if atom_nums != None:
        atom_feat = has_atom_smarts(smi, atom_nums)
        return np.concatenate([rdkit_feat, atom_feat]).tolist()
    return rdkit_feat
