from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys

### To construct original atom-level graph
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),
    'simple_atomic_num_list': [1, 5, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17, 19, 30, 33, 34, 35, 53, 'other'], # 19
    'possible_chirality_list':        [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],  # 4
    'possible_hybridization_list':    [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ],  # 7
    'possible_numH_list':             [0, 1, 2, 3, 4], # 1
    'possible_aromatic': [False, True],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list':           [0, 1, 2, 3, 4, 'other'],
    'possible_formal_charge': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5], #list(range(-5, 7)), # 1
    'possible_bonds':                 [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
        None
    ],
    'possible_bond_dirs':             [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ],
    'possible_is_conjugated': [False, True],
    'possible_is_in_ring': [False, True],
    'possible_motif_atom_num': list(range(1, 18)), # 0 ~ 17
    'possible_motif_ring_num': list(range(0, 6)), # 0 ~ 5
    'possible_motif_hetero_num': list(range(0, 8)), # 0 ~ 7
}

def one_hot_enc(x, allowable_set):
    if 'other' in allowable_set:
        return [float(x == s) for s in allowable_set[:-1]] + [float(x not in allowable_set[:-1])]
    else:
        return [float(x == s) for s in allowable_set]

# For Atom-level 
def atom_featurizer(atom, onehot=True):
    if onehot:
        atomic_num = one_hot_enc(atom.GetAtomicNum(), allowable_features['simple_atomic_num_list'])
        atom_feat = atomic_num + \
                    one_hot_enc(atom.GetTotalDegree(), allowable_features['possible_degree_list']) + \
                    one_hot_enc(atom.GetChiralTag(), allowable_features['possible_chirality_list']) + \
                    [allowable_features['possible_formal_charge'].index(atom.GetFormalCharge())
                     if atom.GetFormalCharge() in allowable_features['possible_formal_charge'] else len(allowable_features['possible_formal_charge'])] + \
                    one_hot_enc(atom.GetHybridization(), allowable_features['possible_hybridization_list']) + \
                    one_hot_enc(atom.GetTotalNumHs(), allowable_features['possible_numH_list']) + \
                    [allowable_features['possible_aromatic'].index(atom.GetIsAromatic())]
    else:
        atomic_num = [allowable_features['simple_atomic_num_list'].index(atom.GetAtomicNum())]
        atom_feat = atomic_num + \
                    [allowable_features['possible_degree_list'].index(atom.GetTotalDegree())] + \
                    [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())] + \
                    [allowable_features['possible_formal_charge'].index(atom.GetFormalCharge())] + \
                    [allowable_features['possible_hybridization_list'].index(atom.GetHybridization())] + \
                    [allowable_features['possible_numH_list'].index(atom.GetTotalNumHs())] + \
                    [allowable_features['possible_aromatic'].index(atom.GetIsAromatic())] 
    return atom_feat 

def bond_featurizer(bond, onehot=True):    
    if onehot:
        bond_feat = one_hot_enc(bond.GetBondType(), allowable_features['possible_bonds']) + \
                    one_hot_enc(bond.GetBondDir(), allowable_features['possible_bond_dirs']) + \
                    [allowable_features['possible_is_conjugated'].index(bond.GetIsConjugated())] + \
                    [allowable_features['possible_is_in_ring'].index(bond.IsInRing())]
    else:
        bond_feat = [allowable_features['possible_bonds'].index(bond.GetBondType())] + \
                    [allowable_features['possible_bond_dirs'].index(bond.GetBondDir())] + \
                    [allowable_features['possible_is_conjugated'].index(bond.GetIsConjugated())] + \
                    [allowable_features['possible_is_in_ring'].index(bond.IsInRing())]
    return bond_feat 
    
    
### For Motif-level
declist = Descriptors.descList
calc = {}
for (i,j) in declist:
    calc[i] = j

def get_motif_element(mol):
    atom_symbol={'C':0,'H':0,'O':0,'N':0,'P':0,
                 'S':0,'F':0,'CL':0,'Br':0,'other':0,}
    feat_element =[]
    # [C,H,O,N,P,S,F,CL,Br,other]
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in atom_symbol.keys():
            atom_symbol[atom.GetSymbol()]+=1
        else:
            atom_symbol['other']+=1
    for key,value in atom_symbol.items():
        feat_element+=[value]
    return feat_element

def maccskeys_emb(mol):
    return list(MACCSkeys.GenMACCSKeys(mol))

def motif_featurizer(motif_smi, vocab=False):
    motif = Chem.MolFromSmiles(motif_smi)
    motif_smi = Chem.MolToSmiles(motif)
    MOTIF_FEATURE_SIZE = 182
    if vocab:
        if motif_smi in vocab:
            motif_idx = vocab[motif_smi]
        else:
            motif_idx = len(vocab)            
        motif_idx = [motif_idx]
        MOTIF_FEATURE_SIZE += 1
    else:
        motif_idx = []
        
    try:
        motif_feat= motif_idx+\
                    [calc['TPSA'](motif)*0.01]+[calc['MolLogP'](motif)]+\
                    [calc['HeavyAtomMolWt'](motif)*0.01] +\
                    [1 if motif.GetRingInfo().NumRings()>0 else 0]+\
                    [motif.GetRingInfo().NumRings()]+\
                    get_motif_element(motif) +\
                    maccskeys_emb(motif)
    except:
        motif_feat = [0]*MOTIF_FEATURE_SIZE
    return motif_feat

def motif_bond_featurizer(features, onehot=True):
    if onehot:
        edge_feature = one_hot_enc(features[0], allowable_features['possible_bonds']) + \
                       one_hot_enc(features[1], allowable_features['possible_bond_dirs']) + \
                       [allowable_features['possible_is_conjugated'].index(features[2])] + \
                       [allowable_features['possible_is_in_ring'].index(features[3])]
    else:
        edge_feature = [allowable_features['possible_bonds'].index(features[0])] + \
                       [allowable_features['possible_bond_dirs'].index(features[1])] + \
                       [allowable_features['possible_is_conjugated'].index(features[2])] + \
                       [allowable_features['possible_is_in_ring'].index(features[3])]
    return edge_feature