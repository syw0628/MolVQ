import csv
from pathlib import Path
import sys 
sys.path.append("../")

import tqdm
import pickle
from collections import Counter
from multiprocessing import Pool

from rdkit import Chem
from rdkit import RDLogger

BOND_FEATURES = ['BondType', 'Stereo', 'BondDir']

def get_bond_feature_name(bond):
    """
    Return the string format of bond features.
    Bond features are surrounded with ()

    """
    ret = []
    for bond_feature in BOND_FEATURES:
        fea = eval(f"bond.Get{bond_feature}")()
        ret.append(str(fea))

    return '(' + '-'.join(ret) + ')'

def atom_to_vocab(mol, atom):
    """
    Convert atom to vocabulary. The convention is based on atom type and bond type.
    :param mol: the molecular.
    :param atom: the target atom.
    :return: the generated atom vocabulary with its contexts.
    """
    nei = Counter()
    for a in atom.GetNeighbors():
        bond = mol.GetBondBetweenAtoms(atom.GetIdx(), a.GetIdx())
        nei[str(a.GetSymbol()) + "-" + str(bond.GetBondType())] += 1
    keys = nei.keys()
    keys = list(keys)
    keys.sort()
    output = atom.GetSymbol()
    for k in keys:
        output = "%s_%s%d" % (output, k, nei[k])
    return output

def bond_to_vocab(mol, bond):
    """
    Convert bond to vocabulary. The convention is based on atom type and bond type.
    Considering one-hop neighbor atoms
    :param mol: the molecular.
    :param atom: the target atom.
    :return: the generated bond vocabulary with its contexts.
    """
    nei = Counter()
    two_neighbors = (bond.GetBeginAtom(), bond.GetEndAtom())
    two_indices = [a.GetIdx() for a in two_neighbors]
    for nei_atom in two_neighbors:
        for a in nei_atom.GetNeighbors():
            a_idx = a.GetIdx()
            if a_idx in two_indices:
                continue
            tmp_bond = mol.GetBondBetweenAtoms(nei_atom.GetIdx(), a_idx)
            nei[str(nei_atom.GetSymbol()) + '-' + get_bond_feature_name(tmp_bond)] += 1
    keys = list(nei.keys())
    keys.sort()
    output = get_bond_feature_name(bond)
    for k in keys:
        output = "%s_%s%d" % (output, k, nei[k])
    return output


class TorchVocab(object):
    """
    Defines the vocabulary for atoms/bonds in molecular.
    """
    def __init__(self, counter, max_size=None, min_freq=1, specials=(), vocab_type='atom'): # specials=('<pad>', '<other>')
        """

        :param counter:
        :param max_size:
        :param min_freq:
        :param specials:
        :param vocab_type: 'atom': atom atom_vocab; 'bond': bond atom_vocab.
        """
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)
        if vocab_type in ('atom', 'bond'):
            self.vocab_type = vocab_type
        else:
            raise ValueError('Wrong input for vocab_type!')
        self.itos = list(specials)

        max_size = None if max_size is None else max_size + len(self.itos)
        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)
        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}


    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        # if self.vectors != other.vectors:
        #    return False
        return True

    def __len__(self):
        return len(self.itos)

    def vocab_rerank(self):
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1
                self.freqs[w] = 0
            self.freqs[w] += v.freqs[w]

    def mol_to_seq(self, mol, with_len=False):
        mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
        if self.vocab_type == 'atom':
            seq = [self.stoi.get(atom_to_vocab(mol, atom)) for i, atom in enumerate(mol.GetAtoms())]
        else:
            seq = [self.stoi.get(bond_to_vocab(mol, bond)) for i, bond in enumerate(mol.GetBonds())]
        return (seq, len(seq)) if with_len else seq

    @staticmethod
    def load_vocab(vocab_path: str) -> 'Vocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


class MolVocab(TorchVocab):
    def __init__(self, smiles, max_size=None, min_freq=1, vocab_type='atom'):
        if vocab_type in ('atom', 'bond'):
            self.vocab_type = vocab_type
        else:
            raise ValueError('Wrong input for vocab_type!')

        print("Building %s vocab from smiles: %d" % (self.vocab_type, len(smiles)))
        counter = Counter()

        for smi in tqdm.tqdm(smiles):
            mol = Chem.MolFromSmiles(smi)
            if self.vocab_type == 'atom':
                for _, atom in enumerate(mol.GetAtoms()):
                    v = atom_to_vocab(mol, atom)
                    counter[v] += 1
            else:
                for _, bond in enumerate(mol.GetBonds()):
                    v = bond_to_vocab(mol, bond)
                    counter[v] += 1
        super().__init__(counter, max_size=max_size, min_freq=min_freq, vocab_type=vocab_type)

    def __init__(self, file_path, max_size=None, min_freq=1, num_workers=1, total_lines=None, vocab_type='atom', smiles_column='st_smiles'):
        if vocab_type in ('atom', 'bond'):
            self.vocab_type = vocab_type
        else:
            raise ValueError('Wrong input for vocab_type!')
        print("Building %s vocab from file: %s" % (self.vocab_type, file_path))

        lg = RDLogger.logger()
        lg.setLevel(RDLogger.CRITICAL)

        with MolVocab.open_smiles_file(file_path) as source:
            smiles = MolVocab.iter_smiles(source, file_path, smiles_column)
            if total_lines is None:
                total_lines = sum(1 for _ in smiles)
            else:
                next(smiles, None)


        counter = Counter()
        pbar = tqdm.tqdm(total=total_lines)
        pool = Pool(num_workers)
        res = []
        batch = 50000
        for start in range(0, total_lines, batch):
            end = min(total_lines, start + batch)
            res.append(pool.apply_async(
                MolVocab.read_smiles_from_file,
                args=(file_path, start, end, vocab_type, smiles_column),
                callback=lambda _, size=end - start: pbar.update(size),
            ))
        pool.close()
        pool.join()
        for r in res:
            sub_counter = r.get()
            for k in sub_counter:
                if k not in counter:
                    counter[k] = 0
                counter[k] += sub_counter[k]
        super().__init__(counter, max_size=max_size, min_freq=min_freq, vocab_type=vocab_type)

    @staticmethod
    def open_smiles_file(file_path):
        if Path(file_path).suffix.lower() == '.csv':
            return open(file_path, encoding='utf-8-sig', newline='')
        return open(file_path)

    @staticmethod
    def iter_smiles(source, file_path, smiles_column):
        if Path(file_path).suffix.lower() == '.csv':
            reader = csv.DictReader(source)
            if smiles_column not in (reader.fieldnames or ()):
                raise ValueError(f"CSV column {smiles_column!r} not found in {file_path}")
            for row in reader:
                yield row[smiles_column]
        else:
            next(source, None)
            yield from source

    @staticmethod
    def read_smiles_from_file(file_path, start, end, vocab_type, smiles_column='st_smiles'):
        sub_counter = Counter()
        with MolVocab.open_smiles_file(file_path) as source:
            smiles = MolVocab.iter_smiles(source, file_path, smiles_column)
            for i, smi in enumerate(smiles):
                if i < start:
                    continue
                if i >= end:
                    break
                mol = Chem.MolFromSmiles(smi)
                if vocab_type == 'atom':
                    for atom in mol.GetAtoms():
                        v = atom_to_vocab(mol, atom)
                        sub_counter[v] += 1
                else:
                    for bond in mol.GetBonds():
                        v = bond_to_vocab(mol, bond)
                        sub_counter[v] += 1
        return sub_counter

    @staticmethod
    def load_vocab(vocab_path: str) -> 'MolVocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)