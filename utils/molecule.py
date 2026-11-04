# From https://github.com/THUNLP-MT/PS-VAE
from copy import copy, deepcopy

import networkx as nx
from .chem_utils import smi_to_mol, get_submol, get_submol_atom_map, get_clique_mol


class SubgraphNode:
    '''
    The node representing a subgraph
    '''
    def __init__(self, smiles: str, pos: int, atom_mapping: dict, kekulize: bool):
        self.smiles = smiles
        self.pos = pos
        self.mol = smi_to_mol(smiles, kekulize)#, sanitize=False)
        # map atom idx in the molecule to atom idx in the subgraph (submol)
        self.atom_mapping = copy(atom_mapping)
    
    def get_mol(self):
        '''return molecule in rdkit form'''
        return self.mol

    def get_atom_mapping(self):
        return copy(self.atom_mapping)

    def __str__(self):
        return f'''
                    smiles: {self.smiles},
                    position: {self.pos},
                    atom map: {self.atom_mapping}
                '''

class SubgraphEdge:
    '''
    Edges between two subgraphs
    '''
    def __init__(self, src: int, dst: int, edges: list):
        self.edges = copy(edges)  # list of tuple (a, b, type) where the canonical order is used
        self.src = src
        self.dst = dst
        self.bond_type = edges[-1]
        self.dummy = False
        if len(self.edges) == 0:
            self.dummy = True
    
    def get_edges(self):
        return copy(self.edges)
    
    def get_num_edges(self):
        return len(self.edges)

    def __str__(self):
        return f'''
                    src subgraph: {self.src}, dst subgraph: {self.dst},
                    atom bonds: {self.edges}
                '''


class Molecule(nx.Graph):
    '''molecule represented in subgraph-level'''

    def __init__(self, smiles: str=None, groups: list=None, kekulize: bool=False):
        super().__init__()
        if smiles is None:
            return
        self.graph['smiles'] = smiles
        rdkit_mol = smi_to_mol(smiles, kekulize)
        # processing atoms
        aid2pos = [[] for _ in range(rdkit_mol.GetNumAtoms())]
        for pos, group in enumerate(groups):
            subgraph_smi, subgraph = group
            subgraph_mol = get_clique_mol(rdkit_mol, subgraph)

            for aid in subgraph:
                aid2pos[aid].append(pos)
            try:
                self.atom_mapping = get_submol_atom_map(rdkit_mol, subgraph_mol, subgraph, kekulize)
            except:
                subgraph_mol = get_submol(rdkit_mol, subgraph)
                self.atom_mapping = get_submol_atom_map(rdkit_mol, subgraph_mol, subgraph, kekulize)
            
            if len(self.atom_mapping) < 1:
                print(smiles)
                raise ""
                
            node = SubgraphNode(subgraph_smi, pos, self.atom_mapping, kekulize)
            self.add_node(node)

        # process edges
        edges_arr = [[[] for _ in groups] for _ in groups]  # adjacent
        for edge_idx in range(rdkit_mol.GetNumBonds()):
            bond = rdkit_mol.GetBondWithIdx(edge_idx)
            begin = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()

            bond_type = bond.GetBondType()
            bond_dir = bond.GetBondDir()
            bond_conj = bond.GetIsConjugated()
            bond_ring = bond.IsInRing()

            for begin_subgraph_pos in aid2pos[begin]:
                for end_subgraph_pos in aid2pos[end]:
                    if begin_subgraph_pos == end_subgraph_pos:
                        continue
                    
                    # ONLY add edge info if the subgraphs share at least one atom
                    if not (set(groups[begin_subgraph_pos][1]) & set(groups[end_subgraph_pos][1])):
                        continue

                    begin_mapped = self.get_node(begin_subgraph_pos).atom_mapping[begin]
                    end_mapped = self.get_node(end_subgraph_pos).atom_mapping[end]

                    edges_arr[begin_subgraph_pos][end_subgraph_pos].append((begin_mapped, end_mapped, bond_type, bond_dir, bond_conj, bond_ring))
                    edges_arr[end_subgraph_pos][begin_subgraph_pos].append((end_mapped, begin_mapped, bond_type, bond_dir, bond_conj, bond_ring))

        # add edges into the graph
        for i in range(len(groups)):
            for j in range(len(groups)):
                if not i < j or len(edges_arr[i][j]) == 0:
                    continue
                edge = SubgraphEdge(i, j, edges_arr[i][j])
                self.add_edge(edge)
    
    @classmethod
    def from_nx_graph(cls, graph: nx.Graph, deepcopy=True):
        if deepcopy:
            graph = deepcopy(graph)
        graph.__class__ = Molecule
        return graph

    @classmethod
    def merge(cls, mol0, mol1, edge=None):
        # reorder
        node_mappings = [{}, {}]
        mols = [mol0, mol1]
        mol = Molecule.from_nx_graph(nx.Graph())
        for i in range(2):
            for n in mols[i].nodes:
                node_mappings[i][n] = len(node_mappings[i])
                node = deepcopy(mols[i].get_node(n))
                node.pos = node_mappings[i][n]
                mol.add_node(node)
            for src, dst in mols[i].edges:
                edge = deepcopy(mols[i].get_edge(src, dst))
                edge.src = node_mappings[i][src]
                edge.dst = node_mappings[i][dst]
                mol.add_edge(src, dst, connects=edge)
        # add new edge
        edge = deepcopy(edge)
        edge.src = node_mappings[0][edge.src]
        edge.dst = node_mappings[1][edge.dst]
        return mol

    def get_edge(self, i, j) -> SubgraphEdge:
        return self[i][j]['connects']
    
    def get_node(self, i) -> SubgraphNode:
        return self.nodes[i]['subgraph']

    def add_edge(self, edge: SubgraphEdge) -> None:
        src, dst = edge.src, edge.dst
        super().add_edge(src, dst, connects=edge)
    
    def add_node(self, node: SubgraphNode) -> None:
        n = node.pos
        super().add_node(n, subgraph=node)

    def subgraph(self, nodes: list):
        graph = super().subgraph(nodes)
        assert isinstance(graph, Molecule)
        return graph

    def __str__(self):
        desc = 'nodes: \n'
        for ni, node in enumerate(self.nodes):
            desc += f'{ni}:{self.get_node(node)}\n'
        desc += 'edges: \n'
        for src, dst in self.edges:
            desc += f'{src}-{dst}:{self.get_edge(src, dst)}\n'
        return desc
