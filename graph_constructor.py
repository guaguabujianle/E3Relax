# %%
import itertools
import numpy as np
import torch
from torch_geometric.data import Data

try:
    from pymatgen.io.ase import AseAtomsAdaptor
except Exception:
    pass

OFFSET_LIST = [
[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]
]

class AtomsToGraphs:
    """A class to help convert periodic atomic structures to graphs.
    """

    def __init__(
        self,
        radius=6,
        max_neigh=50
    ):
        self.radius = radius
        self.max_neigh = max_neigh

    def _get_neighbors_pymatgen(self, atoms):
        """Preforms nearest neighbor search and returns edge index, distances,
        and cell offsets"""
        struct = AseAtomsAdaptor.get_structure(atoms)
        _c_index, _n_index, _offsets, n_distance = struct.get_neighbor_list(
            r=self.radius, numerical_tol=0, exclude_self=True
        )

        _nonmax_idx = []
        for i in range(len(atoms)):
            idx_i = (_c_index == i).nonzero()[0]
            # sort neighbors by distance, remove edges larger than max_neighbors
            idx_sorted = np.argsort(n_distance[idx_i])[: self.max_neigh]
            _nonmax_idx.append(idx_i[idx_sorted])
        _nonmax_idx = np.concatenate(_nonmax_idx)

        _c_index = _c_index[_nonmax_idx]
        _n_index = _n_index[_nonmax_idx]
        n_distance = n_distance[_nonmax_idx]
        _offsets = _offsets[_nonmax_idx]

        edge_index = np.vstack((_n_index, _c_index))
        
        edge_index = torch.LongTensor(edge_index)
        cell_offsets = torch.LongTensor(_offsets)

        # remove self-connecting edges
        non_self_connect_mask = edge_index[0, :] != edge_index[1, :]
        edge_index = edge_index[:, non_self_connect_mask]
        cell_offsets = cell_offsets[non_self_connect_mask]

        # add six types of self-connecting edges
        self_connect_edge_index = torch.arange(len(atoms)).repeat_interleave(len(OFFSET_LIST), dim=0)
        self_connect_edge_index = torch.stack([self_connect_edge_index, self_connect_edge_index])
        self_connect_cell_offsets = torch.FloatTensor(OFFSET_LIST * len(atoms))

        edge_index = torch.cat([edge_index, self_connect_edge_index], dim=-1)
        cell_offsets = torch.cat([cell_offsets, self_connect_cell_offsets], dim=0)

        return edge_index, cell_offsets

    def unwrap_cartesian_positions(self, coords_u, coords_r, cell_r):
        supercells = torch.FloatTensor(list(itertools.product((-1, 0, 1), repeat=3)))
        super_coords_r = coords_r.unsqueeze(1) + (supercells @ cell_r).unsqueeze(0)
        dists = torch.cdist(coords_u.unsqueeze(1), super_coords_r)
        image = dists.argmin(dim=-1).squeeze()
        cell_offsets = supercells[image]
        coords_r = coords_r + cell_offsets @ cell_r

        return coords_r
    
    def convert_single(
        self,
        atoms_u
    ):
        positions_u = torch.Tensor(atoms_u.get_positions())
        cell_u = torch.Tensor(atoms_u.get_cell())
        edge_index, cell_offsets = self._get_neighbors_pymatgen(atoms_u)
        neighbors = edge_index.size(-1)

        atomic_numbers = torch.Tensor(atoms_u.get_atomic_numbers())
        natoms = positions_u.shape[0]
        pbc = torch.tensor(atoms_u.pbc)

        data = Data(
            cell_u=cell_u.view(1, 3, 3),
            pos_u=positions_u,

            x=atomic_numbers,
            cell_offsets=cell_offsets,
            edge_index=edge_index,
            natoms=natoms,
            pbc=pbc,
            neighbors=neighbors,
        )     

        return data

    def convert_pairs(
        self,
        atoms_u,
        atoms_r
    ):
        """
        Convert a pair of atomic stuctures to a graph.
        """
        positions_u = torch.Tensor(atoms_u.get_positions())
        cell_u = torch.Tensor(atoms_u.get_cell())
        edge_index, cell_offsets = self._get_neighbors_pymatgen(atoms_u)

        positions_r = torch.Tensor(atoms_r.get_positions())
        cell_r = torch.Tensor(atoms_r.get_cell())
        unwrapped_positions_r = self.unwrap_cartesian_positions(positions_u, positions_r, cell_r)
        atoms_r.set_positions(unwrapped_positions_r)
        positions_r = torch.Tensor(atoms_r.get_positions()) # update position

        atomic_numbers = torch.Tensor(atoms_u.get_atomic_numbers())
        natoms = positions_u.shape[0]
        pbc = torch.tensor(atoms_u.pbc)

        data = Data(
            cell_u=cell_u.view(1, 3, 3),
            pos_u=positions_u,

            cell_r=cell_r.view(1, 3, 3),
            pos_r=positions_r,

            x=atomic_numbers,
            cell_offsets=cell_offsets,
            edge_index=edge_index,
            natoms=natoms,
            pbc=pbc,
        )

        return data

# %%
# import os
# from ase.io import read

# data_root = '/scratch/yangzd/materials/project/Cryslator/cifs_xmno'
# data_id = 'mp-754231_pox_lu_ba'
# data_id = 'mp-778013_pox_f_mg'
# data_id = 'mp-9600_pox_cu_ba'

# unrelaxed_path = os.path.join(data_root, data_id+'_unrelaxed.cif')
# relaxed_path = os.path.join(data_root, data_id+'_relaxed.cif')

# atoms_u = read(unrelaxed_path)
# atoms_r = read(relaxed_path)

# a2g = AtomsToGraphs(
#     radius=6,
#     max_neigh=50,
    
# )
# data = a2g.convert_pairs(atoms_u, atoms_r)
# print("edge_index: ", data.edge_index.shape)
# print("cell_offsets: ", data.cell_offsets.shape)
# print(data.lc_u)
# print(data.lc_r)
# print(atoms_u.get_atomic_numbers())
# print(atoms_r.get_atomic_numbers())

# print("cell_u: ", data.cell_u)
# print("cell_r: ", data.cell_r)
# print("pos_u: ", data.pos_u)
# print("pos_r: ", data.pos_r)
# print("edge_index: ", data.edge_index)
# print("pbc: ", data.pbc)
# print("edge_index: ", data.edge_index.shape)
# print("cell_offsets: ", data.cell_offsets.shape)

# print(len(data.x))

# %%
