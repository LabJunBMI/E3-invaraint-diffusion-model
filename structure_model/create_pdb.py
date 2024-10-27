import os
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from functools import cached_property
from dataset import LigandBindingSiteDataset

import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile

from typing import (
    List,
    Union,
    Tuple,
    Optional,
    Sequence,
)

import torch


N_CA_LENGTH = 1.46  # Check, approxiamtely right
CA_C_LENGTH = 1.54  # Check, approximately right
C_N_LENGTH = 1.34   # Check, approximately right
C_O_LENGTH = 1.22   # Check, approximately right

# Taken from initial coords from 1CRN, which is a THR
N_INIT = np.array([17.047, 14.099, 3.625])
CA_INIT = np.array([16.967, 12.784, 4.338])
C_INIT = np.array([15.685, 12.755, 5.133])
O_INIT = np.array([15.268,  13.825,   5.594])

TRUE_DATA = "./data/biolip.pt"
GENERATED_DATA = "./data/output.pkl"
OUTPUT_FODLER = "./data/output"

COLS=["phi", "psi", "omega", "dihedral_o", "tau", "CA:C:1N", "1C:N:CA", "CA:C:O"]

class NERFBuilder:
    """
    Builder for NERF
    """

    def __init__(
        self,
        phi_dihedrals: np.ndarray,
        psi_dihedrals: np.ndarray,
        omega_dihedrals: np.ndarray,
        oxygen_dihedrals: np.ndarray,
        bond_len_n_ca: Union[float, np.ndarray] = N_CA_LENGTH,
        bond_len_ca_c: Union[float, np.ndarray] = CA_C_LENGTH,
        bond_len_c_n: Union[float, np.ndarray] = C_N_LENGTH,  # 0C:1N distance
        bond_len_c_o: Union[float, np.ndarray] = C_O_LENGTH,  # 0C:1N distance
        bond_angle_n_ca: Union[float, np.ndarray] = 121 / 180 * np.pi,
        bond_angle_ca_c: Union[float, np.ndarray] = 109 / 180 * np.pi,  # aka tau
        bond_angle_c_n: Union[float, np.ndarray] = 115 / 180 * np.pi,
        bond_angle_c_o: Union[float, np.ndarray] = 115 / 180 * np.pi,
        init_coords: np.ndarray = [N_INIT, CA_INIT, C_INIT],
    ) -> None:
        self.use_torch = False
        if any(
            [
                isinstance(v, torch.Tensor)
                for v in [phi_dihedrals, psi_dihedrals, omega_dihedrals, oxygen_dihedrals]
            ]
        ):
            self.use_torch = True

        self.phi = phi_dihedrals.squeeze()
        self.psi = psi_dihedrals.squeeze()
        self.omega = omega_dihedrals.squeeze()

        # We start with coordinates for N --> CA --> C so the next atom we add
        # is the next N. Therefore, the first angle we need is the C --> N bond
        self.bond_lengths = {
            ("C", "N"): bond_len_c_n,
            ("N", "CA"): bond_len_n_ca,
            ("CA", "C"): bond_len_ca_c,
        }
        self.bond_angles = {
            ("C", "N"): bond_angle_c_n,
            ("N", "CA"): bond_angle_n_ca,
            ("CA", "C"): bond_angle_ca_c,
        }
        self.init_coords = [c.squeeze() for c in init_coords]
        assert (
            len(self.init_coords) == 3
        ), f"Requires 3 initial coords for N-Ca-C but got {len(self.init_coords)}"
        assert all(
            [c.size == 3 for c in self.init_coords]
        ), "Initial coords should be 3-dimensional"

        
        # Handle Oxygen atoms separatly
        self.o_dihedral = oxygen_dihedrals.squeeze()
        self.bond_len_c_o = bond_len_c_o
        self.bond_angle_c_o = bond_angle_c_o
    @cached_property
    def cartesian_coords(self) -> Union[np.ndarray, torch.Tensor]:
        """Build out the molecule"""
        bb_coords = self.init_coords.copy()
        if self.use_torch:
            bb_coords = [torch.tensor(x, requires_grad=True) for x in bb_coords]

        # The first value of phi at the N terminus is not defined
        # The last value of psi and omega at the C terminus are not defined
        phi = self.phi[1:]
        psi = self.psi[:-1]
        omega = self.omega[:-1]
        dih_angles = (
            torch.stack([psi, omega, phi])
            if self.use_torch
            else np.stack([psi, omega, phi])
        ).T
        assert (
            dih_angles.shape[1] == 3
        ), f"Unexpected dih_angles shape: {dih_angles.shape}"

        for i in range(dih_angles.shape[0]):
            # for i, (phi, psi, omega) in enumerate(
            #     zip(self.phi[1:], self.psi[:-1], self.omega[:-1])
            # ):
            dih = dih_angles[i]
            # Procedure for placing N-CA-C
            # Place the next N atom, which requires the C-N bond length/angle, and the psi dihedral
            # Place the alpha carbon, which requires the N-CA bond length/angle, and the omega dihedral
            # Place the carbon, which requires the the CA-C bond length/angle, and the phi dihedral
            for j, bond in enumerate(self.bond_lengths.keys()):
                coords = place_dihedral(
                    bb_coords[-3],
                    bb_coords[-2],
                    bb_coords[-1],
                    bond_angle=self._get_bond_angle(bond, i),
                    bond_length=self._get_bond_length(bond, i),
                    torsion_angle=dih[j],
                    use_torch=self.use_torch,
                )
                bb_coords.append(coords)
        result = []
        for i, (n,ca,c) in enumerate([bb_coords[i:i + 3] for i in range(0, len(bb_coords), 3)]):
            o = place_dihedral(
                n,ca,c,
                self.bond_angle_c_o[i],
                self.bond_len_c_o,
                self.o_dihedral[i],
                use_torch=self.use_torch,
            )
            result.extend([n,ca,c,o]) # ~= result += [n,ca,c,o]
        if self.use_torch:
            return torch.stack(result)
        return np.array(result)

    @cached_property
    def centered_cartesian_coords(self) -> Union[np.ndarray, torch.Tensor]:
        """Returns the centered coords"""
        means = self.cartesian_coords.mean(axis=0)
        return self.cartesian_coords - means

    def _get_bond_length(self, bond: Tuple[str, str], idx: int):
        """Get the ith bond distance"""
        v = self.bond_lengths[bond]
        if isinstance(v, float):
            return v
        return v[idx]

    def _get_bond_angle(self, bond: Tuple[str, str], idx: int):
        """Get the ith bond angle"""
        v = self.bond_angles[bond]
        if isinstance(v, float):
            return v
        return v[idx]


def place_dihedral(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    bond_angle: float,
    bond_length: float,
    torsion_angle: float,
    use_torch: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Place the point d such that the bond angle, length, and torsion angle are satisfied
    with the series a, b, c, d.
    """
    assert a.shape == b.shape == c.shape
    assert a.shape[-1] == b.shape[-1] == c.shape[-1] == 3

    if not use_torch:
        unit_vec = lambda x: x / np.linalg.norm(x, axis=-1)
        cross = lambda x, y: np.cross(x, y, axis=-1)
    else:
        ensure_tensor = (
            lambda x: torch.tensor(x, requires_grad=False).to(a.device)
            if not isinstance(x, torch.Tensor)
            else x.to(a.device)
        )
        a, b, c, bond_angle, bond_length, torsion_angle = [
            ensure_tensor(x) for x in (a, b, c, bond_angle, bond_length, torsion_angle)
        ]
        unit_vec = lambda x: x / torch.linalg.norm(x, dim=-1, keepdim=True)
        cross = lambda x, y: torch.linalg.cross(x, y, dim=-1)

    ab = b - a
    bc = unit_vec(c - b)
    n = unit_vec(cross(ab, bc))
    nbc = cross(n, bc)

    if not use_torch:
        m = np.stack([bc, nbc, n], axis=-1)
        d = np.stack(
            [
                -bond_length * np.cos(bond_angle),
                bond_length * np.cos(torsion_angle) * np.sin(bond_angle),
                bond_length * np.sin(torsion_angle) * np.sin(bond_angle),
            ],
            axis=a.ndim - 1,
        )
        d = m.dot(d)
    else:
        m = torch.stack([bc, nbc, n], dim=-1)
        d = torch.stack(
            [
                -bond_length * torch.cos(bond_angle),
                bond_length * torch.cos(torsion_angle) * torch.sin(bond_angle),
                bond_length * torch.sin(torsion_angle) * torch.sin(bond_angle),
            ],
            dim=a.ndim - 1,
        ).type(m.dtype)
        d = torch.matmul(m, d).squeeze()

    return d + c

def write_coords_to_pdb(coords: np.ndarray, out_fname: str) -> str:
    """
    Write the coordinates to the given pdb fname
    """
    # Create a new PDB file using biotite
    # https://www.biotite-python.org/tutorial/target/index.html#creating-structures
    assert len(coords) % 4 == 0, f"Expected 4N coords, got {len(coords)}"
    atoms = []
    for i, (n_coord, ca_coord, c_coord, o_coord) in enumerate(
        (coords[j : j + 4] for j in range(0, len(coords), 4))
    ):
        atom1 = struc.Atom(
            n_coord,
            chain_id="A",
            res_id=i + 1,
            atom_id=i * 4 + 1,
            res_name="GLY",
            atom_name="N",
            element="N",
            occupancy=1.0,
            hetero=False,
            b_factor=5.0,
        )
        atom2 = struc.Atom(
            ca_coord,
            chain_id="A",
            res_id=i + 1,
            atom_id=i * 4 + 2,
            res_name="GLY",
            atom_name="CA",
            element="C",
            occupancy=1.0,
            hetero=False,
            b_factor=5.0,
        )
        atom3 = struc.Atom(
            c_coord,
            chain_id="A",
            res_id=i + 1,
            atom_id=i * 4 + 3,
            res_name="GLY",
            atom_name="C",
            element="C",
            occupancy=1.0,
            hetero=False,
            b_factor=5.0,
        )
        atom4 = struc.Atom(
            o_coord,
            chain_id="A",
            res_id=i + 1,
            atom_id=i * 4 + 4,
            res_name="GLY",
            atom_name="O",
            element="O",
            occupancy=1.0,
            hetero=False,
            b_factor=5.0,
        )
        atoms.extend([atom1, atom2, atom3, atom4])
        # atoms.extend([atom1, atom2, atom3, atom4])
    full_structure = struc.array(atoms)

    # Add bonds
    full_structure.bonds = struc.BondList(full_structure.array_length())
    indices = list(range(full_structure.array_length()))
    # for a, b in zip(indices[:-1], indices[1:]):
    #     full_structure.bonds.add_bond(a, b, bond_type=struc.BondType.SINGLE)
    is_first = True
    prev_c = None
    for (n,ca,c,o) in [indices[i:i + 4] for i in range(0, len(indices), 4)]:
        if (not is_first):
            full_structure.bonds.add_bond(prev_c, n, bond_type=struc.BondType.SINGLE)# N->C
        full_structure.bonds.add_bond(n, ca, bond_type=struc.BondType.SINGLE)# N->Ca
        full_structure.bonds.add_bond(ca, c, bond_type=struc.BondType.SINGLE)# Ca->C
        full_structure.bonds.add_bond(c, o, bond_type=struc.BondType.DOUBLE)#  C->O
        prev_c = c
        is_first=False
    # Annotate secondary structure using CA coordinates
    # https://www.biotite-python.org/apidoc/biotite.structure.annotate_sse.html
    # https://academic.oup.com/bioinformatics/article/13/3/291/423201
    # a = alpha helix, b = beta sheet, c = coil
    # ss = struc.annotate_sse(full_structure, "A")
    # full_structure.set_annotation("secondary_structure_psea", ss)

    sink = PDBFile()
    sink.set_structure(full_structure)
    sink.write(out_fname)
    return out_fname

def create_new_chain_nerf(
    out_fname: str,
    dists_and_angles: pd.DataFrame,
    angles_to_set: Optional[List[str]] = None,
    dists_to_set: Optional[List[str]] = None,
    center_coords: bool = True,
) -> str:
    """
    Create a new chain using NERF to convert to cartesian coordinates. Returns
    the path to the newly create file if successful, empty string if fails.
    """
    if angles_to_set is None and dists_to_set is None:
        angles_to_set, dists_to_set = [], []
        for c in dists_and_angles.columns:
            # Distances are always specified using one : separating two atoms
            # Angles are defined either as : separating 3+ atoms, or as names
            if c.count(":") == 1:
                dists_to_set.append(c)
            else:
                angles_to_set.append(c)

    else:
        assert angles_to_set is not None
        assert dists_to_set is not None

    # Check that we are at least setting the dihedrals
    required_dihedrals = ["phi", "psi", "omega", "dihedral_o"]
    assert all([a in angles_to_set for a in required_dihedrals])

    nerf_build_kwargs = dict(
        phi_dihedrals=dists_and_angles["phi"],
        psi_dihedrals=dists_and_angles["psi"],
        omega_dihedrals=dists_and_angles["omega"],
        oxygen_dihedrals=dists_and_angles["dihedral_o"]
    )
    for a in angles_to_set:
        if a in required_dihedrals:
            continue
        assert a in dists_and_angles
        if a == "tau" or a == "N:CA:C":
            nerf_build_kwargs["bond_angle_ca_c"] = dists_and_angles[a]
        elif a == "CA:C:1N":
            nerf_build_kwargs["bond_angle_c_n"] = dists_and_angles[a]
        elif a == "1C:N:CA":
            nerf_build_kwargs["bond_angle_n_ca"] = dists_and_angles[a]
        elif a == "CA:C:O":
            nerf_build_kwargs["bond_angle_c_o"] = dists_and_angles[a]
        else:
            raise ValueError(f"Unrecognized angle: {a}")

    for d in dists_to_set:
        assert d in dists_and_angles.columns
        if d == "0C:1N":
            nerf_build_kwargs["bond_len_c_n"] = dists_and_angles[d]
        elif d == "N:CA":
            nerf_build_kwargs["bond_len_n_ca"] = dists_and_angles[d]
        elif d == "CA:C":
            nerf_build_kwargs["bond_len_ca_c"] = dists_and_angles[d]
        else:
            raise ValueError(f"Unrecognized distance: {d}")

    nerf_builder = NERFBuilder(**nerf_build_kwargs)
    coords = (
        nerf_builder.centered_cartesian_coords
        if center_coords
        else nerf_builder.cartesian_coords
    )
    if np.any(np.isnan(coords)):
        print(f"Found NaN values, not writing pdb file {out_fname}")
        return ""

    assert coords.shape == (
        int(dists_and_angles.shape[0] * 4),# number of atoms per residue
        3,# 3D coordinates
    ), f"Unexpected shape: {coords.shape} for input of {len(dists_and_angles)}"
    return write_coords_to_pdb(coords, out_fname)

def write_preds_pdb_folder(
    final_sampled: Sequence[pd.DataFrame],
    outdir: str,
    basename_prefix: str = "generated_",
) -> List[str]:
    """
    Write the predictions as pdb files in the given folder along with information regarding the
    tm_score for each prediction. Returns the list of files written.
    """
    os.makedirs(outdir, exist_ok=True)
    # Create the pairs of arguments
    files_written = []
    for i, samp in tqdm(enumerate(final_sampled), total=len(final_sampled)):
        f = create_new_chain_nerf(
            os.path.join(outdir, f"{basename_prefix}{i}.pdb"),
            samp
        )
        files_written.append(f)
    return files_written

def load_sampled_angle_seq():
    with open(GENERATED_DATA, "rb") as f:
        sampled_result = pickle.load(f)
    sampled_dfs = [pd.DataFrame(s, columns=COLS) for s in sampled_result[0]]
    return sampled_dfs

def load_sampled_angles():
    with open(GENERATED_DATA, "rb") as f:
        sampled_result = pickle.load(f)
    sampled_dfs = [pd.DataFrame(s, columns=COLS) for s in sampled_result]
    return sampled_dfs

def load_ground_truth_angles():
    graphs = LigandBindingSiteDataset(TRUE_DATA, "test", pocket_ext=0)
    angle_dfs = []
    for i in range(len(graphs)):
        g = graphs[i]
        angles = g["ligand_angles"][g["ligand_attn_mask"].bool()]
        angles = pd.DataFrame(angles, columns=COLS)
        angle_dfs.append(angles)
    return angle_dfs

if __name__ == "__main__":
    print("Loading Angles")
    sampled_dfs = load_sampled_angle_seq()
    # sampled_dfs = load_ground_truth_angles()
    print("Creating PDBs")
    write_preds_pdb_folder(sampled_dfs, OUTPUT_FODLER)
    # to calculate the error rate, 
    # d = peptide-generated
    # abs(modulo_with_wrapped_range(d).mean(axis=0)/np.pi*180)/360
