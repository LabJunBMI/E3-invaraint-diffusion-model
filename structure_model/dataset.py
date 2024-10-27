import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


from utils import modulo_with_wrapped_range, cosine_beta_schedule, compute_alphas
from typing import Dict, Optional
import numpy as np
import random


RANDOM_SEED = 0
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
SS_VOCAB = "HBEGITS-"

class LigandBindingSiteDataset(Dataset):
    feature_names = ["phi", "psi", "omega", "dihedral_o", "tau", "CA:C:1N", "1C:N:CA", "CA:C:O"]
    def __init__(
            self, 
            filepath:str,
            split:str,
            max_len:int=64,
            pocket_ext:int=1
    )->None:
        """Create dataset
        Args:
            split (str):
                String of train, test, validation
            filepath (str): 
                A filepath of pickle file of dataframe with columns:
                    ligand_angle,binding_site_sequence
            tokenizer (PreTrainedTokenizer): 
                A tokenizer for proteins
            ligand_min_len (int):
                minmun length for sequences
            max_len (int):
                Maximun length for sequences
        """
        super().__init__()
        self._load_file(filepath)
        self._split_data(split)
        self.max_len = max_len
        self.pocket_ext = pocket_ext
    
    def _pad(self, angles):
        if(angles.shape[0]>self.max_len):
            raise RuntimeError("Length exceed")
        angles = F.pad(
            angles,
            (0,0,0, self.max_len - angles.shape[0]),
            mode="constant", value=0
        )
        return angles

    def _one_hot_encode(self, sequence, vocab):
        indices = [vocab.index(char) for char in sequence]
        one_hot = F.one_hot(torch.tensor(indices), num_classes=len(vocab))
        return one_hot.float()

    def _split_data(self, split_name):
        random.seed(RANDOM_SEED)
        random.shuffle(self.data)
        if split_name is not None:
            split_idx = int(len(self.data) * 0.8)
            if split_name == "train":
                self.data = self.data[:split_idx]
            elif split_name == "validation":
                self.data = self.data[split_idx : split_idx + int(len(self.data) * 0.1)]
            elif split_name == "test":
                self.data = self.data[split_idx + int(len(self.data) * 0.1) :]
    
    def _load_file(self, filepath:str)->None:
        print(f"Loading data from {filepath}")
        self.data = torch.load(filepath)
        for d in self.data:
            d["amino_acid"] = self._one_hot_encode("".join(d["amino_acid"]), AA_VOCAB)
            d["secondary_structure"] = self._one_hot_encode("".join(d["secondary_structure"]), SS_VOCAB)
        
    def __len__(self)->int:
        return len(self.data)

    def get_structure_id(self, index):
        return self.data[index]["structure_ids"]
    
    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError("Index out of range")
        """
        edge_index[# edges],
        pocket_coors[#pocket_node, 3],
        coors[#nodes, 3],
        amino_acid[#nodes, 20], 
        secondary_structure[#node, 8],
        numerical_features[#node, 5], 
        angle_features[#node, 6],
        ligand_mask[#node],
        ligand_idx[#node],
        pocket_mask[#node],
        pocket_idx[#node],
        """
        data = self.data[index]
        ligand_mask = data["ligand_mask"]
        ligand_angles = data["angle_features"][ligand_mask]
        # create extend mask by 1
        pocket_shit_left = torch.roll(data["pocket_mask"], self.pocket_ext)
        pocket_shit_left[0] = False
        pocket_shit_right= torch.roll(data["pocket_mask"], -self.pocket_ext)
        pocket_shit_right[-1] = False
        pocket_mask = data["pocket_mask"] | pocket_shit_left | pocket_shit_right
        # pocket_index = torch.nonzero(pocket_mask).squeeze(dim=-1)
        pocket_angles = data["angle_features"][pocket_mask]
        pocket_seq = data["amino_acid"][pocket_mask]
        
        ligand_attn_mask = torch.zeros(size=(self.max_len,))
        ligand_attn_mask[:ligand_mask.sum()] = 1.0
        
        pocket_attn_mask = torch.zeros(size=(self.max_len,))
        pocket_attn_mask[:pocket_mask.sum()] = 1.0
        return {
            "ligand_angles": self._pad(ligand_angles),
            "ligand_attn_mask": ligand_attn_mask,
            "ligand_pos_id": 0,
            
            "receptor_angles":self._pad(pocket_angles),
            "receptor_attn_mask": pocket_attn_mask,
            "receptor_seq":self._pad(pocket_seq),
            "receptor_pos_id": 0,
            
            "ligand_length": ligand_mask.sum(),
            "receptor_length": pocket_mask.sum(),
            "structure_ids": data["structure_ids"]
        }

class NoisedAnglesDataset(Dataset):
    """
    class that produces noised outputs given a wrapped dataset.
    Wrapped dset should return a tensor from __getitem__ if dset_key
    is not specified; otherwise, returns a dictionary where the item
    to noise is under dset_key

    modulo can be given as either a float or a list of floats
    """
    def __init__(
        self,
        dset: Dataset,
        timesteps: int = 250,
    ) -> None:
        super().__init__()
        self.dset = dset
        self.n_features = len(dset.feature_names)
        self.angular_var_scale = 1.0
        self.timesteps = timesteps
        betas = cosine_beta_schedule(timesteps)
        self.alpha_beta_terms = compute_alphas(betas)
    @property
    def feature_names(self):
        """Pass through feature names property of wrapped dset"""
        return self.dset.feature_names

    @property
    def pad(self):
        """Pas through the pad property of wrapped dset"""
        return self.dset.pad
    def sample_length(self, *args, **kwargs):
        return self.dset.sample_length(*args, **kwargs)
    def __str__(self) -> str:
        return f"NoisedAnglesDataset wrapping {self.dset} with {len(self)} examples with cosine-{self.timesteps} with variance scales {self.angular_var_scale}"
    def __len__(self) -> int:
        return len(self.dset)
    def sample_noise(self, vals: torch.Tensor) -> torch.Tensor:
        """
        Adaptively sample noise based on modulo. We scale only the variance because
        we want the noise to remain zero centered
        """
        # Noise is always 0 centered
        noise = torch.randn_like(vals)
        # Shapes of vals couled be (batch, seq, feat) or (seq, feat)
        # Therefore we need to index into last dimension consistently
        # Scale by provided variance scales based on angular or not
        if self.angular_var_scale != 1.0:
            for j in range(noise.shape[-1]):  # Last dim = feature dim
                noise[..., j] *= self.angular_var_scale
        # Make sure that the noise doesn't run over the boundaries
        noise = modulo_with_wrapped_range(noise, -np.pi, np.pi)
        return noise

    def __getitem__(
        self, index: int,
        use_timestep: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        assert 0 <= index < len(self), f"Index {index} out of bounds for {len(self)}"
        # Handle cases where we exhaustively loop over t
        item = self.dset.__getitem__(index)
        ligand_angles = item["ligand_angles"]
        # Sample a random timepoint
        if use_timestep is not None:
            timestep = np.clip(np.array([use_timestep]), 0, self.timesteps - 1)
            timestep = torch.from_numpy(timestep).long()
        else:
            timestep = torch.randint(0, self.timesteps, (1,)).long()
        # Add noise
        noised_liangd_angle = self._add_noise_by_timestep(ligand_angles, timestep)
        retval = {
            "timestep": timestep,
            "known_noise": noised_liangd_angle["noise"],
            "noised_ligand_angle": noised_liangd_angle["noised_value"],
            "sqrt_alphas_cumprod_t": noised_liangd_angle["sqrt_alphas_cumprod_t"],
            "sqrt_one_minus_alphas_cumprod_t": noised_liangd_angle["sqrt_one_minus_alphas_cumprod_t"],
        }
        # Update dictionary
        item.update(retval)
        return item

    def _add_noise_by_timestep(self, v:torch.Tensor, timestep:torch.Tensor):
        sqrt_alphas_cumprod_t = self.alpha_beta_terms["sqrt_alphas_cumprod"][timestep.item()]
        sqrt_one_minus_alphas_cumprod_t = self.alpha_beta_terms[
            "sqrt_one_minus_alphas_cumprod"
        ][timestep.item()]
        noise = self.sample_noise(v)  # Vals passed in only for shape
        noised_value = (
            sqrt_alphas_cumprod_t * v + sqrt_one_minus_alphas_cumprod_t * noise
        )
        noised_value = modulo_with_wrapped_range(noised_value, -np.pi, np.pi)
        return {
            "noise":noise,
            "noised_value":noised_value,
            "sqrt_alphas_cumprod_t":sqrt_alphas_cumprod_t,
            "sqrt_one_minus_alphas_cumprod_t":sqrt_one_minus_alphas_cumprod_t
        }