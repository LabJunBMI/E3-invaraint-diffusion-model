import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import random


RANDOM_SEED = 0
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
SS_VOCAB = "HBEGITS-"

class LigandBindingSiteDataset(Dataset):
    feature_names = list(AA_VOCAB)
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
        ligand_seq = data["amino_acid"][ligand_mask]
        # create extend mask by 1
        pocket_shit_left = torch.roll(data["pocket_mask"], self.pocket_ext)
        pocket_shit_left[0] = False
        pocket_shit_right= torch.roll(data["pocket_mask"], -self.pocket_ext)
        pocket_shit_right[-1] = False
        pocket_mask = data["pocket_mask"] | pocket_shit_left | pocket_shit_right
        pocket_angles = data["angle_features"][pocket_mask]
        pocket_seq = data["amino_acid"][pocket_mask]
        
        ligand_attn_mask = torch.zeros(size=(self.max_len,))
        ligand_attn_mask[:ligand_mask.sum()] = 1.0
        
        pocket_attn_mask = torch.zeros(size=(self.max_len,))
        pocket_attn_mask[:pocket_mask.sum()] = 1.0
        return {
            "ligand_angles": self._pad(ligand_angles),
            "ligand_attn_mask": ligand_attn_mask,
            "ligand_seq":self._pad(ligand_seq),
            "ligand_pos_id": 0,

            "receptor_angles":self._pad(pocket_angles),
            "receptor_attn_mask": pocket_attn_mask,
            "receptor_seq":self._pad(pocket_seq),
            "receptor_pos_id": 0,

            "ligand_length": ligand_mask.sum(),
            "receptor_length": pocket_mask.sum(),
            "structure_ids":data["structure_ids"]
        }