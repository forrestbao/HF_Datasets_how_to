from typing import List, Dict, Tuple, Literal
import torch 
from datasets import load_dataset, Dataset


from hhem2 import procssed_datasets_dir

curriculumn = {"abc":10, "xyz":20}

class HHEMDataset(torch.utils.data.IterableDataset):    
    def load_one_ds(self, ds_name, seed=42):
        # TODO: This is English only. 
        # Load the data from the disk
        jsonl = f"{procssed_datasets_dir}/{ds_name}_train.jsonl"
        ds = Dataset.from_json(jsonl)
        return ds 
    
    def load_curriculum_dss(self,
                        curriculum: Dict[str, int], 
                        seed: int = 42):
        """Load the datasets for the curriculum to RAM"""

        dss = []
        for ds_name, ds_size in curriculum.items():
            dss[ds_name] = self.load_one_ds(ds_name=ds_name, ds_size=ds_size, seed=seed)

        # concatenate the datasets
        dss = Dataset.concatenate(dss)
     
    def __init__(self, 
                 curriculum: Dict[str, int], # the curriculum to load
                 negativity= 0.5,
                 seed = 42):
        self.curriculum = curriculum
        self.negativity = negativity
        self.seed = seed
        self.dataset_index = 0 # the index of the dataset in the curriculum to load

        # buffer all data to RAM
        self.dss = self.load_curriculum_dss(curriculum=curriculum, seed=seed)
        
    def __len__(self):
        return len(self.dss)
    
    def __getitem__(self, idx):
        return self.dss[idx]