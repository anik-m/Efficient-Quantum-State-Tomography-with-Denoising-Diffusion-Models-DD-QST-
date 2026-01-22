import torch
from torch.utils.data import Dataset
from itertools import product
import glob
import os

class PreGeneratedDataset(Dataset):
    def __init__(self, data_source, num_qubits):
        """
        Args:
            data_source: Can be a single .pt file path OR a directory containing part_*.pt files.
            num_qubits: Used to generate the master basis map.
        """
        self.samples = []
        
        # 1. Load Data (Handle File vs Directory)
        raw_data_list = []
        
        if os.path.isdir(data_source):
            # It's a directory (from batch_build.py)
            files = glob.glob(os.path.join(data_source, "*.pt"))
            files.sort() # Ensure consistent order
            print(f"Loading {len(files)} parts from {data_source}...")
            for f in files:
                part_data = torch.load(f)
                raw_data_list.extend(part_data)
        else:
            # It's a single file
            print(f"Loading single file: {data_source}")
            raw_data_list = torch.load(data_source)
            
        print(f"Total Circuits Loaded: {len(raw_data_list)}")

        # 2. Create Basis Map (String -> Index)
        all_bases = [''.join(p) for p in product(['X', 'Y', 'Z'], repeat=num_qubits)]
        self.basis_to_idx = {b: i for i, b in enumerate(all_bases)}
        
        # 3. Flatten Data for Training
        for circuit in raw_data_list:
            for meas in circuit['measurements']:
                basis_str = meas['basis']
                if basis_str not in self.basis_to_idx:
                    continue 
                
                basis_idx = self.basis_to_idx[basis_str]
                
                for bitstr, count in meas['counts'].items():
                    # FIX: Endianness reversal
                    bits = [int(b) for b in bitstr][::-1]
                    self.samples.extend([(bits, basis_idx)] * count)

        # Convert to Tensors (RAM heavy but fast)
        # If dataset > 10GB, you must move this to __getitem__
        self.data_tensor = torch.tensor([s[0] for s in self.samples], dtype=torch.long)
        self.basis_tensor = torch.tensor([s[1] for s in self.samples], dtype=torch.long)
        
        print(f"-> Created {len(self.samples)} training samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.data_tensor[idx], self.basis_tensor[idx]

    def get_num_bases(self):
        return len(self.basis_to_idx)
