import torch
from torch.utils.data import Dataset
from itertools import product
import numpy as np

class PreGeneratedDataset(Dataset):
    def __init__(self, raw_data_list, num_qubits):
        """
        Args:
            raw_data_list: List of circuit dictionaries (from build_dataset.py).
            num_qubits: Used to generate the master basis map.
        """
        self.samples = []
        
        # 1. Create Basis Map (String -> Index)
        # We need a fixed index for every possible basis string for the embedding layer.
        # Note: This works well for N <= 5. For N > 5, we need a dynamic tokenizer.
        all_bases = [''.join(p) for p in product(['X', 'Y', 'Z'], repeat=num_qubits)]
        self.basis_to_idx = {b: i for i, b in enumerate(all_bases)}
        
        print(f"Processing {len(raw_data_list)} circuits for training...")
        
        # 2. Flatten Data
        for circuit in raw_data_list:
            for meas in circuit['measurements']:
                basis_str = meas['basis']
                if basis_str not in self.basis_to_idx:
                    continue # Should not happen if map is complete
                
                basis_idx = self.basis_to_idx[basis_str]
                
                for bitstr, count in meas['counts'].items():
                    # CRITICAL FIX: Endianness
                    # Qiskit (Little-Endian) -> Python List (Big-Endian)
                    # "01" (q1=0, q0=1) -> [0, 1] (q0, q1) if reversed properly
                    # For consistency with reconstruct.py, we reverse here.
                    bits = [int(b) for b in bitstr][::-1]
                    
                    # Add this sample 'count' times
                    self.samples.extend([(bits, basis_idx)] * count)

        # Convert to Tensors for speed
        # This might consume RAM for huge datasets. 
        # If >10GB data, use __getitem__ logic instead of pre-list.
        self.data_tensor = [torch.tensor(s[0], dtype=torch.long) for s in self.samples]
        self.basis_tensor = [torch.tensor(s[1], dtype=torch.long) for s in self.samples]
        
        print(f"-> Created {len(self.samples)} training samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.data_tensor[idx], self.basis_tensor[idx]

    def get_num_bases(self):
        return len(self.basis_to_idx)
