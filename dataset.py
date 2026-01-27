import torch
from torch.utils.data import Dataset
from itertools import product
import glob
import os

class QuantumStateDataset(Dataset):
    def __init__(self, data_source, num_qubits):
        """
        Args:
            data_source: Directory containing part_*.pt files
            num_qubits: 2, 3, or 4
        """
        self.samples = []
        raw_data_list = []
        self.metadata = [] # Stores depth/type info for plotting later

        # 1. Robust Loading
        if os.path.isdir(data_source):
            files = sorted(glob.glob(os.path.join(data_source, "*.pt")))
            print(f"Loading {len(files)} parts from {data_source}...")
            for f in files:
                try:
                    part_data = torch.load(f)
                    raw_data_list.extend(part_data)
                except Exception as e:
                    print(f"Skipping corrupt file {f}: {e}")
        else:
            raw_data_list = torch.load(data_source)

        print(f"Total Circuits Loaded: {len(raw_data_list)}")

        # 2. Basis Mapping
        all_bases = [''.join(p) for p in product(['X', 'Y', 'Z'], repeat=num_qubits)]
        self.basis_to_idx = {b: i for i, b in enumerate(all_bases)}
        
        # 3. Flattening with Endianness Fix
        for circuit in raw_data_list:
            # Keep metadata for evaluation mapping
            meta = {
                'depth': circuit.get('depth', 0),
                'clean_state_vec': circuit.get('clean_state_vec'),
                'type': circuit.get('type', 'rqc') # e.g. 'rqc', 'ghz'
            }
            
            for meas in circuit['measurements']:
                basis_str = meas['basis']
                if basis_str not in self.basis_to_idx: continue
                basis_idx = self.basis_to_idx[basis_str]
                
                for bitstr, count in meas['counts'].items():
                    # CRITICAL: Qiskit returns 'qN...q0'. 
                    # We reverse to [q0, ... qN] for array indexing.
                    bits = [int(b) for b in bitstr][::-1]
                    
                    # Store tuple
                    self.samples.extend([(bits, basis_idx, meta)] * count)

        # Convert to Tensor (ignoring metadata for Tensor speed)
        self.data_tensor = torch.tensor([s[0] for s in self.samples], dtype=torch.long)
        self.basis_tensor = torch.tensor([s[1] for s in self.samples], dtype=torch.long)
        # We store metadata separately to avoid overhead in training loop
        self.meta_lookup = [s[2] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Training only needs Data + Basis. Metadata is for eval.
        return self.data_tensor[idx], self.basis_tensor[idx]

    def get_metadata(self, idx):
        """Helper for evaluate.py to retrieve depth/state info"""
        return self.meta_lookup[idx]
