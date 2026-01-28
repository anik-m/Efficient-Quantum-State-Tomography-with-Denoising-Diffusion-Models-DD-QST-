import torch
from torch.utils.data import Dataset
from itertools import product
import glob
import os

class QuantumStateDataset(Dataset):
    def __init__(self, data_input, num_qubits):
        """
        Args:
            data_input: Can be ONE of:
                        1. A list of circuit dictionaries (already loaded).
                        2. A path to a directory containing .pt files.
                        3. A path to a single .pt file.
            num_qubits: 2, 3, or 4.
        """
        self.samples = []
        raw_data_list = []

        # --- 1. Universal Loading Logic ---
        if isinstance(data_input, list):
            # Case A: We passed a list of data directly (from main.py split)
            raw_data_list = data_input
        elif isinstance(data_input, str):
            if os.path.isdir(data_input):
                # Case B: Directory of parts
                files = sorted(glob.glob(os.path.join(data_input, "*.pt")))
                print(f"Dataset: Loading {len(files)} files from directory...")
                for f in files:
                    try:
                        raw_data_list.extend(torch.load(f))
                    except Exception as e:
                        print(f"Skipping corrupt file {f}: {e}")
            elif os.path.isfile(data_input):
                # Case C: Single .pt file
                print(f"Dataset: Loading single file {data_input}...")
                raw_data_list = torch.load(data_input)
            else:
                raise FileNotFoundError(f"Path not found: {data_input}")

        # --- 2. Basis Mapping ---
        all_bases = [''.join(p) for p in product(['X', 'Y', 'Z'], repeat=num_qubits)]
        self.basis_to_idx = {b: i for i, b in enumerate(all_bases)}
        
        # --- 3. Flatten Data (Circuit -> Shots) ---
        # This converts the list of circuits into a massive list of training examples
        for circuit in raw_data_list:
            # We assume circuit is a dict with 'measurements'
            for meas in circuit.get('measurements', []):
                basis_str = meas['basis']
                if basis_str not in self.basis_to_idx: continue
                basis_idx = self.basis_to_idx[basis_str]
                
                for bitstr, count in meas['counts'].items():
                    # CRITICAL FIX: Reverse bits for PyTorch (Big-Endian)
                    # Qiskit '01' (q1=0, q0=1) -> [1, 0]
                    bits = [int(b) for b in bitstr][::-1]
                    
                    # Add this sample 'count' times
                    self.samples.extend([(bits, basis_idx)] * count)

        # --- 4. Tensor Conversion ---
        if len(self.samples) > 0:
            self.data_tensor = torch.tensor([s[0] for s in self.samples], dtype=torch.long)
            self.basis_tensor = torch.tensor([s[1] for s in self.samples], dtype=torch.long)
        else:
            print("WARNING: Dataset is empty.")
            self.data_tensor = torch.empty(0)
            self.basis_tensor = torch.empty(0)

        print(f"Dataset Ready: {len(self.samples)} training shots.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.data_tensor[idx], self.basis_tensor[idx]
