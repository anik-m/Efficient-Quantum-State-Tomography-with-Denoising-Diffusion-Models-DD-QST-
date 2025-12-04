import torch
from torch.utils.data import Dataset

class QuantumStateDataset(Dataset):
    """
    Unrolls Qiskit counts into bitstring tensors.
    Input: counts {'00': 45, '11': 55}
    Output Tensor: [[0,0], ... [1,1]]
    """
    def __init__(self, raw_data_list, num_qubits):
        self.bitstrings = []
        self.basis_indices = []

        for entry in raw_data_list:
            basis_idx = entry['basis_idx']
            counts = entry['counts']
            
            for bit_str, count in counts.items():
                # Convert string '01' to list [0, 1]
                # Note: Qiskit is Little-Endian, but for learning correlation
                # strict ordering matters less than consistency.
                bits = [int(b) for b in bit_str] 
                
                # Add 'count' copies of this bitstring
                self.bitstrings.extend([bits] * count)
                self.basis_indices.extend([basis_idx] * count)

        self.bitstrings = torch.tensor(self.bitstrings, dtype=torch.long)
        self.basis_indices = torch.tensor(self.basis_indices, dtype=torch.long)
        
    def __len__(self):
        return len(self.bitstrings)

    def __getitem__(self, idx):
        # Returns: (Bitstring [N], Basis_Index [1])
        return self.bitstrings[idx], self.basis_indices[idx]
