import torch
from torch.utils.data import Dataset

class QuantumStateDataset(Dataset):
    """
    Unrolls Qiskit counts into bitstring tensors.
    HANDLES QISKIT ENDIANNESS: Reverses bitstrings so Index 0 == Qubit 0.
    """
    def __init__(self, raw_data_list, num_qubits):
        self.bitstrings = []
        self.basis_indices = []

        for entry in raw_data_list:
            basis_idx = entry['basis_idx']
            counts = entry['counts']
            
            for bit_str, count in counts.items():
                # Qiskit returns 'q2 q1 q0' (Little Endian).
                # We want list [q0, q1, q2] so it matches our Basis list.
                
                # 1. Convert string to list
                bits = [int(b) for b in bit_str] 
                
                # 2. FLIP IT REVERSE IT
                bits = bits[::-1] 
                
                # Add 'count' copies
                self.bitstrings.extend([bits] * count)
                self.basis_indices.extend([basis_idx] * count)

        self.bitstrings = torch.tensor(self.bitstrings, dtype=torch.long)
        self.basis_indices = torch.tensor(self.basis_indices, dtype=torch.long)
        
    def __len__(self):
        return len(self.bitstrings)

    def __getitem__(self, idx):
        return self.bitstrings[idx], self.basis_indices[idx]
