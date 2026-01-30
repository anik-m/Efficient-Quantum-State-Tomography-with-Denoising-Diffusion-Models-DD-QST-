import numpy as np
from itertools import product
from qiskit.quantum_info import DensityMatrix

def get_pauli_matrix(label):
    """Constructs the tensor product matrix for a Pauli string (e.g., 'XZ')."""
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    mapping = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
    
    mat = mapping[label[0]]
    for char in label[1:]:
        mat = np.kron(mat, mapping[char])
    return mat

def get_coefficient(pauli_str, synthetic_data):
    """
    Calculates expectation value <P> for a specific Pauli string (e.g., 'XI').
    Finds a compatible basis in the synthetic data to compute it.
    """
    if all(c == 'I' for c in pauli_str):
        return 1.0 # <II> is always 1
        
    # Find a basis in synthetic_data that measures the required Paulis
    # e.g., To find <XI>, we can use data from basis 'XX', 'XY', or 'XZ'.
    for basis_key, samples in synthetic_data.items():
        is_compatible = True
        for p_char, b_char in zip(pauli_str, basis_key):
            if p_char != 'I' and p_char != b_char:
                is_compatible = False
                break
        
        if is_compatible:
            # We found a valid basis (e.g., using 'XX' to find 'XI')
            # Convert bits 0->1, 1->-1
            vals = 1 - 2 * samples 
            
            # Only multiply columns where the Pauli operator is NOT Identity
            relevant_indices = [i for i, char in enumerate(pauli_str) if char != 'I']
            relevant_cols = vals[:, relevant_indices]
            
            # Calculate parity mean
            parities = np.prod(relevant_cols, axis=1)
            return np.mean(parities)
            
    # If no compatible basis found (should not happen with complete data)
    return 0.0

def make_positive_semidefinite(rho):
    """
    Projects a matrix onto the subspace of valid density matrices.
    1. PSD: eigenvalues >= 0
    2. Trace: sum(eigenvalues) = 1
    """
    # 1. Eigendecomposition
    evals, evecs = np.linalg.eigh(rho)
    
    # 2. Set negative eigenvalues to 0 (approximate projection)
    evals = np.maximum(evals, 0)
    
    # 3. Re-normalize trace to 1
    if np.sum(evals) > 0:
        evals /= np.sum(evals)
        
    # 4. Reconstruct
    rho_psd = (evecs * evals) @ evecs.conj().T
    
    return DensityMatrix(rho_psd)

def linear_inversion(synthetic_data, num_qubits):
    """
    Reconstructs Rho = 1/2^N * sum(<P> * P)
    Iterates over ALL 4^N Pauli strings, not just the measured ones.
    """
    dim = 2**num_qubits
    rho = np.zeros((dim, dim), dtype=complex)
    
    # Generate all 4^N Pauli strings (II, IX, IY, IZ, XI...)
    all_paulis = [''.join(p) for p in product(['I', 'X', 'Y', 'Z'], repeat=num_qubits)]
    
    for pauli_str in all_paulis:
        # 1. Get coefficient <P> from the data
        coeff = get_coefficient(pauli_str, synthetic_data)
        
        # 2. Get Matrix P
        mat = get_pauli_matrix(pauli_str)
        
        # 3. Add to sum
        rho += coeff * mat
        
    rho /= dim
    
    # 4. Clean up (PSD Projection) - Critical Fix for Qiskit Error
    return make_positive_semidefinite(rho)

