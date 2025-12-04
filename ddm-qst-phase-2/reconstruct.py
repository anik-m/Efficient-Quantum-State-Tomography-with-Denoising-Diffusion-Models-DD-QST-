
import numpy as np
from qiskit.quantum_info import DensityMatrix, state_fidelity, Statevector

def get_pauli_matrix(label):
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    mapping = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
    
    # Tensor product for multi-qubit Pauli
    # e.g., 'XZ' -> kron(X, Z)
    mat = mapping[label[0]]
    for char in label[1:]:
        mat = np.kron(mat, mapping[char])
    return mat

def calculate_expectation_value(bitstrings, basis_str):
    """
    Calculates <P> from synthetic bitstrings.
    Parity measurement: <P> = (Count_Even - Count_Odd) / Total
    """
    # Convert bitstrings (Numpy) to parity (+1 or -1)
    # If a bit is 1, it contributes -1 to Z measurement.
    # Total parity is product of (-1)^bit for measured qubits.
    
    # In Qiskit/Physics:
    # 0 -> eigenvalue +1
    # 1 -> eigenvalue -1
    
    # Map 0->1, 1->-1
    mapped_bits = 1 - 2 * bitstrings # (N_samples, N_qubits)
    
    # Multiply across qubits for this basis
    parities = np.prod(mapped_bits, axis=1)
    return np.mean(parities)

def linear_inversion(basis_data_map, num_qubits):
    """
    Reconstructs Rho = 1/2^N * sum(<P> * P) [cite: 172]
    """
    dim = 2**num_qubits
    rho = np.zeros((dim, dim), dtype=complex)
    
    for basis_str, samples in basis_data_map.items():
        # Get Expectation Value <P>
        exp_val = calculate_expectation_value(samples, basis_str)
        
        # Get Operator Matrix P
        pauli_op = get_pauli_matrix(basis_str)
        
        rho += exp_val * pauli_op
        
    return DensityMatrix(rho / dim)
