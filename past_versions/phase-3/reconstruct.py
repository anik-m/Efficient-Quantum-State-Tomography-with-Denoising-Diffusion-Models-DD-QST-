import numpy as np
from itertools import product
from qiskit.quantum_info import DensityMatrix, partial_trace, entropy

def get_pauli_matrix(label):
    """
    Constructs the tensor product matrix for a Pauli string.
    CRITICAL FIX: Qiskit uses Little-Endian (qN...q0), but label is [q0...qN].
    We must reverse the label before applying np.kron to match Qiskit's statevector.
    """
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    mapping = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
    
    # REVERSE the label for Kronecker product to match Qiskit's Little-Endian
    # 'XZ' (X on q0, Z on q1) becomes Z (q1) kron X (q0)
    rev_label = label[::-1]
    
    mat = mapping[rev_label[0]]
    for char in rev_label[1:]:
        mat = np.kron(mat, mapping[char])
    return mat

def get_coefficient(pauli_str, synthetic_data):
    # This logic remains correct because dataset.py already aligned 
    # the bitstrings [q0, q1...] with the basis_str [q0, q1...]
    if all(c == 'I' for c in pauli_str):
        return 1.0 
        
    for basis_key, samples in synthetic_data.items():
        is_compatible = True
        for p_char, b_char in zip(pauli_str, basis_key):
            if p_char != 'I' and p_char != b_char:
                is_compatible = False
                break
        
        if is_compatible:
            # Convert 0->1, 1->-1
            vals = 1 - 2 * samples 
            relevant_indices = [i for i, char in enumerate(pauli_str) if char != 'I']
            relevant_cols = vals[:, relevant_indices]
            parities = np.prod(relevant_cols, axis=1)
            return np.mean(parities)
    return 0.0

def make_positive_semidefinite(rho):
    evals, evecs = np.linalg.eigh(rho)
    evals = np.maximum(evals, 0)
    if np.sum(evals) > 0:
        evals /= np.sum(evals)
    rho_psd = (evecs * evals) @ evecs.conj().T
    return DensityMatrix(rho_psd)

def linear_inversion(synthetic_data, num_qubits):
    dim = 2**num_qubits
    rho = np.zeros((dim, dim), dtype=complex)
    all_paulis = [''.join(p) for p in product(['I', 'X', 'Y', 'Z'], repeat=num_qubits)]
    
    for pauli_str in all_paulis:
        coeff = get_coefficient(pauli_str, synthetic_data)
        mat = get_pauli_matrix(pauli_str) # Now returns correct Endianness
        rho += coeff * mat
        
    rho /= dim
    return make_positive_semidefinite(rho)

def get_metrics(rho, num_qubits):
    purity = np.real(np.trace(rho.data @ rho.data))
    vn_entropy = entropy(rho)
    cut = num_qubits // 2
    traced_qubits = list(range(cut, num_qubits))
    reduced_rho = partial_trace(rho, traced_qubits)
    ent_entropy = entropy(reduced_rho)
    return purity, vn_entropy, ent_entropy
