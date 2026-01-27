import numpy as np
from itertools import product
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector

# Note: No import from config here anymore!

def get_basis_combinations(num_qubits):
    """Generates all 3^N Pauli basis strings (e.g., 'XX', 'ZY')."""
    bases = ['X', 'Y', 'Z'] # Identity 'I' removed as discussed
    return [''.join(p) for p in product(bases, repeat=num_qubits)]

def create_circuit(state_type, num_qubits, basis_str):
    """Creates the state preparation + measurement rotation circuit."""
    qc = QuantumCircuit(num_qubits)

    # 1. State Preparation
    if state_type == 'plus':
        for i in range(num_qubits): qc.h(i)
        
    elif state_type == 'bell' or state_type == 'ghz':
        # Generalized GHZ/Bell creation: H on first, CNOT cascade
        qc.h(0)
        for i in range(num_qubits - 1):
            qc.cx(i, i+1)

    # 2. Measurement Basis Rotation
    for i, basis in enumerate(basis_str):
        if basis == 'X':
            qc.h(i)
        elif basis == 'Y':
            qc.sdg(i)
            qc.h(i)
        # Z requires no gate
        
    qc.measure_all()
    return qc

def generate_synthetic_data(num_qubits, state_type, shots_train):
    """Generates noisy training data for all bases based on inputs."""
    print(f"--- Generating Data for {num_qubits}-Qubit {state_type.upper()} State ---")
    
    backend = AerSimulator() 
    
    basis_combs = get_basis_combinations(num_qubits)
    dataset = []

    for basis_idx, basis_str in enumerate(basis_combs):
        qc = create_circuit(state_type, num_qubits, basis_str)
        t_qc = transpile(qc, backend)
        job = backend.run(t_qc, shots=shots_train)
        result = job.result()
        counts = result.get_counts()
        
        dataset.append({
            'basis_str': basis_str,
            'basis_idx': basis_idx,
            'counts': counts
        })
        
    print(f"Generated data for {len(dataset)} distinct bases.")
    return dataset, basis_combs
