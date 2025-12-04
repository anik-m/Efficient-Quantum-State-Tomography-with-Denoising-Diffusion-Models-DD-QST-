import numpy as np
from itertools import product
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from config import NUM_QUBITS, SHOTS_TRAIN, STATE_TYPE

def get_basis_combinations(num_qubits):
    """Generates all 3^N Pauli basis strings (e.g., 'XX', 'ZY')."""
    bases = ['X', 'Y', 'Z', 'I']# remove II, III
    return [''.join(p) for p in product(bases, repeat=num_qubits)]

def create_circuit(state_type, num_qubits, basis_str):
    """Creates the state preparation + measurement rotation circuit."""
    qc = QuantumCircuit(num_qubits)

    # 1. State Preparation [cite: 139, 140]
    if state_type == 'plus':
        for i in range(num_qubits): qc.h(i)
    elif state_type == 'bell':
        qc.h(0)
        qc.cx(0, 1)
    elif state_type == 'ghz':
        qc.h(0)
        for i in range(num_qubits - 1): qc.cx(i, i+1)
    # Random State should also be considered

    # 2. Measurement Basis Rotation [cite: 71, 72]
    # Qiskit measures in Z by default. 
    # To measure X: Apply H. To measure Y: Apply Sdg + H.
    for i, basis in enumerate(basis_str):
        if basis == 'X':
            qc.h(i)
        elif basis == 'Y':
            qc.sdg(i)
            qc.h(i)
        # Z requires no gate
        
    qc.measure_all()
    return qc

def generate_synthetic_data():
    """Generates noisy training data for all bases."""
    print(f"--- Generating Data for {NUM_QUBITS}-Qubit {STATE_TYPE.upper()} State ---")
    
    # Use a noisy simulator (or Fake backend as per plan [cite: 133])
    # For simplicity in this file, we use a standard AerSimulator with noise
    # In a full run, insert: from qiskit_ibm_runtime.fake_provider import FakeTorino
    backend = AerSimulator() 
    # Noise models - fakevigov2 
    basis_combs = get_basis_combinations(NUM_QUBITS)
    dataset = []

    for basis_idx, basis_str in enumerate(basis_combs):
        qc = create_circuit(STATE_TYPE, NUM_QUBITS, basis_str)
        t_qc = transpile(qc, backend)
        job = backend.run(t_qc, shots=SHOTS_TRAIN)
        result = job.result()
        counts = result.get_counts()
        
        # Store data: (Basis Index, Counts Dictionary)
        dataset.append({
            'basis_str': basis_str,
            'basis_idx': basis_idx,
            'counts': counts
        })
        
    print(f"Generated data for {len(dataset)} distinct bases.")
    return dataset, basis_combs
