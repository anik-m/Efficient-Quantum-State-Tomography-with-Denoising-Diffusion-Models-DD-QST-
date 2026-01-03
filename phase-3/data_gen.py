import numpy as np
from itertools import product
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.circuit.random import random_circuit
from qiskit_aer.noise import (
    NoiseModel, 
    depolarizing_error, 
    thermal_relaxation_error,
    ReadoutError
)

def get_basis_combinations(num_qubits):
    """Generates all 3^N Pauli basis strings (e.g., 'XX', 'ZY')."""
    bases = ['X', 'Y', 'Z']
    return [''.join(p) for p in product(bases, repeat=num_qubits)]

def get_noise_model(noise_type, error_rate=0.01):
    """
    Returns a Qiskit NoiseModel based on the selected type.
    Options: 'ideal', 'readout', 'depolarizing', 'thermal'
    """
    noise_model = NoiseModel()
    
    if noise_type == 'ideal':
        return None
        
    elif noise_type == 'readout':
        # Symmetric bit-flip error on measurement
        p_error = error_rate
        ro_error = ReadoutError([[1 - p_error, p_error], [p_error, 1 - p_error]])
        noise_model.add_all_qubit_readout_error(ro_error)
        
    elif noise_type == 'depolarizing':
        # Gate errors (1-qubit and 2-qubit)
        # 1-qubit error
        e1 = depolarizing_error(error_rate, 1)
        noise_model.add_all_qubit_quantum_error(e1, ['h', 'x', 'sx', 'rz', 'id'])
        # 2-qubit error (usually 10x larger)
        e2 = depolarizing_error(error_rate * 10, 2)
        noise_model.add_all_qubit_quantum_error(e2, ['cx', 'cz'])
        
    elif noise_type == 'thermal':
        # T1/T2 relaxation errors
        # Modeled for a typical superconducting qubit
        t1 = 50e3  # 50 microseconds
        t2 = 70e3  # 70 microseconds
        gate_time_1q = 50   # 50 ns
        gate_time_2q = 300  # 300 ns
        
        e1 = thermal_relaxation_error(t1, t2, gate_time_1q)
        e2 = thermal_relaxation_error(t1, t2, gate_time_2q).tensor(
             thermal_relaxation_error(t1, t2, gate_time_2q))
             
        noise_model.add_all_qubit_quantum_error(e1, ['h', 'x', 'sx', 'id'])
        noise_model.add_all_qubit_quantum_error(e2, ['cx'])
        
    return noise_model

def create_circuit(state_type, num_qubits, basis_str, rqc_depth=4):
    """
    Creates the state preparation + measurement rotation circuit.
    Returns: (QuantumCircuit, Target_Statevector)
    """
    # 1. State Preparation
    if state_type == 'plus':
        qc = QuantumCircuit(num_qubits)
        for i in range(num_qubits): qc.h(i)
        target = Statevector(qc)
        
    elif state_type == 'bell':
        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        qc.cx(0, 1)
        target = Statevector(qc)
        
    elif state_type == 'ghz':
        qc = QuantumCircuit(num_qubits)
        qc.h(0)
        for i in range(num_qubits - 1): qc.cx(i, i+1)
        target = Statevector(qc)
        
    elif state_type == 'rqc':
        # Generate a Random Quantum Circuit
        # We perform NO measurement inside the RQC generation to get the clean state
        # seed is random per call
        qc = random_circuit(num_qubits, depth=rqc_depth, measure=False)
        target = Statevector(qc)
        
    # Copy for measurement so we don't modify the target state logic
    meas_qc = qc.copy()

    # 2. Measurement Basis Rotation
    for i, basis in enumerate(basis_str):
        if basis == 'X':
            meas_qc.h(i)
        elif basis == 'Y':
            meas_qc.sdg(i)
            meas_qc.h(i)
        # Z requires no gate
        
    meas_qc.measure_all()
    return meas_qc, target

def generate_synthetic_data(num_qubits, state_type, shots, noise_type='readout', rqc_depth=5):
    """Generates noisy training data."""
    print(f"--- Generating Data for {num_qubits}-Qubit {state_type.upper()} State ---")
    print(f"--- Noise Model: {noise_type.upper()} ---")
    
    noise_model = get_noise_model(noise_type)
    backend = AerSimulator(noise_model=noise_model)
    
    basis_combs = get_basis_combinations(num_qubits)
    dataset = []
    
    # IMPORTANT: For RQC, we generate ONE random circuit and measure it in all bases.
    # If we generated a new random circuit for every basis, we'd be doing tomography on noise.
    # We first create the base circuit:
    if state_type == 'rqc':
        base_qc, target_state = create_circuit('rqc', num_qubits, 'Z'*num_qubits, rqc_depth)
        # We only need the structure (gates) from base_qc, the rotations happen below
        print("Random Circuit Generated.")
    else:
        # For fixed states (Bell/GHZ), target is static
        _, target_state = create_circuit(state_type, num_qubits, 'Z'*num_qubits)

    for basis_idx, basis_str in enumerate(basis_combs):
        # We need to recreate/copy the circuit to apply different measurement rotations
        if state_type == 'rqc':
            # Remove the previous measurements from the base RQC to apply new rotations
            # Simplest way: Re-generate the rotation part on top of the base instruction
            qc = base_qc.copy()
            # Remove existing measurements (if any)
            qc.remove_final_measurements()
            
            # Apply Rotations
            for i, basis in enumerate(basis_str):
                if basis == 'X':
                    qc.h(i)
                elif basis == 'Y':
                    qc.sdg(i)
                    qc.h(i)
            qc.measure_all()
        else:
            qc, _ = create_circuit(state_type, num_qubits, basis_str)

        t_qc = transpile(qc, backend)
        job = backend.run(t_qc, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        dataset.append({
            'basis_str': basis_str,
            'basis_idx': basis_idx,
            'counts': counts
        })
        
    return dataset, basis_combs, target_state
