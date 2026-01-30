import argparse
import torch
import numpy as np
import hashlib
import sys
import qiskit.qasm2 
from itertools import product
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector

# Safe Import for Torino
try:
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    HAS_TORINO = True
except ImportError:
    HAS_TORINO = False

def get_circuit_hash(qc):
    """Generates a deterministic hash of the circuit structure."""
    # We use qasm2.dumps because it captures the exact gates and parameters
    qasm_str = qiskit.qasm2.dumps(qc)
    return hashlib.md5(qasm_str.encode('utf-8')).hexdigest()

def generate_strict_dataset(n_samples, n_qubits, min_depth, max_depth, shots, noise_type, save_path):
    print(f"\n=== Strict Unique Dataset Generation (N={n_qubits}) ===")
    
    # --- PHASE 1: GENERATE & DEDUPLICATE (CPU Only, Very Fast) ---
    print("Phase 1: Generating unique circuit pool...")
    
    unique_circuits = {} # Map: hash -> circuit
    attempts = 0
    
    while len(unique_circuits) < n_samples:
        attempts += 1
        
        # 1. Generate Random Structure
        depth = np.random.randint(min_depth, max_depth + 1)
        qc = random_circuit(n_qubits, depth, measure=False)
        
        # 2. Strict Hash Check
        c_hash = get_circuit_hash(qc)
        
        if c_hash not in unique_circuits:
            # Store the circuit object for Phase 2
            unique_circuits[c_hash] = {'qc': qc, 'depth': depth, 'hash': c_hash}
            
        if attempts % 1000 == 0:
            print(f"  -> Pool Size: {len(unique_circuits)}/{n_samples} (scanned {attempts} candidates)")
            
        # Safety break if parameters make uniqueness impossible (unlikely for N=2)
        if attempts > n_samples * 50:
            raise RuntimeError(f"Could not find {n_samples} unique circuits. Try increasing depth or qubit count.")

    print(f"SUCCESS: Found {len(unique_circuits)} unique circuits.")
    
    # --- PHASE 2: SIMULATION (Expensive Part) ---
    print(f"Phase 2: Simulating {n_samples} circuits on {noise_type} backend...")
    
    # Setup Backend
    backend = AerSimulator()
    if noise_type == 'torino' and HAS_TORINO:
        backend = AerSimulator.from_backend(FakeTorino())

    dataset = []
    
    # Bases Setup
    use_shadow = (n_qubits >= 5)
    bases_per_circuit = 100 if use_shadow else 3**n_qubits
    if not use_shadow:
        all_bases = [''.join(p) for p in product(['X', 'Y', 'Z'], repeat=n_qubits)]

    # Convert dictionary to list for iteration
    circuit_pool = list(unique_circuits.values())

    for idx, item in enumerate(circuit_pool):
        qc = item['qc']
        
        # 1. Ground Truth
        clean_state = Statevector(qc)
        
        # 2. Determine Bases for this circuit
        if use_shadow:
            bases = ["".join(np.random.choice(['X','Y','Z'], size=n_qubits)) for _ in range(bases_per_circuit)]
        else:
            bases = all_bases

        # 3. Batch Simulation (Optimization)
        # We create all measurement circuits first, then transpile in batch
        circuits_to_run = []
        for basis_str in bases:
            meas_qc = qc.copy()
            for q, b in enumerate(basis_str):
                if b == 'X': meas_qc.h(q)
                elif b == 'Y': meas_qc.sdg(q); meas_qc.h(q)
            meas_qc.measure_all()
            circuits_to_run.append(meas_qc)
        
        # Batch Transpile (Optimization Level 1 is sufficient for random)
        t_qcs = transpile(circuits_to_run, backend, optimization_level=1)
        
        # Batch Run
        # backend.run can take a list of circuits!
        result = backend.run(t_qcs, shots=shots).result()
        
        # 4. Pack Data
        measurements = []
        for i, basis_str in enumerate(bases):
            measurements.append({
                'basis': basis_str,
                'counts': result.get_counts(i)
            })
            
        dataset.append({
            'id': idx,
            'hash': item['hash'],
            'depth': item['depth'],
            'clean_state_vec': clean_state,
            'measurements': measurements
        })
        
        if (idx+1) % 50 == 0:
            print(f"  -> Simulated {idx+1}/{n_samples}")

    # --- PHASE 3: SAVE ---
    print(f"--- Saving to {save_path} ---")
    torch.save(dataset, save_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=2000)
    parser.add_argument('--qubits', type=int, default=2)
    parser.add_argument('--min_depth', type=int, default=2)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--shots', type=int, default=1024)
    parser.add_argument('--noise', type=str, default='torino')
    parser.add_argument('--out', type=str, required=True)
    
    args = parser.parse_args()
    generate_strict_dataset(
        args.samples, args.qubits, args.min_depth, args.max_depth, 
        args.shots, args.noise, args.out
    )
