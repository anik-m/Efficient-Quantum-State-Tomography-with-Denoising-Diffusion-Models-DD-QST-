import argparse
import torch
import numpy as np
import hashlib
import sys
from itertools import product

# Qiskit Imports
import qiskit.qasm2 
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector

# Safe Import for Torino (Optional Real Hardware Noise)
try:
    from qiskit_ibm_runtime.fake_provider import FakeTorino
    HAS_TORINO = True
except ImportError:
    HAS_TORINO = False

def get_backend(noise_type):
    """Returns the requested backend (Ideal, Generic Noisy, or FakeTorino)."""
    if noise_type == 'torino':
        if HAS_TORINO:
            print("--- Loading FakeTorino (Real Hardware Snapshot) ---")
            return AerSimulator.from_backend(FakeTorino())
        else:
            print("WARNING: qiskit-ibm-runtime missing. Falling back to ideal simulator.")
            return AerSimulator()
    # Add generic noise models here if needed (depolarizing etc)
    return AerSimulator()

def get_circuit_hash(qc):
    """Unique fingerprint for deduplication (MD5 of QASM)."""
    qasm_str = qiskit.qasm2.export(qc)
    return hashlib.md5(qasm_str.encode('utf-8')).hexdigest()

def generate_master_dataset(n_samples, n_qubits, min_depth, max_depth, shots, noise_type, save_path):
    print(f"\n=== Generating Master RQC Dataset (N={n_qubits}) ===")
    print(f"Target: {n_samples} Unique Circuits | Depth: {min_depth}-{max_depth} | Noise: {noise_type}")

    backend = get_backend(noise_type)
    dataset = []
    seen_hashes = set()
    
    # Strategy Switch: Full Tomography vs Shadow Sampling
    # N <= 4: Full Tomography (Capture all 3^N correlations)
    # N >= 5: Shadow Sampling (Too many bases, just sample a subset)
    use_shadow = (n_qubits >= 5)
    bases_per_circuit = 100 if use_shadow else 3**n_qubits
    
    print(f"Strategy: {'Shadow Sampling' if use_shadow else 'Full Tomography'} ({bases_per_circuit} bases/circuit)")

    attempts = 0
    collisions = 0
    
    while len(dataset) < n_samples:
        attempts += 1
        
        # 1. Random Depth (Curriculum)
        depth = np.random.randint(min_depth, max_depth + 1)
        qc = random_circuit(n_qubits, depth, measure=False)
        
        # 2. Deduplication
        c_hash = get_circuit_hash(qc)
        if c_hash in seen_hashes:
            collisions += 1
            continue
        seen_hashes.add(c_hash)
        
        # 3. Ground Truth (Ideal State)
        clean_state = Statevector(qc)
        
        # 4. Generate Bases
        if use_shadow:
            # Randomly sample 'bases_per_circuit' strings (e.g., "XZY")
            bases = ["".join(np.random.choice(['X','Y','Z'], size=n_qubits)) for _ in range(bases_per_circuit)]
        else:
            # Generate ALL combinations
            bases = [''.join(p) for p in product(['X', 'Y', 'Z'], repeat=n_qubits)]
            
        # 5. Simulate Measurements
        measurements = []
        for basis_str in bases:
            meas_qc = qc.copy()
            
            # Apply Basis Rotation
            for q, b in enumerate(basis_str):
                if b == 'X': meas_qc.h(q)
                elif b == 'Y': meas_qc.sdg(q); meas_qc.h(q)
            # Z requires no gate
            
            meas_qc.measure_all()
            
            # Transpile & Run
            # Opt level 1 is fast; use 3 if you want realistic mapping
            t_qc = transpile(meas_qc, backend, optimization_level=1)
            result = backend.run(t_qc, shots=shots).result()
            
            # Save sparse counts
            measurements.append({
                'basis': basis_str,
                'counts': result.get_counts()
            })
            
        dataset.append({
            'id': len(dataset),
            'depth': depth,
            'hash': c_hash,
            'clean_state_vec': clean_state,
            'measurements': measurements
        })
        
        # Progress Log
        if len(dataset) % 100 == 0:
            sys.stdout.write(f"\rProgress: {len(dataset)}/{n_samples} | Collisions: {collisions}")
            sys.stdout.flush()

    print(f"\n--- Saving to {save_path} ---")
    torch.save(dataset, save_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--qubits', type=int, default=2)
    parser.add_argument('--min_depth', type=int, default=2)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--shots', type=int, default=1024)
    parser.add_argument('--noise', type=str, default='torino')
    parser.add_argument('--out', type=str, required=True)
    
    args = parser.parse_args()
    generate_master_dataset(
        args.samples, args.qubits, args.min_depth, args.max_depth, 
        args.shots, args.noise, args.out
    )
