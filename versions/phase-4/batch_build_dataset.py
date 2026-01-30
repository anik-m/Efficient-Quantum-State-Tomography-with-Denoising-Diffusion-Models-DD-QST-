import argparse
import torch
import numpy as np
import hashlib
import sys
import os
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
    return hashlib.md5(qiskit.qasm2.dumps(qc).encode('utf-8')).hexdigest()

def generate_batched_dataset(n_samples, n_qubits, min_depth, max_depth, shots, chunk_size, noise_type, save_dir):
    print(f"\n=== Batched Generation (N={n_qubits}) ===")
    print(f"Target: {n_samples} samples | Chunk Size: {chunk_size}")
    
    # 1. Setup Backend
    backend = AerSimulator()
    if noise_type == 'torino' and HAS_TORINO:
        backend = AerSimulator.from_backend(FakeTorino())

    # 2. Determine Basis Strategy
    # N=3 -> 27 bases (Do Full)
    # N=4 -> 81 bases (Do Shadow - limit to 50)
    MAX_BASES = 50 
    full_basis_count = 3**n_qubits
    
    if full_basis_count <= MAX_BASES:
        print(f"Strategy: Full Tomography ({full_basis_count} bases/circuit)")
        use_shadow = False
        all_bases = [''.join(p) for p in product(['X', 'Y', 'Z'], repeat=n_qubits)]
    else:
        print(f"Strategy: Shadow Sampling (Capped at {MAX_BASES} bases/circuit)")
        use_shadow = True

    # 3. Main Loop (Process in Chunks)
    total_generated = 0
    seen_hashes = set()
    
    os.makedirs(save_dir, exist_ok=True)
    
    while total_generated < n_samples:
        # --- Step A: Generate a Chunk of Unique Circuits ---
        chunk_circuits = []
        chunk_data = [] # To store metadata
        
        # Keep generating until we fill the chunk
        attempts = 0
        while len(chunk_circuits) < chunk_size and total_generated + len(chunk_circuits) < n_samples:
            attempts += 1
            depth = np.random.randint(min_depth, max_depth + 1)
            qc = random_circuit(n_qubits, depth, measure=False)
            c_hash = get_circuit_hash(qc)
            
            if c_hash not in seen_hashes:
                seen_hashes.add(c_hash)
                
                # Determine bases for this specific circuit
                if use_shadow:
                    bases = ["".join(np.random.choice(['X','Y','Z'], size=n_qubits)) for _ in range(MAX_BASES)]
                else:
                    bases = all_bases
                
                # Create Measurement Circuits
                meas_circs_for_this_qc = []
                for basis_str in bases:
                    m_qc = qc.copy()
                    for q, b in enumerate(basis_str):
                        if b == 'X': m_qc.h(q)
                        elif b == 'Y': m_qc.sdg(q); m_qc.h(q)
                    m_qc.measure_all()
                    meas_circs_for_this_qc.append(m_qc)
                
                # Store for bulk processing
                chunk_circuits.extend(meas_circs_for_this_qc)
                
                # Store metadata (we'll link results back later)
                chunk_data.append({
                    'qc': qc,
                    'depth': depth,
                    'hash': c_hash,
                    'bases': bases,
                    'num_meas': len(bases)
                })
                
            if attempts > chunk_size * 100:
                print("Warning: Difficulty finding unique circuits. Stopping early.")
                break
        
        if not chunk_data:
            break

        # --- Step B: Mega-Batch Simulation ---
        # Transpile & Run ALL circuits in the chunk at once (Massive Speedup)
        # E.g., 500 circuits * 27 bases = 13,500 circuits in one job
        print(f"  -> Simulating Chunk {total_generated // chunk_size + 1} ({len(chunk_circuits)} sub-circuits)...")
        
        # Optimization 0 is fast and fine for random circuits
        t_qcs = transpile(chunk_circuits, backend, optimization_level=0) 
        job = backend.run(t_qcs, shots=shots)
        result = job.result()
        
        # --- Step C: Unpack Results ---
        # We need to map the flat list of results back to the individual RQC entries
        result_cursor = 0
        final_chunk_dataset = []
        
        for item in chunk_data:
            clean_state = Statevector(item['qc'])
            num_meas = item['num_meas']
            
            measurements = []
            for i in range(num_meas):
                measurements.append({
                    'basis': item['bases'][i],
                    'counts': result.get_counts(result_cursor + i)
                })
            
            result_cursor += num_meas
            
            final_chunk_dataset.append({
                'id': total_generated,
                'hash': item['hash'],
                'depth': item['depth'],
                'clean_state_vec': clean_state,
                'measurements': measurements
            })
            total_generated += 1

        # --- Step D: Save Chunk ---
        chunk_filename = f"{save_dir}/part_{total_generated // chunk_size}.pt"
        torch.save(final_chunk_dataset, chunk_filename)
        print(f"  -> Saved {len(final_chunk_dataset)} samples to {chunk_filename}")

    # --- Step E: Merge (Optional) ---
    # We can leave them as parts, or you can use a separate script to merge.
    # For now, saving as parts is safer against timeouts.
    print("Done. Datasets saved in parts.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--qubits', type=int, default=3)
    parser.add_argument('--min_depth', type=int, default=2)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--shots', type=int, default=1024)
    parser.add_argument('--chunk_size', type=int, default=500)
    parser.add_argument('--noise', type=str, default='torino')
    parser.add_argument('--out_dir', type=str, default='dataset_parts')
    
    args = parser.parse_args()
    generate_batched_dataset(
        args.samples, args.qubits, args.min_depth, args.max_depth, 
        args.shots, args.chunk_size, args.noise,  args.out_dir
    )
