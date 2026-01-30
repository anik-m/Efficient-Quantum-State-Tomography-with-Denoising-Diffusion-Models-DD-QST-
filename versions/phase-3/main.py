import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Qiskit & Visualization
from qiskit.quantum_info import state_fidelity, DensityMatrix, Statevector
from qiskit.visualization import plot_state_city

# Project Modules
from config import DEFAULTS
from data_gen import generate_synthetic_data, generate_universal_dataset
from dataset import QuantumStateDataset
from model import ConditionalD3PM
from diffusion import DiscreteDiffusion
from reconstruct import linear_inversion, get_metrics

# -----------------------------------------------------------------------------
# Visualization & Helper Functions
# -----------------------------------------------------------------------------

def counts_to_samples(counts_dict, num_qubits, num_shots_target):
    """
    Converts Qiskit counts (from raw data) into the sample format expected 
    by linear_inversion. Matches endianness logic from dataset.py.
    """
    samples = []
    for bitstr, count in counts_dict.items():
        # REVERSE bitstring to match dataset.py logic (Qiskit Little Endian -> List)
        # '01' (q1=0, q0=1) -> [1, 0]
        bits = [int(b) for b in bitstr[::-1]]
        samples.extend([bits] * count)
    
    # Handle exact shot counts (pad or trim)
    current_len = len(samples)
    if current_len < num_shots_target:
        # Pad with the first sample if we are short (rare/edge case)
        if current_len > 0:
            samples.extend([samples[0]] * (num_shots_target - current_len))
        else:
            # If empty, return zeros
            return np.zeros((num_shots_target, num_qubits))
    
    # Trim to exact target shots
    return np.array(samples[:num_shots_target])

def plot_comparison(target_rho, raw_rho, denoised_rho, fid_raw, fid_denoised, save_prefix="qst_result"):
    """
    Generates three key visualizations for presentation slides:
    1. 3D State City Plot (Qualitative comparison)
    2. Fidelity Bar Chart (Quantitative lift)
    3. Error Heatmap (Where the noise was fixed)
    """
    print("Generating visualizations...")
    
    # --- VISUAL 1: State City Plots (Real Component) ---
    fig = plt.figure(figsize=(20, 6))
    
    # Target
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    plot_state_city(target_rho, ax=ax1, title="Ground Truth (Target)")
    
    # Baseline
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    plot_state_city(raw_rho, ax=ax2, title=f"Baseline (Linear Inv)\nFidelity: {fid_raw:.4f}")
    
    # Denoised
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    plot_state_city(denoised_rho, ax=ax3, title=f"Denoised (D3PM)\nFidelity: {fid_denoised:.4f}")
    
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_city_plot.png", dpi=300)
    plt.show()

    # --- VISUAL 2: Fidelity Improvement Bar Chart ---
    plt.figure(figsize=(8, 6))
    methods = ['Baseline', 'Denoised (Ours)']
    fidelities = [fid_raw, fid_denoised]
    colors = ['#95a5a6', '#2ecc71'] # Grey vs Green
    
    bars = plt.bar(methods, fidelities, color=colors, width=0.6)
    plt.ylim(0, 1.1)
    plt.ylabel("State Fidelity", fontsize=12)
    plt.title("Reconstruction Fidelity Comparison", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
                 
    plt.savefig(f"{save_prefix}_fidelity_bar.png", dpi=300)
    plt.show()
    
    # --- VISUAL 3: Error Heatmaps ---
    # Calculate absolute difference matrices
    diff_raw = np.abs(target_rho.data - raw_rho.data)
    diff_denoised = np.abs(target_rho.data - denoised_rho.data)
    
    # Determine common scale for comparison
    max_err = max(np.max(diff_raw), np.max(diff_denoised))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.heatmap(diff_raw, ax=ax1, cmap="Reds", vmin=0, vmax=max_err, annot=False)
    ax1.set_title(f"Baseline Error |Target - Raw|\nAvg Error: {np.mean(diff_raw):.4f}")
    
    sns.heatmap(diff_denoised, ax=ax2, cmap="Reds", vmin=0, vmax=max_err, annot=False)
    ax2.set_title(f"Denoised Error |Target - D3PM|\nAvg Error: {np.mean(diff_denoised):.4f}")
    
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_error_heatmap.png", dpi=300)
    plt.show()

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="D3PM Quantum Tomography")
    
    # Experiment Config
    parser.add_argument('--num_qubits', type=int, default=DEFAULTS['num_qubits'])
    parser.add_argument('--state_type', type=str, default='rqc', 
                        choices=['plus', 'bell', 'ghz', 'rqc'])
    parser.add_argument('--noise_type', type=str, default='readout',
                        choices=['torino', 'ideal', 'readout', 'depolarizing', 'thermal'])
    parser.add_argument('--rqc_depth', type=int, default=5, help='Depth of random circuit')
    
    # Training Config
    parser.add_argument('--shots_train', type=int, default=DEFAULTS['shots_train'])
    parser.add_argument('--epochs', type=int, default=DEFAULTS['num_epochs'])
    parser.add_argument('--batch_size', type=int, default=DEFAULTS['batch_size'])
    parser.add_argument('--lr', type=float, default=DEFAULTS['learning_rate'])
    
    # Architecture
    parser.add_argument('--hidden_dim', type=int, default=DEFAULTS['hidden_dim'])
    parser.add_argument('--embed_dim', type=int, default=DEFAULTS['embed_dim'])
    parser.add_argument('--timesteps', type=int, default=DEFAULTS['num_timesteps'])

    # RQC Universal Config
    parser.add_argument('--num_samples', type=int, default=100, help='Number of RQC circuits')
    parser.add_argument('--min_depth', type=int, default=2)
    parser.add_argument('--max_depth', type=int, default=5)

    if 'ipykernel_launcher' in sys.argv[0]:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    return args

def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n=== Configuration ===")
    print(f"State: {args.state_type.upper()} (N={args.num_qubits})")
    print(f"Noise: {args.noise_type.upper()}")
    
    # 1. Generate Data
    if args.state_type == 'rqc':
        print(f"RQC Depth: {args.rqc_depth}")
        # Note: generate_universal_dataset creates a dataset for general training
        # For single-instance tomography demo, we stick to generate_synthetic_data logic below
        # but we use the flags to control RQC generation.
    
    print(f"Training: {args.shots_train} shots | {args.epochs} epochs")
    
    # Generate Training Data & Ground Truth
    # raw_data is a list of dicts: [{'basis_str': 'XX', 'counts': {...}}, ...]
    raw_data, basis_list, target_state = generate_synthetic_data(
        args.num_qubits, 
        args.state_type, 
        args.shots_train,
        args.noise_type,
        args.rqc_depth
    )
    
    # 2. Dataset Setup
    dataset = QuantumStateDataset(raw_data, args.num_qubits)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 3. Model Setup
    model = ConditionalD3PM(
        num_qubits=args.num_qubits,
        num_bases=len(basis_list),
        num_timesteps=args.timesteps,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_blocks=DEFAULTS['num_res_blocks']
    ).to(device)
    
    diffusion = DiscreteDiffusion(model, args.timesteps, device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # 4. Training Loop
    print(f"\n--- Starting Training ---")
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for x_0, basis_idx in loader:
            x_0 = x_0.to(device)
            basis_idx = basis_idx.to(device)
            
            t = torch.randint(1, args.timesteps + 1, (x_0.shape[0],), device=device)
            x_t = diffusion.q_sample(x_0, t)
            pred_logits = model(x_t, t, basis_idx)
            
            loss = F.cross_entropy(pred_logits.permute(0, 2, 1), x_0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}: Loss {total_loss/len(loader):.4f}")

    # 5. Inference (Generate Denoised Data)
    print("\n--- Generating Synthetic Samples (Denoised) ---")
    model.eval()
    synthetic_results = {}
    
    # Generate MORE shots than experimental data to smooth out statistical noise
    shots_infer = 10000 
    for i, basis in enumerate(basis_list):
        samples = diffusion.p_sample(shots_infer, i, args.num_qubits)
        synthetic_results[basis] = samples.cpu().numpy()

    # 6. Baseline Reconstruction (No AI)
    print("--- Computing Baseline (Linear Inversion on Noisy Data) ---")
    # Convert raw counts to sample arrays for linear_inversion
    raw_samples_dict = {}
    for entry in raw_data:
        basis = entry['basis_str']
        counts = entry['counts']
        raw_samples_dict[basis] = counts_to_samples(counts, args.num_qubits, args.shots_train)
        
    rho_raw = linear_inversion(raw_samples_dict, args.num_qubits)

    # 7. Denoised Reconstruction (With AI)
    print("--- Computing Denoised Reconstruction ---")
    rho_denoised = linear_inversion(synthetic_results, args.num_qubits)
    
    # 8. Metrics & Comparison
    if isinstance(target_state, Statevector):
        target_dm = DensityMatrix(target_state)
    else:
        target_dm = target_state

    fid_raw = state_fidelity(target_dm, rho_raw)
    fid_denoised = state_fidelity(target_dm, rho_denoised)
    
    purity, vn_ent, ent_ent = get_metrics(rho_denoised, args.num_qubits)
    
    print(f"\n=== COMPARISON RESULTS ===")
    print(f"Baseline Fidelity:    {fid_raw:.5f}")
    print(f"Denoised Fidelity:    {fid_denoised:.5f}")
    print(f"Improvement:          {((fid_denoised - fid_raw)/fid_raw)*100:.2f}%")
    print(f"--------------------------")
    print(f"Denoised Purity:      {purity:.5f}")
    print(f"Denoised Von Neumann: {vn_ent:.5f}")

    if fid_denoised > 0.9:
        print("SUCCESS: High fidelity reconstruction.")
    else:
        print("WARNING: Low fidelity.")
        
    # 9. Visualization
    # Only visualize if running locally or in a notebook (prevents headless errors)
    try:
        plot_comparison(target_dm, rho_raw, rho_denoised, fid_raw, fid_denoised)
    except Exception as e:
        print(f"Visualization skipped: {e}")

if __name__ == "__main__":
    main()
