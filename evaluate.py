import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import os
import torch.nn.functional as F
from itertools import product
from qiskit.quantum_info import DensityMatrix, state_fidelity
from qiskit.visualization import plot_state_city

# Custom Project Imports
from model import ConditionalD3PM
from diffusion import DiscreteDiffusion
from reconstruct import linear_inversion, get_metrics

# Set professional plotting style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

# --- HELPER FUNCTIONS ---

def calculate_z_bias(counts_dict, num_qubits):
    """Calculates population imbalance (Bias towards 0) in Z-basis."""
    z_key = 'Z' * num_qubits
    if z_key in counts_dict:
        samples = counts_dict[z_key] # (Shots, Qubits)
        total_zeros = np.sum(samples == 0)
        return total_zeros / samples.size
    return 0.5 

def format_raw_counts_for_inversion(measurements_list):
    """
    Adapts Qiskit data structure for Linear Inversion.
    CRITICAL: Applies [::-1] fix for Qiskit Little-Endian -> PyTorch Big-Endian.
    """
    formatted = {}
    for m in measurements_list:
        basis = m['basis']
        expanded_samples = []
        for bitstr, count in m['counts'].items():
            # Flip string to match model's qubit ordering
            bits = [int(b) for b in bitstr][::-1] 
            expanded_samples.extend([bits] * count)
        formatted[basis] = np.array(expanded_samples)
    return formatted

def get_denoising_trajectory(model, diffusion, basis_idx, num_qubits, device):
    """Captures the mean magnetization of qubits over time t."""
    history = []
    # Start with random noise
    x = torch.randint(0, 2, (500, num_qubits)).to(device)
    b_vec = torch.full((500,), basis_idx, dtype=torch.long).to(device)
    
    # Iterate backwards from T to 1
    for t in reversed(range(1, diffusion.num_timesteps + 1)):
        t_vec = torch.full((500,), t, dtype=torch.long).to(device)
        logits = model(x, t_vec, b_vec)
        probs = F.softmax(logits, dim=2)
        
        # Sample next step
        flat_probs = probs.view(-1, 2)
        x = torch.multinomial(flat_probs, 1).view(500, num_qubits)
        
        # Calculate mean magnetization (probability of being 1)
        mean_mag = x.float().mean(dim=0).cpu().numpy() 
        history.append(mean_mag)
        
    return np.array(history) # Shape: (T, Q)

# --- MAIN EVALUATION LOGIC ---

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 🚀 Starting Evaluation (N={args.num_qubits}) ---")
    
    # 1. Load Unseen Test Data
    if not os.path.exists(args.test_data_path):
        raise FileNotFoundError(f"Test data file not found: {args.test_data_path}")
    
    print(f"Loading unseen circuits from: {args.test_data_path}")
    raw_dataset = torch.load(args.test_data_path, weights_only=False)
    print(f"Loaded {len(raw_dataset)} unseen test circuits.")

    # 2. Load Model
    num_bases = 3 ** args.num_qubits
    model = ConditionalD3PM(args.num_qubits, num_bases, 
                           args.num_timesteps, args.embed_dim, 
                           args.hidden_dim, args.num_res_blocks).to(device)
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model weights not found: {args.model_path}")
        
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))
    model.eval()
    diffusion = DiscreteDiffusion(model, args.num_timesteps, device)

    # 3. Evaluation Loop
    records = []
    bases_strs = [''.join(p) for p in product(['X', 'Y', 'Z'], repeat=args.num_qubits)]
    basis_to_idx = {b: i for i, b in enumerate(bases_strs)}
    
    # Limit evaluation count if needed
    eval_count = min(len(raw_dataset), args.num_eval_states)
    print(f"Evaluating {eval_count} states...")

    for i in range(eval_count):
        state_data = raw_dataset[i]
        target_vec = state_data['clean_state_vec']
        target_dm = DensityMatrix(target_vec)
        depth = state_data.get('depth', 0)
        state_type = state_data.get('type', 'RQC') 
        
        # A. Baseline (Raw Noisy Data)
        raw_input = format_raw_counts_for_inversion(state_data['measurements'])
        rho_raw = linear_inversion(raw_input, args.num_qubits)
        fid_raw = state_fidelity(target_dm, rho_raw)
        _, s_raw, _ = get_metrics(rho_raw, args.num_qubits)
        
        # B. D3PM (AI Denoising)
        syn_counts = {}
        for b_str in bases_strs:
            samples = diffusion.p_sample(args.shots_infer, basis_to_idx[b_str], args.num_qubits)
            syn_counts[b_str] = samples.cpu().numpy()
            
        rho_d3pm = linear_inversion(syn_counts, args.num_qubits)
        fid_d3pm = state_fidelity(target_dm, rho_d3pm)
        _, s_d3pm, _ = get_metrics(rho_d3pm, args.num_qubits)
        bias_val = calculate_z_bias(syn_counts, args.num_qubits)

        print(f"State {i} (D={depth}): Raw={fid_raw:.3f} -> D3PM={fid_d3pm:.3f}")
        
        records.append({
            'ID': i, 'Depth': depth, 'Type': state_type, 
            'Raw_Fidelity': fid_raw, 'D3PM_Fidelity': fid_d3pm,
            'D3PM_Entropy': s_d3pm, 'Raw_Entropy': s_raw, 'Bias': bias_val
        })

    # 4. Save Metrics & Generate Plots
    df = pd.DataFrame(records)
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(f"{args.out_dir}/eval_metrics.csv", index=False)
    
    print("\n--- Generating Plots ---")
    
    # Plot 1: Fidelity Lift (Scatter)
    
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df, x='Raw_Fidelity', y='D3PM_Fidelity', hue='Depth', palette='viridis', s=100)
    plt.plot([0,1], [0,1], 'r--', label='No Improvement')
    plt.title("1. Fidelity Lift: AI vs Standard")
    plt.xlim(0.4, 1.0); plt.ylim(0.4, 1.0)
    plt.savefig(f"{args.out_dir}/1_fidelity_lift.png")
    plt.close()
    
    # Plot 2: Universality (Line)
    
    df_melt = pd.melt(df, id_vars=['Depth'], value_vars=['Raw_Fidelity', 'D3PM_Fidelity'], 
                      var_name='Method', value_name='Fidelity')
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_melt, x='Depth', y='Fidelity', hue='Method', marker='o')
    plt.axhline(1.0, color='gray', linestyle='--')
    plt.title("2. Universality: Robustness vs Complexity")
    plt.savefig(f"{args.out_dir}/2_universality.png")
    plt.close()
    
    # Plot 3: Entropy (Violin)
    
    df_ent = pd.melt(df, id_vars=['ID'], value_vars=['Raw_Entropy', 'D3PM_Entropy'], 
                     var_name='Method', value_name='Entropy')
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df_ent, x='Method', y='Entropy', palette="muted")
    plt.axhline(0, color='black', linestyle='--')
    plt.title("3. Physics Awareness: Entropy Reduction")
    plt.savefig(f"{args.out_dir}/3_entropy.png")
    plt.close()

    # Plot 4: Bias Histogram
    

# [Image of data distribution histogram]

    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x='Bias', bins=10, kde=True, color='purple')
    plt.axvline(0.5, color='red', linestyle='--')
    plt.title("4. Thermal Bias Correction")
    plt.savefig(f"{args.out_dir}/4_bias.png")
    plt.close()
    
    # Plot 5: Dynamics (Trajectory)
    
    print("Generating Dynamics Plot...")
    z_idx = basis_to_idx['Z' * args.num_qubits]
    traj = get_denoising_trajectory(model, diffusion, z_idx, args.num_qubits, device)
    plt.figure(figsize=(10, 6))
    steps = np.arange(args.num_timesteps)
    for q in range(args.num_qubits):
        plt.plot(steps, traj[:, q], label=f'Qubit {q}')
    plt.gca().invert_xaxis()
    plt.title("5. Denoising Trajectory")
    plt.legend()
    plt.savefig(f"{args.out_dir}/5_dynamics.png")
    plt.close()

    # Plot 6: Shot Efficiency
    
    print("Generating Efficiency Plot...")
    shot_steps = [100, 500, 1000, 2500, 5000]
    eff_fids = []
    # Use first state for efficiency test
    t_vec = raw_dataset[0]['clean_state_vec']
    t_dm = DensityMatrix(t_vec)
    large_counts = {}
    for b_str in bases_strs:
        s = diffusion.p_sample(max(shot_steps), basis_to_idx[b_str], args.num_qubits)
        large_counts[b_str] = s.cpu().numpy()
        
    for s in shot_steps:
        sub = {k: v[:s] for k, v in large_counts.items()}
        rho = linear_inversion(sub, args.num_qubits)
        eff_fids.append(state_fidelity(t_dm, rho))
        
    plt.figure(figsize=(8, 6))
    plt.plot(shot_steps, eff_fids, 'b-o')
    plt.xscale('log')
    plt.title("6. Data Amplification Efficiency")
    plt.ylabel("Fidelity"); plt.xlabel("Synthetic Shots")
    plt.savefig(f"{args.out_dir}/6_efficiency.png")
    plt.close()
    
    print(f"\n✅ Evaluation Complete. Results in {args.out_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--model_path', required=True, help="Path to .pt weights")
    parser.add_argument('--test_data_path', required=True, help="Path to *_test_circuits.pt")
    parser.add_argument('--out_dir', default='paper_results')
    
    # Physics
    parser.add_argument('--num_qubits', type=int, default=3)
    parser.add_argument('--shots_infer', type=int, default=2000)
    parser.add_argument('--num_eval_states', type=int, default=50)
    
    # Architecture (Must match Training)
    parser.add_argument('--num_timesteps', type=int, default=100)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_res_blocks', type=int, default=4)
    
    args = parser.parse_args()
    evaluate(args)
