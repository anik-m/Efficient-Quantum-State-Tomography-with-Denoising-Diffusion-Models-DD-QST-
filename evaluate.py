import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
import os
import torch.nn.functional as F
from itertools import product
from qiskit.quantum_info import DensityMatrix, state_fidelity, entropy
from qiskit.visualization import plot_state_city

# Custom Project Imports
from model import ConditionalD3PM
from diffusion import DiscreteDiffusion
from reconstruct import linear_inversion, get_metrics

sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

def calculate_z_bias(counts_dict, num_qubits):
    z_key = 'Z' * num_qubits
    if z_key in counts_dict:
        samples = counts_dict[z_key] 
        total_zeros = np.sum(samples == 0)
        return total_zeros / samples.size
    return 0.5 

def format_raw_counts_for_inversion(measurements_list):
    formatted = {}
    for m in measurements_list:
        basis = m['basis']
        expanded_samples = []
        for bitstr, count in m['counts'].items():
            bits = [int(b) for b in bitstr][::-1] 
            expanded_samples.extend([bits] * count)
        formatted[basis] = np.array(expanded_samples)
    return formatted

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 🚀 Evaluation (N={args.num_qubits}) ---")
    
    # 1. Load Unseen Test Data
    test_file = "checkpoints/test_circuits_unseen.pt"
    if not os.path.exists(test_file):
        raise FileNotFoundError("Unseen test data not found. Run main.py first.")
    
    print(f"Loading unseen data from {test_file}...")
    raw_dataset = torch.load(test_file)
    print(f"Loaded {len(raw_dataset)} unseen test circuits.")

    # 2. Load Model
    num_bases = 3 ** args.num_qubits
    model = ConditionalD3PM(args.num_qubits, num_bases, 
                           args.num_timesteps, args.embed_dim, 
                           args.hidden_dim, args.num_res_blocks).to(device)
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found at {args.model_path}")
        
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    diffusion = DiscreteDiffusion(model, args.num_timesteps, device)

    # 3. Evaluation Loop
    records = []
    bases_strs = [''.join(p) for p in product(['X', 'Y', 'Z'], repeat=args.num_qubits)]
    basis_to_idx = {b: i for i, b in enumerate(bases_strs)}
    
    eval_count = min(len(raw_dataset), args.num_eval_states)

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

    # 4. Save & Plot
    df = pd.DataFrame(records)
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(f"{args.out_dir}/eval_metrics.csv", index=False)
    
    # --- PLOTS ---
    print("Generating Plots...")
    
    # 1. Fidelity Lift
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df, x='Raw_Fidelity', y='D3PM_Fidelity', hue='Depth', s=100)
    plt.plot([0,1], [0,1], 'r--', label='No Improvement')
    plt.title("AI Denoising Performance")
    plt.savefig(f"{args.out_dir}/1_fidelity_lift.png")
    plt.close()
    
    # 2. Universality (Melted)
    df_melt = pd.melt(df, id_vars=['Depth'], value_vars=['Raw_Fidelity', 'D3PM_Fidelity'], 
                      var_name='Method', value_name='Fidelity')
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_melt, x='Depth', y='Fidelity', hue='Method', marker='o')
    plt.title("Performance vs Complexity")
    plt.savefig(f"{args.out_dir}/2_universality.png")
    plt.close()
    
    print(f"Done. Results in {args.out_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--out_dir', default='paper_results')
    parser.add_argument('--num_qubits', type=int, default=3)
    parser.add_argument('--shots_infer', type=int, default=2000)
    parser.add_argument('--num_eval_states', type=int, default=50)
    
    # Architecture Defaults
    parser.add_argument('--num_timesteps', type=int, default=100)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_res_blocks', type=int, default=4)
    
    args = parser.parse_args()
    evaluate(args)
