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

sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

def format_raw_counts_for_inversion(measurements_list):
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

def calculate_z_bias(counts_dict, num_qubits):
    z_key = 'Z' * num_qubits
    if z_key in counts_dict:
        samples = counts_dict[z_key] 
        total_zeros = np.sum(samples == 0)
        return total_zeros / samples.size
    return 0.5 

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 🚀 Evaluating Reconstruction (N={args.num_qubits}) ---")
    
    # 1. Load Eval Data (Subset of Training Data)
    if not os.path.exists(args.eval_data_path):
        raise FileNotFoundError(f"Eval file not found: {args.eval_data_path}")
    
    # weights_only=False required for Qiskit objects
    raw_dataset = torch.load(args.eval_data_path, weights_only=False)
    print(f"Loaded {len(raw_dataset)} circuits for evaluation.")

    # 2. Load Model
    num_bases = 3 ** args.num_qubits
    model = ConditionalD3PM(args.num_qubits, num_bases, 
                           args.num_timesteps, args.embed_dim, 
                           args.hidden_dim, args.num_res_blocks).to(device)
    
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path invalid: {args.model_path}")
        
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))
    model.eval()
    diffusion = DiscreteDiffusion(model, args.num_timesteps, device)

    # 3. Evaluation Loop
    records = []
    bases_strs = [''.join(p) for p in product(['X', 'Y', 'Z'], repeat=args.num_qubits)]
    basis_to_idx = {b: i for i, b in enumerate(bases_strs)}
    
    for i, state_data in enumerate(raw_dataset):
        target_dm = DensityMatrix(state_data['clean_state_vec'])
        depth = state_data.get('depth', 0)
        
        # A. Baseline (Linear Inversion on Raw Data)
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
            'ID': i, 'Depth': depth,
            'Raw_Fidelity': fid_raw, 'D3PM_Fidelity': fid_d3pm,
            'Raw_Entropy': s_raw, 'D3PM_Entropy': s_d3pm, 'Bias': bias_val
        })

    # 4. Save & Plot
    df = pd.DataFrame(records)
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(f"{args.out_dir}/metrics.csv", index=False)
    
    print("Generating Plots...")
    # Scatter Lift
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df, x='Raw_Fidelity', y='D3PM_Fidelity', hue='Depth', s=100)
    plt.plot([0,1], [0,1], 'r--', label="Identity")
    plt.title("Fidelity Lift (Training Set Reconstruction)")
    plt.savefig(f"{args.out_dir}/fidelity_lift.png"); plt.close()
    
    # Universality
    df_melt = pd.melt(df, id_vars=['Depth'], value_vars=['Raw_Fidelity', 'D3PM_Fidelity'], var_name='Method', value_name='Fidelity')
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_melt, x='Depth', y='Fidelity', hue='Method', marker='o')
    plt.title("Reconstruction vs Circuit Depth"); plt.savefig(f"{args.out_dir}/universality.png"); plt.close()
    
    print(f"Done. Check {args.out_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--eval_data_path', required=True, help="Path to *_eval_subset.pt")
    parser.add_argument('--out_dir', default='paper_results')
    parser.add_argument('--num_qubits', type=int, default=3)
    parser.add_argument('--shots_infer', type=int, default=2000)
    # Model Args
    parser.add_argument('--num_timesteps', type=int, default=100)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_res_blocks', type=int, default=4)
    
    args = parser.parse_args()
    evaluate(args)
