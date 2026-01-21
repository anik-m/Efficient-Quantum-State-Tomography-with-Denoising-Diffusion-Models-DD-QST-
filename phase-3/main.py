import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import sys
from sklearn.model_selection import train_test_split

from config import DEFAULTS
from data_gen import generate_synthetic_data, generate_universal_dataset
from dataset import QuantumStateDataset
from model import ConditionalD3PM
from diffusion import DiscreteDiffusion
from reconstruct import linear_inversion, get_metrics
from qiskit.quantum_info import state_fidelity

def load_and_split_data(file_path, test_size=0.2):
    """
    Loads Master RQC dataset and performs a STRATIFIED split based on depth.
    This ensures the Test Set covers all depths (2, 4, 6...) equally.
    """
    print(f"--- Loading Master Dataset: {file_path} ---")
    full_dataset = torch.load(file_path)
    
    # Extract depths for stratification
    depths = [item['depth'] for item in full_dataset]
    
    # Split INDICES, not data (to save memory during split)
    indices = list(range(len(full_dataset)))
    train_idx, test_idx = train_test_split(
        indices, 
        test_size=test_size, 
        stratify=depths,  # <--- CRITICAL for Plots
        random_state=42   # Fixed seed for reproducibility
    )
    
    train_data = [full_dataset[i] for i in train_idx]
    test_data = [full_dataset[i] for i in test_idx]
    
    print(f"Split Complete: {len(train_data)} Train | {len(test_data)} Test")
    return train_data, test_data
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

    # Handle notebook execution
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
    if args.state_type == 'rqc':
        print(f"RQC Depth: {args.rqc_depth}")
        raw_data, basis_list = generate_universal_dataset(
            num_samples=args.num_samples,
            num_qubits=args.num_qubits,
            min_depth=args.min_depth,
            max_depth=args.max_depth,
            shots=args.shots_train,
            noise_type=args.noise_type
        )
    else:
        raw_data, basis_list, target_state = generate_synthetic_data(
            args.num_qubits, 
            args.state_type, 
            args.shots_train,
            args.noise_type,
            args.rqc_depth
        )
    print(f"Training: {args.shots_train} shots | {args.epochs} epochs")
    
    # 1. Generate Training Data
    # Note: For RQC, 'target_state' is generated dynamically
    # raw_data, basis_list, target_state = generate_synthetic_data(
    #     args.num_qubits, 
    #     args.state_type, 
    #     args.shots_train,
    #     args.noise_type,
    #     args.rqc_depth
    # )

    # 1. Load & Split

    train_data_raw, test_data_raw = load_and_split_data(args.dataset_path)   
    # 2. Dataset Setup
    # dataset = QuantumStateDataset(raw_data, args.num_qubits)
    # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    train_dataset = QuantumStateDataset(train_data_raw)
    test_dataset = QuantumStateDataset(test_data_raw)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
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

    # 5. Inference
    print("\n--- Generating Synthetic Samples ---")
    model.eval()
    synthetic_results = {}
    
    # Generate MORE shots than experimental data to smooth noise
    shots_infer = 10000 
    for i, basis in enumerate(basis_list):
        samples = diffusion.p_sample(shots_infer, i, args.num_qubits)
        synthetic_results[basis] = samples.cpu().numpy()

    # 6. Reconstruction
    print("--- Reconstructing State ---")
    rho_recon = linear_inversion(synthetic_results, args.num_qubits)
    
    # 7. Metrics
    fid = state_fidelity(target_state, rho_recon)
    purity, vn_ent, ent_ent = get_metrics(rho_recon, args.num_qubits)
    
    print(f"\n=== Final Results ===")
    print(f"Fidelity:             {fid:.5f}")
    print(f"Purity:               {purity:.5f}")
    print(f"Von Neumann Entropy:  {vn_ent:.5f}")
    print(f"Entanglement Entropy: {ent_ent:.5f}")
    
    if fid > 0.9:
        print("SUCCESS: High fidelity reconstruction.")
    else:
        print("WARNING: Low fidelity.")

if __name__ == "__main__":
    main()
