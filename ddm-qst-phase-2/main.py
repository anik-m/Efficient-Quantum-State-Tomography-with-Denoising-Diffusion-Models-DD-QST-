import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import sys

# Imports
from config import DEFAULTS
from data_gen import generate_synthetic_data
from dataset import QuantumStateDataset
from model import ConditionalD3PM
from diffusion import DiscreteDiffusion
from reconstruct import linear_inversion
from qiskit.quantum_info import state_fidelity, Statevector

def get_args():
    parser = argparse.ArgumentParser(description="Phase 2: DDM Quantum State Tomography")
    
    # Physics Args
    parser.add_argument('--num_qubits', type=int, default=DEFAULTS['num_qubits'],
                        help='Number of qubits (N)')
    parser.add_argument('--state_type', type=str, default=DEFAULTS['state_type'],
                        choices=['plus', 'bell', 'ghz'], help='Target quantum state')
    
    # Training Args
    parser.add_argument('--epochs', type=int, default=DEFAULTS['num_epochs'],
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=DEFAULTS['batch_size'])
    parser.add_argument('--lr', type=float, default=DEFAULTS['learning_rate'])
    
    # Data Args
    parser.add_argument('--shots_train', type=int, default=DEFAULTS['shots_train'])
    parser.add_argument('--shots_infer', type=int, default=DEFAULTS['shots_infer'])

    # Model Args (Advanced)
    parser.add_argument('--timesteps', type=int, default=DEFAULTS['num_timesteps'])
    parser.add_argument('--hidden_dim', type=int, default=DEFAULTS['hidden_dim'])
    parser.add_argument('--embed_dim', type=int, default=DEFAULTS['embed_dim'])
    
    # If running in a notebook, use empty args to avoid errors with sys.argv
    if 'ipykernel_launcher' in sys.argv[0]:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
        
    return args

def main():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Configuration: N={args.num_qubits} | State={args.state_type} | Device={device} ---")

    # 1. Generate Training Data (Pass args dynamically)
    raw_data, basis_list = generate_synthetic_data(
        args.num_qubits, 
        args.state_type, 
        args.shots_train
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
        num_blocks=DEFAULTS['num_res_blocks'] # Keep this constant or add arg if needed
    ).to(device)
    
    diffusion = DiscreteDiffusion(model, args.timesteps, device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # 4. Training Loop
    print(f"--- Starting Training ({args.epochs} Epochs) ---")
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for x_0, basis_idx in loader:
            x_0 = x_0.to(device)
            basis_idx = basis_idx.to(device)
            
            t = torch.randint(1, args.timesteps + 1, (x_0.shape[0],), device=device)
            x_t = diffusion.q_sample(x_0, t)
            
            pred_logits = model(x_t, t, basis_idx)
            
            # Loss: (Batch, Classes, N)
            loss = F.cross_entropy(pred_logits.permute(0, 2, 1), x_0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}: Loss {total_loss/len(loader):.4f}")

    # 5. Inference
    print("--- Generating Synthetic Samples ---")
    model.eval()
    synthetic_results = {}
    
    for i, basis in enumerate(basis_list):
        samples = diffusion.p_sample(args.shots_infer, i, args.num_qubits)
        synthetic_results[basis] = samples.cpu().numpy()

    # 6. Reconstruction
    print("--- Reconstructing State ---")
    rho_recon = linear_inversion(synthetic_results, args.num_qubits)
    
    # Define Target Logic dynamically based on args
    if args.state_type == 'plus':
        # |+> tensor product N times
        target = Statevector.from_label('+' * args.num_qubits)
    elif args.state_type in ['bell', 'ghz']:
        # GHZ state: (|0...0> + |1...1>) / sqrt(2)
        zero_state = Statevector.from_label('0' * args.num_qubits)
        one_state = Statevector.from_label('1' * args.num_qubits)
        target = (zero_state + one_state) / np.sqrt(2)
    
    fid = state_fidelity(target, rho_recon)
    print(f"\nFinal Result for {args.num_qubits}-Qubit {args.state_type.upper()}:")
    print(f"Reconstruction Fidelity: {fid:.5f}")
    
    if fid > 0.9:
        print("SUCCESS: High fidelity entanglement verification.")
    else:
        print("WARNING: Low fidelity.")
    # Insert inside main() after generate_synthetic_data()

    print("--- Running Baseline Check on Training Data ---")
    # Group raw data by basis for linear inversion
    baseline_data = {}
    for entry in raw_data:
        baseline_data[entry['basis_str']] = np.array([list(k) for k in entry['counts'].keys() for _ in range(entry['counts'][k])])

    # Reconstruct using the training data directly
    rho_baseline = linear_inversion(baseline_data, cfg.NUM_QUBITS)

    # Check fidelity
    # Note: Re-define 'target' here or move target definition up
    if cfg.STATE_TYPE == 'ghz':
        target_check = (Statevector.from_label('0'*cfg.NUM_QUBITS) + Statevector.from_label('1'*cfg.NUM_QUBITS)) / np.sqrt(2)
        print(f"Baseline Fidelity (Upper Bound): {state_fidelity(target_check, rho_baseline):.5f}")

if __name__ == "__main__":
    main()
