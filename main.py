import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import argparse
import os
import glob
import random
import sys
import numpy as np

# Custom Imports
from dataset import QuantumStateDataset
from model import ConditionalD3PM
from diffusion import DiscreteDiffusion

def load_all_circuits(data_path):
    all_data = []
    if os.path.isfile(data_path):
        print(f"Loading file: {data_path}")
        try:
            data = torch.load(data_path, weights_only=False)
            if isinstance(data, list): all_data = data
            else: all_data = [data]
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    elif os.path.isdir(data_path):
        print(f"Loading folder: {data_path}")
        files = sorted(glob.glob(os.path.join(data_path, "*.pt")))
        for f in files:
            try:
                part = torch.load(f, weights_only=False)
                all_data.extend(part)
            except: pass
    return all_data

def create_sanity_circuit(num_qubits):
    """Creates a synthetic Bell State (00 + 11) to test if model can learn."""
    print("⚠️ GENERATING SYNTHETIC BELL STATE FOR SANITY CHECK ⚠️")
    # Perfect correlations: 50% '00', 50% '11' in Z-basis
    # Uniform in X-basis (for Bell state)
    
    measurements = []
    
    # Z-Basis: 00 and 11
    counts_z = {'0'*num_qubits: 500, '1'*num_qubits: 500}
    measurements.append({'basis': 'Z'*num_qubits, 'counts': counts_z})
    
    # X-Basis: 00, 11, 01, 10 (approx uniform for Bell |00>+|11>)
    # Actually Bell State (|00>+|11>) in X basis is |++> + |--> ...
    # Let's just make it learn to COPY "00" and "11" perfectly first (Classical Correlation)
    
    return [{
        'measurements': measurements,
        'clean_state_vec': np.zeros(2**num_qubits), # Dummy
        'depth': 0,
        'type': 'sanity'
    }]

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 🧠 Run: '{args.run_name}' on {device} ---")
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Load Data
    if args.sanity_check:
        all_circuits = create_sanity_circuit(args.num_qubits)
    else:
        all_circuits = load_all_circuits(args.data_path)
    
    # 2. Select Training Data
    random.shuffle(all_circuits)
    num_train = int(len(all_circuits) * args.train_ratio)
    training_circuits = all_circuits[:max(1, num_train)]
    
    # Eval on the SAME circuits we train on (Memorization test)
    eval_subset = training_circuits[:args.num_eval_circuits]
    
    # Save Eval Subset
    torch.save(eval_subset, os.path.join(args.save_dir, f"{args.run_name}_eval.pt"))

    # 3. Create Datasets
    train_ds = QuantumStateDataset(training_circuits, args.num_qubits)
    # Val on same set to check overfitting capability
    val_ds = QuantumStateDataset(eval_subset, args.num_qubits)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    
    print(f"Training on {len(train_ds)} shots.")

    # 4. Model & Diffusion
    model = ConditionalD3PM(args.num_qubits, 3**args.num_qubits, 
                           args.num_timesteps, args.embed_dim, 
                           args.hidden_dim, args.num_res_blocks).to(device)
    diffusion = DiscreteDiffusion(model, args.num_timesteps, device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 5. Training Loop
    print("\n--- Starting Training ---")
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0
        for x, basis in train_loader:
            x, basis = x.to(device), basis.to(device)
            t = torch.randint(1, args.num_timesteps+1, (x.size(0),)).to(device)
            x_t = diffusion.q_sample(x, t)
            logits = model(x_t, t, basis)
            loss = F.cross_entropy(logits.permute(0,2,1), x)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Monitor Validation
        if (epoch+1) % 5 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, basis in val_loader:
                    x, basis = x.to(device), basis.to(device)
                    t = torch.randint(1, args.num_timesteps+1, (x.size(0),)).to(device)
                    x_t = diffusion.q_sample(x, t)
                    logits = model(x_t, t, basis)
                    val_loss += F.cross_entropy(logits.permute(0,2,1), x).item()
            
            print(f"Ep {epoch+1:03d} | Loss: {train_loss/len(train_loader):.4f} | Val: {val_loss/len(val_loader):.4f}")
            
    # Save Final
    torch.save(model.state_dict(), os.path.join(args.save_dir, f"{args.run_name}_best.pt"))
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='dataset_parts')
    parser.add_argument('--save_dir', type=str, default='experiments/check')
    parser.add_argument('--run_name', type=str, default='model')
    parser.add_argument('--sanity_check', action='store_true', help="Ignore data, train on synthetic Bell state")
    
    parser.add_argument('--train_ratio', type=float, default=1.0)
    parser.add_argument('--num_eval_circuits', type=int, default=50)
    parser.add_argument('--num_qubits', type=int, default=2) # Default 2 for sanity check
    parser.add_argument('--num_epochs', type=int, default=100) # Increased default
    parser.add_argument('--batch_size', type=int, default=256) # Smaller batch for stability
    
    # Model
    parser.add_argument('--num_timesteps', type=int, default=100)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_res_blocks', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    args = parser.parse_args()
    train_model(args)
