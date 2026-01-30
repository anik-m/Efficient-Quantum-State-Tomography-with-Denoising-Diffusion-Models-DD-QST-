import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import argparse
import os
import glob
import random
import sys

# Custom Imports
from dataset import QuantumStateDataset
from model import ConditionalD3PM
from diffusion import DiscreteDiffusion

def load_all_circuits(data_path):
    """Robust data loader for single files or directories."""
    all_data = []
    if os.path.isfile(data_path):
        print(f"Loading single dataset file: {data_path}")
        try:
            # Fix for PyTorch 2.6+ security restriction
            data = torch.load(data_path, weights_only=False)
            if isinstance(data, list): all_data = data
            else: all_data = [data]
        except Exception as e:
            print(f"Error loading file: {e}")
            sys.exit(1)
    elif os.path.isdir(data_path):
        print(f"Loading dataset parts from folder: {data_path}")
        files = sorted(glob.glob(os.path.join(data_path, "*.pt")))
        if not files:
            print("No .pt files found!")
            sys.exit(1)
        for f in files:
            try:
                part = torch.load(f, weights_only=False)
                all_data.extend(part)
            except Exception as e:
                print(f"Skipping {f}: {e}")
    else:
        print(f"Error: Path not found: {data_path}")
        sys.exit(1)
    return all_data

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 🧠 Training Run: '{args.run_name}' on {device} ---")
    
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Load All Available Data
    print("Loading Master Dataset...")
    all_circuits = load_all_circuits(args.data_path)
    total_found = len(all_circuits)
    print(f"Total Circuits Found: {total_found}")
    
    # 2. Select Training Portion
    # We shuffle first to ensure the selection is random
    random.shuffle(all_circuits)
    
    # Calculate how many circuits to train on
    num_train = int(total_found * args.train_ratio)
    if num_train < 1:
        print("Error: train_ratio resulted in 0 circuits.")
        sys.exit(1)
        
    training_circuits = all_circuits[:num_train]
    print(f"Training on: {len(training_circuits)} circuits ({args.train_ratio*100}% of available data)")
    
    # 3. Select Eval Subset FROM INSIDE Training Set
    # We pick N random circuits from the ones we are about to train on.
    num_eval = min(len(training_circuits), args.num_eval_circuits)
    # Use random.sample to pick unique items without replacement
    eval_subset = random.sample(training_circuits, num_eval)
    
    print(f"Evaluation Set: {len(eval_subset)} circuits (Random selection from Train Set)")
    
    # 4. Save the Evaluation Subset
    # evaluate.py will use this file to test reconstruction quality
    eval_data_path = os.path.join(args.save_dir, f"{args.run_name}_eval_subset.pt")
    torch.save(eval_subset, eval_data_path)
    print(f"✅ Evaluation Subset saved to: {eval_data_path}")

    # 5. Create Datasets
    # Train Dataset contains the eval subset implicitly
    train_dataset = QuantumStateDataset(training_circuits, args.num_qubits)
    # Val Dataset is just the subset (for monitoring loss during training)
    val_dataset = QuantumStateDataset(eval_subset, args.num_qubits)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 6. Initialize Model (Using Fixes)
    num_bases = 3 ** args.num_qubits
    model = ConditionalD3PM(args.num_qubits, num_bases, 
                           args.num_timesteps, args.embed_dim, 
                           args.hidden_dim, args.num_res_blocks).to(device)
    
    diffusion = DiscreteDiffusion(model, args.num_timesteps, device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 7. Training Loop
    best_val_loss = float('inf')
    model_save_path = os.path.join(args.save_dir, f"{args.run_name}_best_model.pt")
    
    print("\n--- Starting Training Loop ---")
    for epoch in range(args.num_epochs):
        model.train()
        total_train_loss = 0
        
        for x_0, basis_idx in train_loader:
            x_0, basis_idx = x_0.to(device), basis_idx.to(device)
            t = torch.randint(1, args.num_timesteps + 1, (x_0.shape[0],)).to(device)
            
            x_t = diffusion.q_sample(x_0, t)
            pred_logits = model(x_t, t, basis_idx)
            
            loss = F.cross_entropy(pred_logits.permute(0, 2, 1), x_0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        # Validation (Checking reconstruction of the eval subset)
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_0, basis_idx in val_loader:
                x_0, basis_idx = x_0.to(device), basis_idx.to(device)
                t = torch.randint(1, args.num_timesteps + 1, (x_0.shape[0],)).to(device)
                x_t = diffusion.q_sample(x_0, t)
                logits = model(x_t, t, basis_idx)
                val_loss = F.cross_entropy(logits.permute(0, 2, 1), x_0)
                total_val_loss += val_loss.item()
        
        avg_train = total_train_loss / max(1, len(train_loader))
        avg_val = total_val_loss / max(1, len(val_loader))
        
        print(f"Epoch {epoch+1:02d}/{args.num_epochs} | Train Loss: {avg_train:.4f} | Eval Loss: {avg_val:.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), model_save_path)
            print(f"  ⭐ Saved Model (Loss: {best_val_loss:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='experiments/reconstruction_test')
    parser.add_argument('--run_name', type=str, default='model_recon')
    
    # Control Arguments
    parser.add_argument('--train_ratio', type=float, default=1.0, help="Fraction of total data to use for training (0.0 to 1.0)")
    parser.add_argument('--num_eval_circuits', type=int, default=50, help="Number of trained circuits to verify in evaluation")
    
    parser.add_argument('--num_qubits', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_timesteps', type=int, default=100)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_res_blocks', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    args = parser.parse_args()
    train_model(args)
