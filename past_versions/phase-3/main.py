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
    """
    Universal Loader: Handles both directory of parts AND single .pt file.
    Returns a single list of circuit dictionaries.
    """
    all_data = []
    
    if os.path.isfile(data_path):
        print(f"Loading single dataset file: {data_path}")
        try:
            data = torch.load(data_path, weights_only=False)
            if isinstance(data, list):
                all_data = data
            else:
                # If it's a single dict (rare but possible), wrap in list
                all_data = [data]
        except Exception as e:
            print(f"Error loading file: {e}")
            sys.exit(1)

    elif os.path.isdir(data_path):
        print(f"Loading dataset parts from folder: {data_path}")
        files = sorted(glob.glob(os.path.join(data_path, "*.pt")))
        if not files:
            print("No .pt files found in directory!")
            sys.exit(1)
            
        for f in files:
            try:
                part = torch.load(f)
                all_data.extend(part)
            except Exception as e:
                print(f"Skipping corrupt file {f}: {e}")
    else:
        print(f"Error: Path not found: {data_path}")
        sys.exit(1)

    return all_data

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 🧠 Training Run: '{args.run_name}' on {device} ---")
    
    # 1. Setup Directories
    os.makedirs(args.save_dir, exist_ok=True)

    # 2. Load & Split Data (Strict Circuit Separation)
    print("Loading Master Dataset...")
    all_circuits = load_all_circuits(args.data_path)
    print(f"Total Unique Circuits Found: {len(all_circuits)}")
    
    # Shuffle and Split
    random.shuffle(all_circuits)
    split_idx = int(0.9 * len(all_circuits)) # 90/10 Split
    
    train_circuits = all_circuits[:split_idx]
    test_circuits = all_circuits[split_idx:]
    
    print(f"Train Set: {len(train_circuits)} circuits")
    print(f"Test Set:  {len(test_circuits)} circuits (Saved for Evaluation)")
    
    # 3. SAVE THE TEST DATA (The "Golden Record")
    # We save this specific list of circuits so evaluate.py uses EXACTLY these
    test_data_path = os.path.join(args.save_dir, f"{args.run_name}_test_circuits.pt")
    torch.save(test_circuits, test_data_path)
    print(f"✅ Unseen Test Data saved to: {test_data_path}")

    # 4. Create Datasets (Passing lists directly)
    # dataset.py must be the version that accepts a list in __init__
    train_dataset = QuantumStateDataset(train_circuits, args.num_qubits)
    # We create a validation loader from test circuits to monitor progress during training
    val_dataset = QuantumStateDataset(test_circuits, args.num_qubits)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 5. Initialize Model
    num_bases = 3 ** args.num_qubits
    model = ConditionalD3PM(args.num_qubits, num_bases, 
                           args.num_timesteps, args.embed_dim, 
                           args.hidden_dim, args.num_res_blocks).to(device)
    
    diffusion = DiscreteDiffusion(model, args.num_timesteps, device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 6. Training Loop
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
            
            # Loss: Cross Entropy
            loss = F.cross_entropy(pred_logits.permute(0, 2, 1), x_0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        # Validation Step
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
        
        print(f"Epoch {epoch+1:02d}/{args.num_epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        
        # Save Best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), model_save_path)
            print(f"  ⭐ New Best Model Saved (Val Loss: {best_val_loss:.4f})")

    print(f"\nTraining Complete. Best Model: {model_save_path}")
    print(f"Test Data for Evaluation: {test_data_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Path Arguments
    parser.add_argument('--data_path', type=str, required=True, help="Path to .pt file OR folder")
    parser.add_argument('--save_dir', type=str, default='experiments/default_run', help="Where to save model/test data")
    parser.add_argument('--run_name', type=str, default='model_v1', help="Prefix for saved files")
    
    # Physics Args
    parser.add_argument('--num_qubits', type=int, default=3)
    
    # Hyperparams
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_timesteps', type=int, default=100)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_res_blocks', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    args = parser.parse_args()
    train_model(args)
