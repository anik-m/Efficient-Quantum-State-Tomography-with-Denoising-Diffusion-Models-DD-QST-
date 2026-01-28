import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import argparse
import os
import glob
import random

# Imports
from dataset import QuantumStateDataset
from model import ConditionalD3PM
from diffusion import DiscreteDiffusion

def load_all_data(data_path):
    """Helper to load all circuits into a single list, regardless of file structure."""
    all_circuits = []
    
    if os.path.isdir(data_path):
        # Folder mode
        files = sorted(glob.glob(os.path.join(data_path, "*.pt")))
        print(f"Found {len(files)} parts in folder.")
        for f in files:
            try:
                all_circuits.extend(torch.load(f))
            except:
                print(f"Failed to load {f}")
    elif os.path.isfile(data_path):
        # Single file mode
        print(f"Loading single file: {data_path}")
        all_circuits = torch.load(data_path)
    else:
        raise ValueError(f"Invalid path: {data_path}")
        
    return all_circuits

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training on {device} ---")

    # 1. Load Everything
    print("Loading Master Dataset...")
    all_circuits = load_all_data(args.data_path)
    print(f"Total Unique Circuits Loaded: {len(all_circuits)}")
    
    # 2. Strict Circuit-Level Split
    # We split the CIRCUITS, not the shots.
    random.shuffle(all_circuits)
    split_idx = int(0.9 * len(all_circuits))
    
    train_circuits = all_circuits[:split_idx]
    test_circuits = all_circuits[split_idx:]
    
    print(f"Train Circuits: {len(train_circuits)} | Test Circuits: {len(test_circuits)}")
    
    # 3. Save Test Data for Evaluation (CRITICAL)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(test_circuits, "checkpoints/test_circuits_unseen.pt")
    print("Saved unseen test circuits to 'checkpoints/test_circuits_unseen.pt'")

    # 4. Initialize Datasets (Passing lists, not paths)
    train_dataset = QuantumStateDataset(train_circuits, args.num_qubits)
    test_dataset = QuantumStateDataset(test_circuits, args.num_qubits)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 5. Initialize Model
    num_bases = 3 ** args.num_qubits
    model = ConditionalD3PM(args.num_qubits, num_bases, 
                           args.num_timesteps, args.embed_dim, 
                           args.hidden_dim, args.num_res_blocks).to(device)
    
    diffusion = DiscreteDiffusion(model, args.num_timesteps, device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 6. Training Loop
    best_val_loss = float('inf')
    
    print("\n--- Starting Training Loop ---")
    for epoch in range(args.num_epochs):
        model.train()
        total_train_loss = 0
        
        for x_0, basis_idx in train_loader:
            x_0, basis_idx = x_0.to(device), basis_idx.to(device)
            t = torch.randint(1, args.num_timesteps + 1, (x_0.shape[0],)).to(device)
            
            x_t = diffusion.q_sample(x_0, t)
            pred_logits = model(x_t, t, basis_idx)
            
            # Cross Entropy expects (Batch, Classes, Seq)
            loss = F.cross_entropy(pred_logits.permute(0, 2, 1), x_0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
        # Validation Loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_0, basis_idx in test_loader:
                x_0, basis_idx = x_0.to(device), basis_idx.to(device)
                t = torch.randint(1, args.num_timesteps + 1, (x_0.shape[0],)).to(device)
                x_t = diffusion.q_sample(x_0, t)
                logits = model(x_t, t, basis_idx)
                val_loss = F.cross_entropy(logits.permute(0, 2, 1), x_0)
                total_val_loss += val_loss.item()
        
        avg_train = total_train_loss / max(1, len(train_loader))
        avg_val = total_val_loss / max(1, len(test_loader))
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), f"checkpoints/best_model_n{args.num_qubits}.pt")
            print("  -> Saved Best Model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Can be a folder OR a .pt file
    parser.add_argument('--data_path', type=str, default='dataset_parts') 
    parser.add_argument('--num_qubits', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=1024)
    # Model Args
    parser.add_argument('--num_timesteps', type=int, default=100)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_res_blocks', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    args = parser.parse_args()
    train_model(args)
