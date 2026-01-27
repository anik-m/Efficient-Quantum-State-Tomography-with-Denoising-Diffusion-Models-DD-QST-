import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.nn.functional as F
import argparse, os
from dataset import QuantumStateDataset
from model import ConditionalD3PM
from diffusion import DiscreteDiffusion

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training N={args.num_qubits} on {device} ---")

    # Load Data
    full_ds = QuantumStateDataset(args.data_path, args.num_qubits)
    train_len = int(0.9 * len(full_ds))
    train_ds, val_ds = random_split(full_ds, [train_len, len(full_ds) - train_len])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Init Model
    model = ConditionalD3PM(args.num_qubits, 3**args.num_qubits, 
                           args.num_timesteps, args.embed_dim, 
                           args.hidden_dim, args.num_res_blocks).to(device)
    diffusion = DiscreteDiffusion(model, args.num_timesteps, device)
    opt = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val = float('inf')
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(args.num_epochs):
        model.train()
        train_loss = 0
        for x, basis in train_loader:
            x, basis = x.to(device), basis.to(device)
            t = torch.randint(1, args.num_timesteps+1, (x.size(0),)).to(device)
            x_t = diffusion.q_sample(x, t)
            logits = model(x_t, t, basis)
            loss = F.cross_entropy(logits.permute(0,2,1), x)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, basis in val_loader:
                x, basis = x.to(device), basis.to(device)
                t = torch.randint(1, args.num_timesteps+1, (x.size(0),)).to(device)
                x_t = diffusion.q_sample(x, t)
                logits = model(x_t, t, basis)
                val_loss += F.cross_entropy(logits.permute(0,2,1), x).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Ep {epoch+1} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), f"checkpoints/best_model_n{args.num_qubits}.pt")
            print("  -> Saved Best Model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='dataset_parts')
    parser.add_argument('--num_qubits', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_epochs', type=int, default=30)
    # Architecture defaults from config could go here
    parser.add_argument('--num_timesteps', type=int, default=100)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_res_blocks', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    
    args = parser.parse_args()
    train(args)
