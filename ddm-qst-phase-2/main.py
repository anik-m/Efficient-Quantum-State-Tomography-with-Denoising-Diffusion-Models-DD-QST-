import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Imports from our modules
import config as cfg
from data_gen import generate_synthetic_data
from dataset import QuantumStateDataset
from model import ConditionalD3PM
from diffusion import DiscreteDiffusion
from reconstruct import linear_inversion, get_pauli_matrix
from qiskit.quantum_info import state_fidelity, Statevector

def main():
    # 1. Generate Training Data
    print("--- Phase 2: Initialization ---")
    raw_data, basis_list = generate_synthetic_data()
    
    # 2. Dataset Setup
    dataset = QuantumStateDataset(raw_data, cfg.NUM_QUBITS)
    loader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    print(f"Training on {len(dataset)} shots.")

    # 3. Model Setup
    model = ConditionalD3PM(
        num_qubits=cfg.NUM_QUBITS,
        num_bases=len(basis_list),
        num_timesteps=cfg.NUM_TIMESTEPS,
        embed_dim=cfg.EMBED_DIM,
        hidden_dim=cfg.HIDDEN_DIM,
        num_blocks=cfg.NUM_RES_BLOCKS
    ).to(cfg.DEVICE)
    
    diffusion = DiscreteDiffusion(model, cfg.NUM_TIMESTEPS, cfg.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)

    # 4. Training Loop [cite: 158]
    print(f"--- Starting Training ({cfg.NUM_EPOCHS} Epochs) ---")
    model.train()
    for epoch in range(cfg.NUM_EPOCHS):
        total_loss = 0
        for x_0, basis_idx in loader:
            x_0 = x_0.to(cfg.DEVICE)
            basis_idx = basis_idx.to(cfg.DEVICE)
            
            # Sample random t
            t = torch.randint(1, cfg.NUM_TIMESTEPS + 1, (x_0.shape[0],), device=cfg.DEVICE)
            
            # Forward Diffuse
            x_t = diffusion.q_sample(x_0, t)
            
            # Predict
            pred_logits = model(x_t, t, basis_idx)
            
            # Loss (Cross Entropy) [cite: 125]
            # PyTorch CE expects (Batch, Classes, ...), so permute logits
            loss = F.cross_entropy(pred_logits.permute(0, 2, 1), x_0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}: Loss {total_loss/len(loader):.4f}")

    # 5. Inference (Sampling)
    print("--- Generating Synthetic Samples ---")
    model.eval()
    synthetic_results = {}
    
    for i, basis in enumerate(basis_list):
        # Generate clean samples from noise
        samples = diffusion.p_sample(cfg.SHOTS_INFER, i, cfg.NUM_QUBITS)
        synthetic_results[basis] = samples.cpu().numpy()

    # 6. Reconstruction & Fidelity
    print("--- Reconstructing State ---")
    rho_recon = linear_inversion(synthetic_results, cfg.NUM_QUBITS)
    
    # Define Target for comparison
    if cfg.STATE_TYPE == 'bell':
        # |Phi+> = (|00> + |11>) / sqrt(2)
        target = Statevector.from_label('00') + Statevector.from_label('11')
        target = target / np.sqrt(2)
    
    fid = state_fidelity(target, rho_recon)
    print(f"\nFinal Result for {cfg.NUM_QUBITS}-Qubit {cfg.STATE_TYPE.upper()}:")
    print(f"Reconstruction Fidelity: {fid:.5f}")
    
    if fid > 0.9:
        print("SUCCESS: High fidelity entanglement verification.")
    else:
        print("WARNING: Low fidelity. Check training duration or shot noise.")

if __name__ == "__main__":
    main()
