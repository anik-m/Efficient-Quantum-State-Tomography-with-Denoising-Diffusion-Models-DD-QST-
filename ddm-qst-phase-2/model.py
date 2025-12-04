import torch
import torch.nn as nn
import numpy as np

class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation [cite: 96]
    Outputs gamma (scale) and beta (shift) based on condition.
    """
    def __init__(self, condition_dim, feature_dim):
        super().__init__()
        self.net = nn.Linear(condition_dim, feature_dim * 2)

    def forward(self, x, condition_emb):
        # condition_emb: (Batch, condition_dim)
        # x: (Batch, feature_dim)
        
        gamma_beta = self.net(condition_emb)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        
        # Affine transformation
        return x * (1 + gamma) + beta

class ResBlock(nn.Module):
    """Residual Block with FiLM conditioning."""
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.film = FiLM(cond_dim, dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.act = nn.SiLU()

    def forward(self, x, cond):
        x_in = x
        x = self.film(x, cond) # Apply conditioning
        x = self.net(x)
        return self.act(x + x_in) # Residual connection

class ConditionalD3PM(nn.Module):
    """
    Discrete Denoising Diffusion Probabilistic Model (Conditional).
    Architecture: Embeddings -> MLP Backbone w/ FiLM -> Logits
    """
    def __init__(self, num_qubits, num_bases, num_timesteps, embed_dim, hidden_dim, num_blocks):
        super().__init__()
        self.num_qubits = num_qubits
        
        # Embeddings
        self.time_emb = nn.Embedding(num_timesteps + 1, embed_dim)
        self.basis_emb = nn.Embedding(num_bases, embed_dim)
        
        # Input Projection: N qubits -> Hidden Dim
        self.input_proj = nn.Linear(num_qubits, hidden_dim)
        
        # Conditional Backbone (ResNet + FiLM)
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim, embed_dim * 2) # Cond dim = Time + Basis
            for _ in range(num_blocks)
        ])
        
        # Output Head: Hidden -> N qubits * 2 states (0/1)
        # Reshaped later to (Batch, N, 2)
        self.output_head = nn.Linear(hidden_dim, num_qubits * 2)

    def forward(self, x, t, basis_idx):
        # x shape: (Batch, N) -> needs float for Linear layer
        x_emb = self.input_proj(x.float())
        
        # Create Conditioning Vector [cite: 85]
        t_emb = self.time_emb(t)
        b_emb = self.basis_emb(basis_idx)
        cond = torch.cat([t_emb, b_emb], dim=1) # Concatenate embeddings
        
        # Pass through backbone
        h = x_emb
        for block in self.blocks:
            h = block(h, cond)
            
        # Output logits
        out = self.output_head(h)
        # Reshape to (Batch, Num_Qubits, 2 classes)
        return out.view(-1, self.num_qubits, 2)
