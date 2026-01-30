import torch
import torch.nn as nn

class FiLM(nn.Module):
    def __init__(self, cond_dim, feat_dim):
        super().__init__()
        self.net = nn.Linear(cond_dim, feat_dim * 2)

    def forward(self, x, cond):
        gamma, beta = self.net(cond).chunk(2, dim=1)
        return x * (1 + gamma) + beta

class ResBlock(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.film = FiLM(cond_dim, dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.act = nn.SiLU()

    def forward(self, x, cond):
        return self.act(x + self.net(self.film(x, cond)))

class ConditionalD3PM(nn.Module):
    def __init__(self, num_qubits, num_bases, num_timesteps, embed_dim, hidden_dim, num_blocks):
        super().__init__()
        self.num_qubits = num_qubits
        
        # 1. NEW: Input Embedding (Treats 0/1 as distinct tokens)
        self.x_emb = nn.Embedding(2, embed_dim) # 0 -> Vector, 1 -> Vector
        
        # 2. Project Flattened Embedding to Hidden
        # Input: (Batch, N, Embed) -> Flatten -> (Batch, N*Embed)
        self.input_proj = nn.Linear(num_qubits * embed_dim, hidden_dim)
        
        # Condition Embeddings
        self.time_emb = nn.Embedding(num_timesteps + 1, embed_dim)
        self.basis_emb = nn.Embedding(num_bases, embed_dim)
        
        # Backbone
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim, embed_dim * 2) 
            for _ in range(num_blocks)
        ])
        
        # Output Head
        self.output_head = nn.Linear(hidden_dim, num_qubits * 2)

    def forward(self, x, t, basis_idx):
        # x: (Batch, N) (LongTensor)
        
        # 1. Embed Inputs
        x = self.x_emb(x) # (Batch, N, Embed)
        x = x.view(x.size(0), -1) # Flatten -> (Batch, N*Embed)
        x = self.input_proj(x) # -> (Batch, Hidden)
        
        # 2. Embed Conditions
        t_emb = self.time_emb(t)
        b_emb = self.basis_emb(basis_idx)
        cond = torch.cat([t_emb, b_emb], dim=1)
        
        # 3. Backbone
        for block in self.blocks:
            x = block(x, cond)
            
        # 4. Reshape Output
        # Return (Batch, N, 2)
        return self.output_head(x).view(-1, self.num_qubits, 2)
