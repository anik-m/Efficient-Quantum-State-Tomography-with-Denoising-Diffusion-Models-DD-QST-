import torch
import torch.nn.functional as F

class DiscreteDiffusion:
    def __init__(self, model, num_timesteps, device):
        self.model = model
        self.T = num_timesteps
        self.device = device
        # Linear Beta Schedule
        betas = torch.linspace(0.001, 0.2, num_timesteps + 1).to(device)
        self.Q = torch.zeros(num_timesteps + 1, 2, 2).to(device)
        for t in range(num_timesteps + 1):
            b = betas[t]
            self.Q[t] = torch.tensor([[1-b, b], [b, 1-b]])

    def q_sample(self, x_0, t):
        # Apply noise matrix Q_t
        batch, n = x_0.shape
        # Gather Q matrices for each t in batch
        # Simplified implementation: iterate (optimization possible)
        x_t = x_0.clone()
        for i in range(batch):
            probs = self.Q[t[i]][x_0[i]] # (N, 2)
            x_t[i] = torch.multinomial(probs, 1).squeeze()
        return x_t

    @torch.no_grad()
    def p_sample(self, num_samples, basis_idx, num_qubits):
        # Reverse Process (Generation)
        x = torch.randint(0, 2, (num_samples, num_qubits)).to(self.device)
        basis = torch.full((num_samples,), basis_idx, dtype=torch.long).to(self.device)
        
        for t in reversed(range(1, self.T + 1)):
            time = torch.full((num_samples,), t, dtype=torch.long).to(self.device)
            logits = self.model(x, time, basis)
            probs = F.softmax(logits, dim=2) # (B, N, 2)
            
            # Sample next step
            flat_probs = probs.view(-1, 2)
            x = torch.multinomial(flat_probs, 1).view(num_samples, num_qubits)
        return x
