import torch
import torch.nn.functional as F

class DiscreteDiffusion:
    def __init__(self, model, num_timesteps, device):
        self.model = model
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Define Transition Matrices [cite: 113, 248]
        # Q_t: Matrix where Q[row_i, col_j] = P(x_t = i | x_{t-1} = j)
        self.Q = self._build_transition_matrices()

    def _build_transition_matrices(self):
        """Constructs the bit-flip transition matrices for all t."""
        Q = torch.zeros(self.num_timesteps + 1, 2, 2).to(self.device)
        # Linear schedule from identity to uniform noise
        betas = torch.linspace(0.001, 0.5, self.num_timesteps + 1).to(self.device)
        
        for t in range(self.num_timesteps + 1):
            b = betas[t]
            # [ P(0|0)  P(0|1) ]
            # [ P(1|0)  P(1|1) ]
            Q[t] = torch.tensor([[1-b, b], [b, 1-b]])
        return Q

    def q_sample(self, x_0, t):
        """
        Forward Process (Noise Injection).
        Efficiently samples x_t given x_0 and transition matrix Q_t.
        x_0 shape: (Batch, N_Qubits)
        """
        batch_size, n_qubits = x_0.shape
        Q_t = self.Q[t] # (Batch, 2, 2)
        
        noisy_data = []
        # Apply noise independently to each qubit
        for i in range(n_qubits):
            # Extract specific qubit column
            qubit_x0 = x_0[:, i] # (Batch,)
            
            # Select probabilities based on x0 value
            # If x0=0, take col 0. If x0=1, take col 1.
            # Using gather for efficient indexing
            # Q_t shape: (Batch, 2_out, 2_in)
            probs = Q_t.gather(2, qubit_x0.view(-1, 1, 1).expand(-1, 2, 1)).squeeze(2)
            
            # Sample
            qubit_xt = torch.multinomial(probs, 1)
            noisy_data.append(qubit_xt)
            
        return torch.cat(noisy_data, dim=1) # (Batch, N)

    @torch.no_grad()
    def p_sample(self, num_samples, basis_idx, num_qubits):
        """Reverse Process (Generation)[cite: 166]."""
        # Start with uniform noise
        x_t = torch.randint(0, 2, (num_samples, num_qubits)).to(self.device)
        b_vec = torch.full((num_samples,), basis_idx, dtype=torch.long).to(self.device)
        
        for t in reversed(range(1, self.num_timesteps + 1)):
            t_vec = torch.full((num_samples,), t, dtype=torch.long).to(self.device)
            
            # 1. Model predicts distribution of x_0 (clean state)
            logits = self.model(x_t, t_vec, b_vec) # (B, N, 2)
            probs_x0 = F.softmax(logits, dim=2)
            
            # 2. Sample predicted x_0
            # Reshape for multinomial: (B*N, 2)
            flat_probs = probs_x0.view(-1, 2)
            x_0_hat = torch.multinomial(flat_probs, 1).view(num_samples, num_qubits)
            
            # 3. Post-process (D3PM reverse step)
            # In standard D3PM, we posterior sample. 
            # For this PoC, we use the "predict x0 and re-noise" approximation
            if t > 1:
                # Add noise back to reach t-1
                x_t = self.q_sample(x_0_hat, torch.full_like(t_vec, t-1))
            else:
                x_t = x_0_hat
                
        return x_t
