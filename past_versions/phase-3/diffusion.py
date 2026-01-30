import torch
import torch.nn.functional as F

class DiscreteDiffusion:
    def __init__(self, model, num_timesteps, device):
        self.model = model
        self.num_timesteps = num_timesteps
        self.device = device
        
        # 1. Define One-Step Transition Matrices (Q_t)
        # Linear schedule from beta_min to beta_max
        # Adjusted betas to ensure T=100 reaches full noise
        self.betas = torch.linspace(0.001, 0.1, num_timesteps + 1).to(device)
        
        # 2. Compute Cumulative Transition Matrices (Q_bar_t)
        # Q_bar[t] = Q[t] @ Q[t-1] @ ... @ Q[0]
        self.Q_bar = self._build_cumulative_matrices()

    def _build_cumulative_matrices(self):
        """
        Pre-computes the cumulative noise matrices.
        Q_bar[t] defines the transition P(x_t | x_0).
        """
        Q_bar = torch.zeros(self.num_timesteps + 1, 2, 2).to(self.device)
        
        # Start with Identity (No noise at t=0)
        curr_Q = torch.eye(2).to(self.device)
        Q_bar[0] = curr_Q
        
        for t in range(1, self.num_timesteps + 1):
            b = self.betas[t]
            # One-step matrix Q_t
            # P(0->1) = beta, P(1->0) = beta
            Qt = torch.tensor([[1-b, b], [b, 1-b]]).to(self.device)
            
            # Multiply: New_Total = Current_Total @ One_Step
            # Note: In standard D3PM notation, order is Q_t @ Q_bar_{t-1}
            curr_Q = Qt @ curr_Q
            Q_bar[t] = curr_Q
            
        return Q_bar

    def q_sample(self, x_0, t):
        """
        Forward Process: corrupts x_0 into x_t using Cumulative Q_bar.
        """
        batch, n_qubits = x_0.shape
        x_t = torch.zeros_like(x_0)
        
        # Efficiently sample using the pre-computed Q_bar[t]
        # x_0 is (Batch, N) indices (0 or 1)
        
        for i in range(batch):
            ti = t[i]
            # Get the row of Q_bar corresponding to x_0's value
            # Q_bar[ti] is (2, 2). x_0[i] is (N,).
            # We want probs for each qubit.
            
            # Select probabilities:
            # If bit is 0, probabilities are Q_bar[ti][0]
            # If bit is 1, probabilities are Q_bar[ti][1]
            probs = self.Q_bar[ti][x_0[i]] # Shape (N, 2)
            
            # Sample new state
            x_t[i] = torch.multinomial(probs, 1).squeeze()
            
        return x_t

    @torch.no_grad()
    def p_sample(self, num_samples, basis_idx, num_qubits):
        """
        Reverse Process: Denoising from Pure Noise to Clean Data.
        Uses the D3PM Posterior P(x_{t-1} | x_t, x_0_hat).
        """
        # 1. Start with Uniform Noise (Equal prob of 0 or 1)
        x_t = torch.randint(0, 2, (num_samples, num_qubits)).to(self.device)
        b_vec = torch.full((num_samples,), basis_idx, dtype=torch.long).to(self.device)
        
        for t in reversed(range(1, self.num_timesteps + 1)):
            t_vec = torch.full((num_samples,), t, dtype=torch.long).to(self.device)
            
            # 2. Predict x_0 (Clean Data) from x_t
            logits = self.model(x_t, t_vec, b_vec)
            pred_x0_probs = F.softmax(logits, dim=2) # (B, N, 2)
            
            # 3. Compute Posterior P(x_{t-1} | x_t, x_0_hat)
            # This combines the model's guess with the physics of diffusion
            
            # Get matrices for this step
            Q_t = torch.tensor([[1-self.betas[t], self.betas[t]], 
                                [self.betas[t], 1-self.betas[t]]]).to(self.device)
            Q_bar_tm1 = self.Q_bar[t-1]
            
            # We calculate this for every sample/qubit. 
            # Simplified Logic:
            # P(x_{t-1}=z | x_t, x_0) ~ P(x_t | x_{t-1}=z) * P(x_{t-1}=z | x_0)
            
            # Term A: P(x_t | x_{t-1}) -> Transpose of Q_t (Transition forward)
            # Term B: P(x_{t-1} | x_0) -> Q_bar_{t-1}
            
            # Let's compute probabilities for x_{t-1} being 0 or 1
            
            # Fact_1: P(x_t | x_{t-1}=0) and P(x_t | x_{t-1}=1)
            # This depends on what x_t currently IS.
            x_t_expanded = x_t.unsqueeze(-1) # (B, N, 1)
            # Gather relevant transition probs from Q_t
            # If x_t=0: we want P(0|0) and P(0|1)
            # If x_t=1: we want P(1|0) and P(1|1)
            
            # Term A (Transition):
            # Shape (B, N, 2 states for x_{t-1})
            p_xt_given_xtm1 = torch.zeros(num_samples, num_qubits, 2).to(self.device)
            
            # Fill manually for clarity (optimization possible)
            # Probability that x_{t-1} was k, given x_t
            p_xt_given_xtm1[:,:,0] = Q_t[0, x_t] # P(x_t | 0)
            p_xt_given_xtm1[:,:,1] = Q_t[1, x_t] # P(x_t | 1)
            
            # Term B (Prior from guess): Sum over predicted x_0
            # P(x_{t-1} | x_0_hat) = Sum_x0 P(x_{t-1}|x0) * P(x0)
            # Q_bar_tm1 maps x0 -> x_{t-1}
            # pred_x0_probs is (B, N, 2)
            
            # Matmul: (B, N, 1, 2) @ (2, 2) -> (B, N, 1, 2)
            # We want per-qubit, so efficient batch matmul:
            # P(x_{t-1}) = pred_x0_probs @ Q_bar_{t-1}
            p_xtm1_given_x0 = torch.matmul(pred_x0_probs, self.Q_bar[t-1])
            
            # 4. Combine (Bayes Rule)
            # P(x_{t-1}) ~ Term A * Term B
            unnorm_probs = p_xt_given_xtm1 * p_xtm1_given_x0
            
            # Normalize
            norm_probs = unnorm_probs / (unnorm_probs.sum(dim=-1, keepdim=True) + 1e-8)
            
            # 5. Sample x_{t-1}
            flat_p = norm_probs.view(-1, 2)
            x_t = torch.multinomial(flat_p, 1).view(num_samples, num_qubits)
            
        return x_t
