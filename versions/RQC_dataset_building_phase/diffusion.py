import torch
import torch.nn.functional as F
import numpy as np

class DiscreteDiffusion:
    def __init__(self, model, num_timesteps, device):
        self.model = model
        self.num_timesteps = num_timesteps
        self.device = device
        
        # FIX: Switch to Cosine Schedule (Smoother noise)
        self.betas = self._get_cosine_schedule(num_timesteps).to(device)
        self.Q_bar = self._build_cumulative_matrices()

    def _get_cosine_schedule(self, num_timesteps):
        """
        Cosine schedule as proposed by Nichol & Dhariwal (2021).
        Prevents the signal from being destroyed too early.
        """
        steps = np.arange(num_timesteps + 1, dtype=np.float64) / num_timesteps
        alpha_bar = np.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        
        betas = []
        for i in range(1, num_timesteps + 1):
            t = i
            # Calculate beta_t from alpha_bar
            b = min(1 - alpha_bar[t] / alpha_bar[t-1], 0.999)
            betas.append(b)
            
        return torch.tensor([0.0] + betas, dtype=torch.float32)

    def _build_cumulative_matrices(self):
        Q_bar = torch.zeros(self.num_timesteps + 1, 2, 2).to(self.device)
        curr_Q = torch.eye(2).to(self.device)
        Q_bar[0] = curr_Q
        
        for t in range(1, self.num_timesteps + 1):
            b = self.betas[t]
            Qt = torch.tensor([[1-b, b], [b, 1-b]]).to(self.device)
            curr_Q = Qt @ curr_Q
            Q_bar[t] = curr_Q
        return Q_bar

    def q_sample(self, x_0, t):
        batch = x_0.shape[0]
        x_t = torch.zeros_like(x_0)
        for i in range(batch):
            probs = self.Q_bar[t[i]][x_0[i]]
            x_t[i] = torch.multinomial(probs, 1).squeeze()
        return x_t

    @torch.no_grad()
    def p_sample(self, num_samples, basis_idx, num_qubits):
        x_t = torch.randint(0, 2, (num_samples, num_qubits)).to(self.device)
        b_vec = torch.full((num_samples,), basis_idx, dtype=torch.long).to(self.device)
        
        for t in reversed(range(1, self.num_timesteps + 1)):
            t_vec = torch.full((num_samples,), t, dtype=torch.long).to(self.device)
            
            logits = self.model(x_t, t_vec, b_vec)
            pred_x0_probs = F.softmax(logits, dim=2)
            
            beta = self.betas[t]
            
            # Transition P(x_t | x_{t-1})
            prob_trans = torch.zeros(num_samples, num_qubits, 2).to(self.device)
            prob_trans[:,:,0] = torch.where(x_t==0, 1-beta, beta)
            prob_trans[:,:,1] = torch.where(x_t==0, beta, 1-beta)
            
            # Prior P(x_{t-1} | x_0_hat)
            prob_prior = torch.matmul(pred_x0_probs, self.Q_bar[t-1])
            
            # Combine
            unnorm = prob_trans * prob_prior
            norm = unnorm / (unnorm.sum(dim=-1, keepdim=True) + 1e-8)
            
            flat_p = norm.view(-1, 2)
            x_t = torch.multinomial(flat_p, 1).view(num_samples, num_qubits)
        return x_t
