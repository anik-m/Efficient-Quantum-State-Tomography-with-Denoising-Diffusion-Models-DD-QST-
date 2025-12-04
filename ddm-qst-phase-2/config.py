import torch

# --- Physics / System Settings ---
NUM_QUBITS = 2              # Phase 2 Target: Bell State
STATE_TYPE = 'bell'         # Options: 'plus', 'bell', 'ghz'
BASIS_SET = ['X', 'Y', 'Z'] # Pauli Basis

# --- Diffusion Settings ---
NUM_TIMESTEPS = 100         # T steps
DIFFUSION_SCHEDULE = 'linear' 

# --- Model Architecture ---
EMBED_DIM = 64              # Dimension for Time & Basis embeddings
HIDDEN_DIM = 512            # Width of the ResNet Backbone 
NUM_RES_BLOCKS = 4          # Depth of the network

# --- Training Hyperparameters ---
BATCH_SIZE = 256
LEARNING_RATE = 1e-4        # AdamW LR
NUM_EPOCHS = 300
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Data Generation ---
SHOTS_TRAIN = 1000          # "Medium" dataset
SHOTS_INFER = 10000         # Synthetic samples > Experimental shots
