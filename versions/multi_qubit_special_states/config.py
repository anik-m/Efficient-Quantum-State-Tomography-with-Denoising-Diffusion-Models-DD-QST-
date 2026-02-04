# Dictionary of default settings
# These are used if no command-line arguments are provided.
DEFAULTS = {
    # Physics
    'num_qubits': 2,
    'state_type': 'bell',      # 'plus', 'bell', 'ghz'
    
    # Diffusion
    'num_timesteps': 100,
    
    # Architecture
    'embed_dim': 64,
    'hidden_dim': 512,
    'num_res_blocks': 4,
    
    # Training
    'batch_size': 256,
    'learning_rate': 1e-4,
    'num_epochs': 300,
    
    # Data
    'shots_train': 1000,
    'shots_infer': 10000
}
# import torch
#
# # --- Physics / System Settings ---
# NUM_QUBITS = 2              # Phase 2 Target: Bell State
# STATE_TYPE = 'bell'         # Options: 'plus', 'bell', 'ghz'
# BASIS_SET = ['X', 'Y', 'Z'] # Pauli Basis
#
# # --- Diffusion Settings ---
# NUM_TIMESTEPS = 100         # T steps
# DIFFUSION_SCHEDULE = 'linear' 
#
# # --- Model Architecture ---
# EMBED_DIM = 64              # Dimension for Time & Basis embeddings
# HIDDEN_DIM = 512            # Width of the ResNet Backbone 
# NUM_RES_BLOCKS = 4          # Depth of the network
#
# # --- Training Hyperparameters ---
# BATCH_SIZE = 256
# LEARNING_RATE = 1e-4        # AdamW LR
# NUM_EPOCHS = 300
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#
# # --- Data Generation ---
# SHOTS_TRAIN = 1000          # "Medium" dataset
# SHOTS_INFER = 10000         # Synthetic samples > Experimental shots
