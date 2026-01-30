# Dictionary of default settings
# These are used if no command-line arguments are provided.
DEFAULTS = {
    # Physics
    'num_qubits': 3,
    'basis_set': ['X', 'Y', 'Z'],
    
    # Architecture
    'num_timesteps': 100,
    'embed_dim': 128,
    'hidden_dim': 512,
    'num_res_blocks': 4,
    
    # Training
    'batch_size': 1024,
    'learning_rate': 1e-3,
    'num_epochs': 30,
    
    # Inference / Plotting
    'shots_infer': 5000,  # Shots for reconstruction
    'out_dir': 'results_plots'
}
