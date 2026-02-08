"""Configuration for Visual Grounding."""

class Config:
    # Data
    data_dir = './data'
    lmdb_dir = './data/lmdb/refcoco'
    
    # Model
    img_size = 224
    max_text_len = 16
    hidden_dim = 256
    text_encoder = 'bert-base-uncased'
    
    # Training
    batch_size = 32
    num_workers = 0
    num_epochs = 20
    learning_rate = 1e-3
    weight_decay = 0.05
    
    # Paths
    checkpoint_dir = './checkpoints'
    log_dir = './outputs/logs'