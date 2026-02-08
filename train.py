import argparse
import torch
from pathlib import Path

from config import Config
from data_loader import RefCOCODataLoader
from model import TransVG
from trainer import Trainer
from logger import get_logger


def build_optimizer(model, config):
    """Build optimizer with different LR for pretrained vs new layers."""
    return torch.optim.AdamW([
        {"params": model.vision_encoder.parameters(), "lr": 1e-5},
        {"params": model.text_encoder.parameters(), "lr": 1e-5},
        {"params": [p for n, p in model.named_parameters()
                    if "vision_encoder" not in n and "text_encoder" not in n],
         "lr": config.learning_rate},
    ], weight_decay=config.weight_decay)


def train(backbone_name, config, device, logger, resume=None):
    """Train a single model."""
    logger.info(f'Training {backbone_name.upper()}')
    
    # Data
    train_loader = RefCOCODataLoader(config.data_dir, config, split='train', shuffle=True)
    val_loader = RefCOCODataLoader(config.data_dir, config, split='val', shuffle=False)
    
    logger.info(f'Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}')
    
    # Model
    model = TransVG(backbone_name, config).to(device)
    logger.info(f'{model}')
    
    # Optimizer
    optimizer = build_optimizer(model, config)
    
    # Trainer
    trainer = Trainer(model, optimizer, config, device, train_loader, val_loader, val_loader.dataset)
    trainer.checkpoint_dir = Path(config.checkpoint_dir) / backbone_name
    trainer.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Resume if specified
    if resume:
        trainer.resume_checkpoint(resume)
    
    trainer.train()
    return model


def main():
    parser = argparse.ArgumentParser(description='Visual Grounding Training')
    parser.add_argument('backbone', type=str, choices=['vit', 'convnext'],
                        help='Vision backbone: vit or convnext')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    args = parser.parse_args()
    
    # Config
    config = Config()
    
    # Override config with args
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create directories
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup logger
    logger = get_logger(name=f'train_{args.backbone}', log_dir=config.log_dir)
    logger.info(f'Device: {device}')
    logger.info(f'Config: epochs={config.num_epochs}, batch_size={config.batch_size}, lr={config.learning_rate}')
    
    # Train
    model = train(args.backbone, config, device, logger, resume=args.resume)


if __name__ == '__main__':
    main()
