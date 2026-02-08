"""Base Trainer class."""
import torch
from abc import abstractmethod
from pathlib import Path


class BaseTrainer:
    """Base class for all trainers."""
    
    def __init__(self, model, optimizer, config, device):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device
        
        self.epochs = config.num_epochs
        self.start_epoch = 1
        self.best_metric = 0.0
        
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _train_epoch(self, epoch):
        """Training logic for one epoch. Must return dict with metrics."""
        raise NotImplementedError

    @abstractmethod
    def _valid_epoch(self, epoch):
        """Validation logic for one epoch. Must return dict with metrics."""
        raise NotImplementedError

    def train(self):
        """Full training loop."""
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_log = self._train_epoch(epoch)
            val_log = self._valid_epoch(epoch)
            
            # Print epoch summary
            print(f'Epoch {epoch}: Train Loss={train_log["loss"]:.4f}, IoU={train_log["iou"]:.4f} | '
                  f'Val Loss={val_log["loss"]:.4f}, IoU={val_log["iou"]:.4f}')
            
            # Save best model
            if val_log['iou'] > self.best_metric:
                self.best_metric = val_log['iou']
                self._save_checkpoint(epoch, is_best=True)
                print(f'  ✓ Best IoU: {self.best_metric:.4f}')

        print(f'\nTraining complete. Best IoU: {self.best_metric:.4f}')

    def _save_checkpoint(self, epoch, is_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
        }
        filename = self.checkpoint_dir / f'checkpoint_epoch{epoch}.pth'
        torch.save(state, filename)
        
        if is_best:
            best_path = self.checkpoint_dir / 'model_best.pth'
            torch.save(state, best_path)

    def resume_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_metric = checkpoint['best_metric']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f'Resumed from epoch {checkpoint["epoch"]}')
