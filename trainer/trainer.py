"""Trainer for Visual Grounding."""
import torch
from tqdm import tqdm

from base import BaseTrainer
from model import VisualGroundingLoss, compute_iou


class Trainer(BaseTrainer):
    """Trainer for Visual Grounding models."""
    
    def __init__(self, model, optimizer, config, device, train_loader, val_loader, val_dataset):
        super().__init__(model, optimizer, config, device)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_dataset = val_dataset
        self.criterion = VisualGroundingLoss()

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss, total_iou = 0.0, 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.epochs}')
        for batch in pbar:
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            gt_bbox = batch['bbox'].to(self.device)

            self.optimizer.zero_grad()
            pred_bbox = self.model(images, input_ids, attention_mask)
            
            loss_dict = self.criterion(pred_bbox, gt_bbox)
            loss_dict['loss'].backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.2)
            self.optimizer.step()

            iou = compute_iou(loss_dict['pred_xyxy'], loss_dict['gt_xyxy']).mean().item()
            total_loss += loss_dict['loss'].item()
            total_iou += iou
            
            pbar.set_postfix({'loss': f'{loss_dict["loss"].item():.4f}', 'iou': f'{iou:.4f}'})

        n = len(self.train_loader)
        return {'loss': total_loss / n, 'iou': total_iou / n}

    def _valid_epoch(self, epoch):
        self.model.eval()
        total_loss, total_iou = 0.0, 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                gt_bbox = batch['bbox'].to(self.device)

                pred_bbox = self.model(images, input_ids, attention_mask)
                loss_dict = self.criterion(pred_bbox, gt_bbox)
                
                total_loss += loss_dict['loss'].item()
                total_iou += compute_iou(loss_dict['pred_xyxy'], loss_dict['gt_xyxy']).sum().item()

        return {
            'loss': total_loss / len(self.val_loader),
            'iou': total_iou / len(self.val_dataset)
        }
