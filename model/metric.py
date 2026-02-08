"""Metrics for Visual Grounding."""
import torch


def compute_iou(pred, gt, eps=1e-6):
    """Compute IoU between predicted and ground truth bboxes (xyxy format)."""
    inter_x1 = torch.max(pred[:, 0], gt[:, 0])
    inter_y1 = torch.max(pred[:, 1], gt[:, 1])
    inter_x2 = torch.min(pred[:, 2], gt[:, 2])
    inter_y2 = torch.min(pred[:, 3], gt[:, 3])
    
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    
    pred_area = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
    gt_area = (gt[:, 2] - gt[:, 0]).clamp(min=0) * (gt[:, 3] - gt[:, 1]).clamp(min=0)
    
    return inter / (pred_area + gt_area - inter + eps)


def accuracy_at_iou(pred, gt, threshold=0.5):
    """Accuracy at IoU threshold."""
    return (compute_iou(pred, gt) >= threshold).float().mean().item()
