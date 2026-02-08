"""Loss functions for Visual Grounding."""
import torch
import torch.nn as nn


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)


def sanitize_bbox(boxes):
    """Ensure valid bbox: x1 < x2, y1 < y2, in [0,1]."""
    x1, x2 = torch.min(boxes[:, 0], boxes[:, 2]), torch.max(boxes[:, 0], boxes[:, 2])
    y1, y2 = torch.min(boxes[:, 1], boxes[:, 3]), torch.max(boxes[:, 1], boxes[:, 3])
    return torch.stack([x1, y1, x2, y2], dim=1).clamp(0, 1)


def xywh2xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)


def xyxy2xywh(boxes):
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], dim=-1)


def giou_loss(pred, gt, eps=1e-6):
    """Generalized IoU loss."""
    inter_x1 = torch.max(pred[:, 0], gt[:, 0])
    inter_y1 = torch.max(pred[:, 1], gt[:, 1])
    inter_x2 = torch.min(pred[:, 2], gt[:, 2])
    inter_y2 = torch.min(pred[:, 3], gt[:, 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    union = box_area(pred) + box_area(gt) - inter + eps
    iou = inter / union

    encl_x1 = torch.min(pred[:, 0], gt[:, 0])
    encl_y1 = torch.min(pred[:, 1], gt[:, 1])
    encl_x2 = torch.max(pred[:, 2], gt[:, 2])
    encl_y2 = torch.max(pred[:, 3], gt[:, 3])
    encl_area = (encl_x2 - encl_x1).clamp(min=0) * (encl_y2 - encl_y1).clamp(min=0) + eps

    return (1 - (iou - (encl_area - union) / encl_area)).mean()


class VisualGroundingLoss(nn.Module):
    """Combined L1 + GIoU loss (TransVG style)."""
    
    def __init__(self, l1_weight=1.0, giou_weight=2.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.giou_weight = giou_weight
        self.l1_loss = nn.L1Loss()

    def forward(self, pred_xywh, gt_xyxy):
        pred_xyxy = sanitize_bbox(xywh2xyxy(pred_xywh))
        gt_xyxy = sanitize_bbox(gt_xyxy)
        
        l1 = self.l1_loss(xyxy2xywh(pred_xyxy), xyxy2xywh(gt_xyxy))
        giou = giou_loss(pred_xyxy, gt_xyxy)
        
        return {
            'loss': self.l1_weight * l1 + self.giou_weight * giou,
            'l1': l1,
            'giou': giou,
            'pred_xyxy': pred_xyxy,
            'gt_xyxy': gt_xyxy
        }
