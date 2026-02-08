import argparse
import torch
from pathlib import Path
from tqdm import tqdm

from config import Config
from data_loader import RefCOCODataLoader
from model import TransVG, compute_iou, xywh2xyxy, sanitize_bbox
from utils import visualize_predictions
from logger import get_logger


def main():
    parser = argparse.ArgumentParser(description='Visual Grounding Inference')
    parser.add_argument('backbone', type=str, choices=['vit', 'convnext'],
                        help='Vision backbone: vit or convnext')
    parser.add_argument('-c', '--checkpoint', type=str, required=True,
                        help='Path to checkpoint')
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='Visualize predictions')
    args = parser.parse_args()
    
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger = get_logger(name=f'test_{args.backbone}', log_dir=config.log_dir)
    logger.info(f'Device: {device}')

    # Data
    val_loader = RefCOCODataLoader(config.data_dir, config, split='val', shuffle=False)
    logger.info(f'Val samples: {len(val_loader.dataset)}')

    # Model
    model = TransVG(args.backbone, config).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)['state_dict'])
    model.eval()
    logger.info(f'Loaded: {args.checkpoint}')

    # Evaluate
    total_iou = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            gt_bbox = batch['bbox'].to(device)

            pred = model(images, input_ids, attention_mask)
            pred_xyxy = sanitize_bbox(xywh2xyxy(pred))
            gt_xyxy = sanitize_bbox(gt_bbox)
            
            total_iou += compute_iou(pred_xyxy, gt_xyxy).sum().item()

    mean_iou = total_iou / len(val_loader.dataset)
    logger.info(f'Mean IoU: {mean_iou:.4f}')

    # Visualize
    if args.visualize:
        save_path = Path('outputs/predictions') / f'{args.backbone}_predictions.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        visualize_predictions(model, val_loader.dataset, config, num_samples=4, save_path=save_path)
        logger.info(f'Saved visualization to {save_path}')


if __name__ == '__main__':
    main()