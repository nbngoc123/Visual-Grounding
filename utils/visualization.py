"""Visualization utilities."""
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from model import compute_iou, xywh2xyxy, sanitize_bbox


def denormalize_image(img_tensor):
    """Denormalize image tensor for visualization."""
    img = img_tensor.permute(1, 2, 0).numpy()
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    return np.clip(img, 0, 1)


def visualize_predictions(model, dataset, config, num_samples=4, save_path=None):
    """Visualize model predictions."""
    device = next(model.parameters()).device
    model.eval()

    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        sample = dataset[idx]

        image = sample['image'].unsqueeze(0).to(device)
        input_ids = sample['input_ids'].unsqueeze(0).to(device)
        attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
        gt_bbox = sample['bbox'].numpy()

        with torch.no_grad():
            pred_xywh = model(image, input_ids, attention_mask)[0]
            pred_xyxy = sanitize_bbox(xywh2xyxy(pred_xywh.unsqueeze(0)))[0].cpu().numpy()

        img = denormalize_image(sample['image'])
        iou = compute_iou(torch.tensor([pred_xyxy]), torch.tensor([gt_bbox]))[0].item()

        # Plot GT
        axes[i, 0].imshow(img)
        x1, y1, x2, y2 = gt_bbox * config.img_size
        axes[i, 0].add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                             linewidth=2, edgecolor='green', facecolor='none'))
        axes[i, 0].set_title(f'GT: {sample["text"][:35]}...', fontsize=9)
        axes[i, 0].axis('off')

        # Plot Prediction
        axes[i, 1].imshow(img)
        x1, y1, x2, y2 = pred_xyxy * config.img_size
        axes[i, 1].add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1,
                             linewidth=2, edgecolor='blue', facecolor='none'))
        axes[i, 1].set_title(f'Pred (IoU: {iou:.3f})', fontsize=9)
        axes[i, 1].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_training_curves(histories, names):
    """Plot training curves for multiple models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for history, name in zip(histories, names):
        epochs = range(1, len(history['train_loss']) + 1)
        axes[0].plot(epochs, history['train_loss'], label=f'{name} Train')
        axes[0].plot(epochs, history['val_loss'], '--', label=f'{name} Val')
        axes[1].plot(epochs, history['train_iou'], label=f'{name} Train')
        axes[1].plot(epochs, history['val_iou'], '--', label=f'{name} Val')

    for ax, title, ylabel in zip(axes, ['Loss', 'IoU'], ['Loss', 'IoU']):
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
