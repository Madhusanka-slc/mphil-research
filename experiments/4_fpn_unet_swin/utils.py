"""
Utility functions for vessel segmentation training and evaluation.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


class DiceLoss(nn.Module):
    """Dice Loss for segmentation."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)

        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross Entropy and Dice Loss."""
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(smooth)

    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice


def compute_metrics(pred_binary, target, smooth=1e-7):
    """
    Compute segmentation metrics.

    Args:
        pred_binary: Binary predictions (0 or 1)
        target: Ground truth binary mask
        smooth: Smoothing factor for numerical stability

    Returns:
        Dictionary with Dice, IoU, Sensitivity, Specificity, Precision
    """
    pred_flat = pred_binary.flatten()
    target_flat = target.flatten()

    # True Positives, False Positives, True Negatives, False Negatives
    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum()
    fn = ((1 - pred_flat) * target_flat).sum()

    # Dice Similarity Coefficient
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)

    # Intersection over Union
    iou = (tp + smooth) / (tp + fp + fn + smooth)

    # Sensitivity (Recall / True Positive Rate)
    sensitivity = (tp + smooth) / (tp + fn + smooth)

    # Specificity (True Negative Rate)
    specificity = (tn + smooth) / (tn + fp + smooth)

    # Precision
    precision = (tp + smooth) / (tp + fp + smooth)

    return {
        'dice': dice.item() if isinstance(dice, torch.Tensor) else dice,
        'iou': iou.item() if isinstance(iou, torch.Tensor) else iou,
        'sensitivity': sensitivity.item() if isinstance(sensitivity, torch.Tensor) else sensitivity,
        'specificity': specificity.item() if isinstance(specificity, torch.Tensor) else specificity,
        'precision': precision.item() if isinstance(precision, torch.Tensor) else precision,
        'tp': int(tp.item()) if isinstance(tp, torch.Tensor) else int(tp),
        'fp': int(fp.item()) if isinstance(fp, torch.Tensor) else int(fp),
        'tn': int(tn.item()) if isinstance(tn, torch.Tensor) else int(tn),
        'fn': int(fn.item()) if isinstance(fn, torch.Tensor) else int(fn),
    }


def compute_batch_metrics(predictions, targets, threshold=0.5):
    """
    Compute metrics for a batch of images.

    Args:
        predictions: Logits or probabilities from model
        targets: Ground truth masks
        threshold: Threshold for binarization

    Returns:
        Aggregated metrics
    """
    pred_binary = (torch.sigmoid(predictions) > threshold).float()

    metrics_list = []
    for pred, target in zip(pred_binary, targets):
        metrics = compute_metrics(pred, target)
        metrics_list.append(metrics)

    # Average metrics
    avg_metrics = {}
    for key in metrics_list[0].keys():
        if key not in ['tp', 'fp', 'tn', 'fn']:
            values = [m[key] for m in metrics_list]
            avg_metrics[key] = np.mean(values)
            avg_metrics[f'{key}_std'] = np.std(values)

    return avg_metrics


def plot_training_history(history, save_path=None):
    """
    Plot training history.

    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'val_dice'
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(history['train_loss'], label='Train', marker='o')
    axes[0].plot(history['val_loss'], label='Val', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training/Validation Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Dice
    axes[1].plot(history['val_dice'], label='Validation Dice', marker='o', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].set_title('Validation Dice')
    axes[1].grid(alpha=0.3)

    # Learning rate (if available)
    if 'lr' in history:
        axes[2].plot(history['lr'], marker='o', color='red')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        axes[2].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_confusion_matrix(tn, fp, fn, tp, save_path=None):
    """
    Plot confusion matrix for binary segmentation.

    Args:
        tn, fp, fn, tp: Confusion matrix counts
        save_path: Path to save figure
    """
    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    ax.set_xticklabels(['Background', 'Vessel'])
    ax.set_yticklabels(['Background', 'Vessel'])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_segmentation_examples(images, targets, predictions, num_examples=3, save_path=None):
    """
    Plot example segmentations.

    Args:
        images: Input images [B, 1, H, W]
        targets: Ground truth masks [B, 1, H, W]
        predictions: Model predictions [B, 1, H, W]
        num_examples: Number of examples to show
        save_path: Path to save figure
    """
    num_show = min(num_examples, len(images))

    fig, axes = plt.subplots(num_show, 3, figsize=(12, 4*num_show))
    if num_show == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_show):
        # Input image
        img = images[i, 0].cpu().numpy()
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f'Input {i+1}')
        axes[i, 0].axis('off')

        # Ground truth
        gt = targets[i, 0].cpu().numpy()
        axes[i, 1].imshow(gt, cmap='gray')
        axes[i, 1].set_title(f'Ground Truth {i+1}')
        axes[i, 1].axis('off')

        # Prediction
        pred = torch.sigmoid(predictions[i, 0]).detach().cpu().numpy()
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title(f'Prediction {i+1}')
        axes[i, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


class AverageMeter:
    """Compute and store average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping helper."""
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

    def is_better(self, val_loss):
        return val_loss < self.best_loss - self.min_delta


if __name__ == '__main__':
    # Test utilities
    print("Testing utils...")

    # Test metrics
    pred = torch.randint(0, 2, (10, 10))
    target = torch.randint(0, 2, (10, 10))
    metrics = compute_metrics(pred, target)
    print("\nMetrics:", metrics)

    # Test AverageMeter
    meter = AverageMeter()
    for i in range(5):
        meter.update(0.5 + 0.1*i)
    print(f"\nAverageMeter - avg: {meter.avg:.4f}, val: {meter.val:.4f}")

    # Test EarlyStopping
    es = EarlyStopping(patience=3)
    losses = [0.5, 0.4, 0.35, 0.36, 0.37, 0.38, 0.39]
    for loss in losses:
        es(loss)
        print(f"Loss: {loss:.4f}, Stop: {es.early_stop}")

    print("\n✓ All utilities working!")
