import random
import numpy as np
import torch
import os
from typing import Optional


def get_best_device():
    """
    Automatically detect and return the best available device for training.
    Priority: CUDA > MPS > CPU

    Returns:
        torch.device: The best available device
    """

    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 Using CUDA device: {gpu_name} ({gpu_memory:.1f}GB VRAM)")
        return device


    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🍎 Using MPS device: Apple Silicon GPU")
        return device


    else:
        device = torch.device('cpu')
        print("💻 Using CPU device")
        return device


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EarlyStopper:
    """
    Early stopping utility to prevent overfitting.
    Based on IAM_ADMET_AI implementation.
    """

    def __init__(self, mode: str = 'min', patience: int = 10, min_delta: float = 0.0):
        """
        Args:
            mode: 'min' for loss (lower is better) or 'max' for metric (higher is better)
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
        """
        assert mode in ['max', 'min']
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.num_bad = 0
        self.best_epoch = 0

    def step(self, value: float, epoch: int = 0) -> bool:
        """
        Check if training should stop.

        Args:
            value: Current metric/loss value
            epoch: Current epoch number

        Returns:
            True if training should stop, False otherwise
        """
        if self.best is None:
            self.best = value
            self.num_bad = 0
            self.best_epoch = epoch
            return False

        if self.mode == 'max':
            improved = value > (self.best + self.min_delta)
        else:
            improved = value < (self.best - self.min_delta)

        if improved:
            self.best = value
            self.num_bad = 0
            self.best_epoch = epoch
            return False

        self.num_bad += 1
        return self.num_bad >= self.patience


def compute_metric(task: str, y_true, y_pred, official: str = 'mae'):
    """
    Simple metric computation function (IAM_ADMET_AI pattern).
    Returns a single metric value for Optuna optimization.

    Args:
        task: 'regression' or 'binary'
        y_true: True labels
        y_pred: Predicted values
        official: Metric name ('mae', 'spearman', 'auroc', 'auprc')

    Returns:
        Single metric value (float)
    """

    if hasattr(y_true, 'detach'):
        y_true_np = y_true.detach().cpu().numpy()
    else:
        y_true_np = np.array(y_true)

    if hasattr(y_pred, 'detach'):
        y_pred_np = y_pred.detach().cpu().numpy()
    else:
        y_pred_np = np.array(y_pred)

    y_true_np = np.array(y_true_np).flatten()
    y_pred_np = np.array(y_pred_np).flatten()

    if task == 'binary':

        unique_labels = np.unique(y_true_np)
        if len(unique_labels) < 2:
            return 0.0


        probs = 1 / (1 + np.exp(-y_pred_np))
        if official.lower() == 'auprc':
            from sklearn.metrics import average_precision_score
            return average_precision_score(y_true_np, probs)
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true_np, probs)
    else:
        if official.lower() == 'spearman':
            from scipy.stats import spearmanr
            return spearmanr(y_true_np, y_pred_np).correlation

        return np.mean(np.abs(y_true_np - y_pred_np))


def compute_metrics(y_true, y_pred, task_type: str = 'regression'):
    """
    Compute evaluation metrics for regression or binary classification.
    Based on IAM_ADMET_AI metrics implementation.

    Args:
        y_true: True labels
        y_pred: Predicted values
        task_type: 'regression' or 'binary'

    Returns:
        Dictionary of metrics
    """

    if hasattr(y_true, 'detach'):
        y_true = y_true.detach().cpu().numpy()
    if hasattr(y_pred, 'detach'):
        y_pred = y_pred.detach().cpu().numpy()

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    if task_type == 'binary':
        from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score


        unique_labels = np.unique(y_true)
        if len(unique_labels) < 2:
            return {
                'accuracy': 0.0,
                'auroc': 0.0,
                'auprc': 0.0,
                'f1': 0.0
            }


        probs = 1 / (1 + np.exp(-y_pred))
        preds = (probs > 0.5).astype(int)

        return {
            'accuracy': accuracy_score(y_true, preds),
            'auroc': roc_auc_score(y_true, probs),
            'auprc': average_precision_score(y_true, probs),
            'f1': f1_score(y_true, preds)
        }
    else:
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_error, r2_score,
            mean_absolute_percentage_error, max_error, explained_variance_score,
            median_absolute_error
        )
        from scipy.stats import spearmanr, pearsonr


        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)


        try:
            mape = mean_absolute_percentage_error(y_true, y_pred)
        except:

            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

        try:
            max_err = max_error(y_true, y_pred)
        except:
            max_err = np.max(np.abs(y_true - y_pred))

        try:
            explained_variance = explained_variance_score(y_true, y_pred)
        except:
            explained_variance = 0.0

        try:
            medae = median_absolute_error(y_true, y_pred)
        except:
            medae = np.median(np.abs(y_true - y_pred))


        try:
            spearman_corr = spearmanr(y_true, y_pred).correlation
        except:
            spearman_corr = 0.0

        try:
            pearson_corr, pearson_p = pearsonr(y_true, y_pred)
        except:
            pearson_corr = 0.0
            pearson_p = 1.0


        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        std_true = np.std(y_true)
        std_pred = np.std(y_pred)



        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


        q33 = np.percentile(y_true, 33.33)
        q66 = np.percentile(y_true, 66.67)


        y_true_binned = np.zeros_like(y_true, dtype=int)
        y_true_binned[y_true > q66] = 2
        y_true_binned[(y_true > q33) & (y_true <= q66)] = 1
        y_true_binned[y_true <= q33] = 0


        y_pred_binned = np.zeros_like(y_pred, dtype=int)
        y_pred_binned[y_pred > q66] = 2
        y_pred_binned[(y_pred > q33) & (y_pred <= q66)] = 1
        y_pred_binned[y_pred <= q33] = 0


        try:
            accuracy = accuracy_score(y_true_binned, y_pred_binned)

            precision = precision_score(y_true_binned, y_pred_binned, average='macro', zero_division=0)
            recall = recall_score(y_true_binned, y_pred_binned, average='macro', zero_division=0)
            f1 = f1_score(y_true_binned, y_pred_binned, average='macro', zero_division=0)
        except:
            accuracy = 0.0
            precision = 0.0
            recall = 0.0
            f1 = 0.0

        return {

            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,


            'mape': mape,
            'max_error': max_err,
            'median_ae': medae,


            'spearman': spearman_corr,
            'pearson': pearson_corr,
            'pearson_p': pearson_p,


            'explained_variance': explained_variance,


            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,


            'mean_true': mean_true,
            'mean_pred': mean_pred,
            'std_true': std_true,
            'std_pred': std_pred
        }


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, epoch, metrics, filepath, config=None):
    """
    Save model checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        metrics: Dictionary of metrics
        filepath: Path to save checkpoint
        config: Optional configuration dictionary
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None, device=None):
    """
    Load model checkpoint.

    Args:
        filepath: Path to checkpoint file
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to load checkpoint on

    Returns:
        Dictionary with checkpoint information
    """
    if device is None:
        device = torch.device('cpu')

    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint

