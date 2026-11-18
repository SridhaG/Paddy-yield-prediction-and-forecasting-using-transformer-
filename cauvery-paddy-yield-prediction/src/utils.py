"""
Utility functions for the Cauvery Paddy Yield Prediction project.
"""

import os
import json
import random
import numpy as np
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def compute_metrics(y_true, y_pred):
    """Compute regression metrics.
    
    Returns dict with R2, RMSE, MAE, MAPE.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    # MAPE: avoid division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    return {'R2': round(r2, 4), 'RMSE': round(rmse, 2), 
            'MAE': round(mae, 2), 'MAPE': round(mape, 2)}


def save_checkpoint(model, optimizer, epoch, metrics, path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'metrics': metrics,
    }, path)


def load_checkpoint(path, model, optimizer=None):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and checkpoint.get('optimizer_state_dict'):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=15, min_delta=0.0, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
    
    def load_best(self, model):
        """Load the best model state."""
        if self.best_model_state:
            model.load_state_dict(self.best_model_state)


def save_json(data, path):
    """Save data as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path):
    """Load JSON data."""
    with open(path, 'r') as f:
        return json.load(f)
