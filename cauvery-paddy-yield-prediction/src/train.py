"""
Training pipeline for all models in the Cauvery Delta paddy yield prediction project.

Models trained:
1. AW-MIL Temporal Transformer (main model)
2. LSTM Baseline
3. Random Forest
4. XGBoost

Usage:
    python src/train.py --model all
    python src/train.py --model transformer --epochs 200
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from models import AWMILTransformer, LSTMBaseline, extract_tabular_features
from utils import set_seed, get_device, compute_metrics, EarlyStopping, save_json


class PaddyYieldDataset(Dataset):
    """PyTorch dataset for paddy yield data."""
    
    def __init__(self, samples):
        self.X = torch.stack([s['X'] for s in samples])          # (N, 15, T, 6)
        self.weights = torch.stack([s['weights'] for s in samples])  # (N, 15)
        self.y = torch.stack([s['y'] for s in samples]).squeeze()    # (N,)
        self.meta = [(s['year'], s['season'], s['district']) for s in samples]
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.weights[idx], self.y[idx]


def yield_scale(train_samples):
    """Mean and std of training yields, so losses are on a unit scale."""
    y = torch.stack([s['y'] for s in train_samples]).squeeze()
    std = float(y.std().clamp(min=1e-6))
    return {'mean': float(y.mean()), 'std': std}


def scale_targets(samples, scale):
    """Return copies of samples with yields standardized. Originals stay in kg/ha."""
    scaled = []
    for sample in samples:
        copy = dict(sample)
        copy['y'] = (sample['y'] - scale['mean']) / scale['std']
        scaled.append(copy)
    return scaled


def to_kg_ha(values, scale):
    """Map standardized predictions or targets back to kg/ha."""
    array = np.array(values, dtype=float)
    return array * scale['std'] + scale['mean']


def train_transformer(train_samples, val_samples, test_samples, 
                       models_dir, results_dir, epochs=200, lr=1e-3, device='cpu'):
    """Train the AW-MIL Temporal Transformer."""
    print("\n" + "=" * 60)
    print("Training AW-MIL Temporal Transformer")
    print("=" * 60)
    
    scale = yield_scale(train_samples)
    train_ds = PaddyYieldDataset(scale_targets(train_samples, scale))
    val_ds = PaddyYieldDataset(scale_targets(val_samples, scale))
    test_ds = PaddyYieldDataset(scale_targets(test_samples, scale))
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds))
    test_loader = DataLoader(test_ds, batch_size=len(test_ds))
    
    # Model
    model = AWMILTransformer(
        input_dim=6, d_model=32, nhead=4, num_layers=2,
        d_ff=64, dropout=0.35, num_boxes=15, max_seq_len=20
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.HuberLoss(delta=1.0)
    
    # Early stopping
    early_stop = EarlyStopping(patience=15, path=os.path.join(models_dir, 'transformer_best.pt'))
    
    # Training history
    history = {'train_loss': [], 'val_loss': [], 'val_r2': [], 'val_rmse': []}
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        for X, w, y in train_loader:
            X, w, y = X.to(device), w.to(device), y.to(device)
            
            optimizer.zero_grad()
            y_pred, _, _ = model(X, w)
            loss = criterion(y_pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validate
        model.eval()
        with torch.no_grad():
            val_losses = []
            val_preds = []
            val_targets = []
            for X, w, y in val_loader:
                X, w, y = X.to(device), w.to(device), y.to(device)
                y_pred, _, _ = model(X, w)
                loss = criterion(y_pred, y)
                val_losses.append(loss.item())
                val_preds.extend(y_pred.cpu().numpy())
                val_targets.extend(y.cpu().numpy())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_metrics = compute_metrics(val_targets, val_preds)
        
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['val_r2'].append(val_metrics['R2'])
        history['val_rmse'].append(val_metrics['RMSE'])
        
        scheduler.step(val_loss)
        early_stop(val_loss, model)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"R²: {val_metrics['R2']:.4f} | RMSE: {val_metrics['RMSE']:.1f}")
        
        if early_stop.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    early_stop.load_best(model)
    
    # Save model
    model_path = os.path.join(models_dir, 'transformer_best.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': 6, 'd_model': 32, 'nhead': 4, 'num_layers': 2,
            'd_ff': 64, 'dropout': 0.35, 'num_boxes': 15, 'max_seq_len': 20
        },
        'target_scale': scale,
        'history': history,
    }, model_path)
    print(f"Saved model to {model_path}")
    
    # Test evaluation
    model.eval()
    with torch.no_grad():
        for X, w, y in test_loader:
            X, w, y = X.to(device), w.to(device), y.to(device)
            y_pred, box_preds, attn = model(X, w)
    
    y_kg = to_kg_ha(y.cpu().numpy(), scale)
    pred_kg = to_kg_ha(y_pred.cpu().numpy(), scale)
    box_kg = to_kg_ha(box_preds.cpu().numpy(), scale)
    test_metrics = compute_metrics(y_kg, pred_kg)
    print(f"\nTest Results: R²={test_metrics['R2']:.4f}, RMSE={test_metrics['RMSE']:.1f}, "
          f"MAE={test_metrics['MAE']:.1f}, MAPE={test_metrics['MAPE']:.2f}%")
    
    # Save history and results
    save_json(history, os.path.join(results_dir, 'training_history.json'))
    
    # Save test predictions with metadata
    test_results = {
        'predictions': pred_kg.tolist(),
        'actuals': y_kg.tolist(),
        'box_predictions': box_kg.tolist(),
        'attention_weights': attn.cpu().numpy().tolist(),
        'meta': test_ds.meta,
        'metrics': test_metrics,
    }
    save_json(test_results, os.path.join(results_dir, 'transformer_test_results.json'))
    
    return model, test_metrics, history


def train_lstm(train_samples, val_samples, test_samples,
               models_dir, results_dir, epochs=200, lr=1e-3, device='cpu'):
    """Train LSTM baseline."""
    print("\n" + "=" * 60)
    print("Training LSTM Baseline")
    print("=" * 60)
    
    scale = yield_scale(train_samples)
    train_ds = PaddyYieldDataset(scale_targets(train_samples, scale))
    val_ds = PaddyYieldDataset(scale_targets(val_samples, scale))
    test_ds = PaddyYieldDataset(scale_targets(test_samples, scale))
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds))
    test_loader = DataLoader(test_ds, batch_size=len(test_ds))
    
    model = LSTMBaseline(input_dim=6, hidden_dim=64, num_layers=2, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    early_stop = EarlyStopping(patience=10)
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X, w, y in train_loader:
            X, w, y = X.to(device), w.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(X, w)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            val_losses = []
            for X, w, y in val_loader:
                X, w, y = X.to(device), w.to(device), y.to(device)
                y_pred = model(X, w)
                loss = criterion(y_pred, y)
                val_losses.append(loss.item())
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        
        early_stop(val_loss, model)
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        if early_stop.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    early_stop.load_best(model)
    
    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'target_scale': scale,
    }, os.path.join(models_dir, 'lstm_best.pt'))
    
    # Test
    model.eval()
    with torch.no_grad():
        for X, w, y in test_loader:
            X, w, y = X.to(device), w.to(device), y.to(device)
            y_pred = model(X, w)
    
    y_kg = to_kg_ha(y.cpu().numpy(), scale)
    pred_kg = to_kg_ha(y_pred.cpu().numpy(), scale)
    test_metrics = compute_metrics(y_kg, pred_kg)
    print(f"Test: R²={test_metrics['R2']:.4f}, RMSE={test_metrics['RMSE']:.1f}")
    
    save_json({'metrics': test_metrics, 'predictions': pred_kg.tolist(),
               'actuals': y_kg.tolist(), 'meta': test_ds.meta},
              os.path.join(results_dir, 'lstm_test_results.json'))
    
    return model, test_metrics


def train_rf(train_samples, test_samples, models_dir, results_dir):
    """Train Random Forest baseline."""
    print("\n" + "=" * 60)
    print("Training Random Forest")
    print("=" * 60)
    
    from sklearn.ensemble import RandomForestRegressor
    
    # Extract tabular features
    X_train = torch.stack([s['X'] for s in train_samples])
    w_train = torch.stack([s['weights'] for s in train_samples])
    y_train = np.array([s['y'].item() for s in train_samples])
    
    X_test = torch.stack([s['X'] for s in test_samples])
    w_test = torch.stack([s['weights'] for s in test_samples])
    y_test = np.array([s['y'].item() for s in test_samples])
    
    X_train_tab, feat_names = extract_tabular_features(X_train, w_train)
    X_test_tab, _ = extract_tabular_features(X_test, w_test)
    
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train_tab, y_train)
    
    y_pred = model.predict(X_test_tab)
    test_metrics = compute_metrics(y_test, y_pred)
    print(f"Test: R²={test_metrics['R2']:.4f}, RMSE={test_metrics['RMSE']:.1f}")
    
    joblib.dump(model, os.path.join(models_dir, 'rf_model.pkl'))
    save_json({'metrics': test_metrics, 'predictions': y_pred.tolist(),
               'actuals': y_test.tolist(), 'feature_names': feat_names},
              os.path.join(results_dir, 'rf_test_results.json'))
    
    return model, test_metrics


def train_xgboost(train_samples, test_samples, models_dir, results_dir):
    """Train XGBoost baseline."""
    print("\n" + "=" * 60)
    print("Training XGBoost")
    print("=" * 60)
    
    from xgboost import XGBRegressor
    
    X_train = torch.stack([s['X'] for s in train_samples])
    w_train = torch.stack([s['weights'] for s in train_samples])
    y_train = np.array([s['y'].item() for s in train_samples])
    
    X_test = torch.stack([s['X'] for s in test_samples])
    w_test = torch.stack([s['weights'] for s in test_samples])
    y_test = np.array([s['y'].item() for s in test_samples])
    
    X_train_tab, feat_names = extract_tabular_features(X_train, w_train)
    X_test_tab, _ = extract_tabular_features(X_test, w_test)
    
    model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, 
                          random_state=42, n_jobs=-1)
    model.fit(X_train_tab, y_train)
    
    y_pred = model.predict(X_test_tab)
    test_metrics = compute_metrics(y_test, y_pred)
    print(f"Test: R²={test_metrics['R2']:.4f}, RMSE={test_metrics['RMSE']:.1f}")
    
    joblib.dump(model, os.path.join(models_dir, 'xgb_model.pkl'))
    save_json({'metrics': test_metrics, 'predictions': y_pred.tolist(),
               'actuals': y_test.tolist(), 'feature_names': feat_names},
              os.path.join(results_dir, 'xgb_test_results.json'))
    
    return model, test_metrics


def main():
    parser = argparse.ArgumentParser(description='Train paddy yield prediction models')
    parser.add_argument('--model', default='all', choices=['all', 'transformer', 'lstm', 'rf', 'xgb'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--models-dir', default=None)
    parser.add_argument('--results-dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")
    
    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = args.data_dir or os.path.join(project_root, 'data')
    models_dir = args.models_dir or os.path.join(project_root, 'models')
    results_dir = args.results_dir or os.path.join(project_root, 'results')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load preprocessed data
    tensor_path = os.path.join(data_dir, 'processed', 'tensors.pt')
    if not os.path.exists(tensor_path):
        print("Preprocessed data not found. Running preprocessing...")
        from preprocessing import run_preprocessing
        data = run_preprocessing(data_dir, models_dir=models_dir)
    else:
        print(f"Loading preprocessed data from {tensor_path}")
        data = torch.load(tensor_path, weights_only=False)
    
    train_samples = data['train']
    val_samples = data['val']
    test_samples = data['test']
    
    print(f"\nData: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
    
    # Train models
    all_metrics = {}
    
    if args.model in ['all', 'transformer']:
        _, metrics, _ = train_transformer(
            train_samples, val_samples, test_samples,
            models_dir, results_dir, epochs=args.epochs, lr=args.lr, device=device
        )
        all_metrics['Transformer'] = metrics
    
    if args.model in ['all', 'lstm']:
        _, metrics = train_lstm(
            train_samples, val_samples, test_samples,
            models_dir, results_dir, epochs=args.epochs, lr=args.lr, device=device
        )
        all_metrics['LSTM'] = metrics
    
    if args.model in ['all', 'rf']:
        _, metrics = train_rf(train_samples, test_samples, models_dir, results_dir)
        all_metrics['Random Forest'] = metrics
    
    if args.model in ['all', 'xgb']:
        _, metrics = train_xgboost(train_samples, test_samples, models_dir, results_dir)
        all_metrics['XGBoost'] = metrics
    
    # Save comparison
    if len(all_metrics) > 1:
        save_json(all_metrics, os.path.join(results_dir, 'model_comparison.json'))
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        print(f"{'Model':<25} {'R²':>8} {'RMSE':>10} {'MAE':>10} {'MAPE':>8}")
        print("-" * 61)
        for name, m in all_metrics.items():
            print(f"{name:<25} {m['R2']:>8.4f} {m['RMSE']:>10.1f} {m['MAE']:>10.1f} {m['MAPE']:>7.2f}%")


if __name__ == '__main__':
    main()
