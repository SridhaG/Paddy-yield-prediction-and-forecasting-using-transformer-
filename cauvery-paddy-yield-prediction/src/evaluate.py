"""
Evaluation module for detailed model analysis.

Performs:
1. Full test set evaluation for all models
2. District-wise performance breakdown
3. Season-wise performance breakdown
4. Early-season forecasting evaluation (Weeks 4-16)
5. Ablation study (remove SAR, remove grid, replace attention pooling)
"""

import os
import sys
import json
import numpy as np
import torch
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from models import AWMILTransformer, LSTMBaseline, extract_tabular_features
from utils import set_seed, get_device, compute_metrics, save_json
from train import PaddyYieldDataset


def inverse_yield(pred, scale):
    """Map standardized model outputs back to kg/ha."""
    if not scale:
        return pred
    return pred * scale['std'] + scale['mean']


def load_transformer(models_dir, device='cpu'):
    """Load trained transformer model and its yield scale."""
    path = os.path.join(models_dir, 'transformer_best.pt')
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get('model_config', {
        'input_dim': 6, 'd_model': 32, 'nhead': 4, 'num_layers': 2,
        'd_ff': 64, 'dropout': 0.35, 'num_boxes': 15, 'max_seq_len': 20
    })
    model = AWMILTransformer(**config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint.get('target_scale')


def load_lstm(models_dir, device='cpu'):
    """Load trained LSTM model and its yield scale."""
    path = os.path.join(models_dir, 'lstm_best.pt')
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = LSTMBaseline(input_dim=6, hidden_dim=64, num_layers=2, dropout=0.3).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint.get('target_scale')


def evaluate_all_models(test_samples, models_dir, results_dir, device='cpu'):
    """Evaluate all trained models on the test set."""
    print("\n" + "=" * 60)
    print("FULL MODEL EVALUATION")
    print("=" * 60)
    
    test_ds = PaddyYieldDataset(test_samples)
    X, w, y = test_ds.X, test_ds.weights, test_ds.y
    y_np = y.numpy()
    
    results = {}
    
    # Transformer
    try:
        model, scale = load_transformer(models_dir, device)
        with torch.no_grad():
            X_d, w_d = X.to(device), w.to(device)
            y_pred, box_preds, attn = model(X_d, w_d)
            y_pred = inverse_yield(y_pred, scale)
            box_preds = inverse_yield(box_preds, scale)
        metrics = compute_metrics(y_np, y_pred.cpu().numpy())
        results['AW-MIL Transformer'] = {
            'metrics': metrics,
            'predictions': y_pred.cpu().numpy().tolist(),
            'box_predictions': box_preds.cpu().numpy().tolist(),
            'attention_weights': attn.cpu().numpy().tolist(),
        }
        print(f"Transformer: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.1f}")
    except FileNotFoundError:
        print("Transformer model not found, skipping.")
    
    # LSTM
    try:
        model, scale = load_lstm(models_dir, device)
        with torch.no_grad():
            y_pred = inverse_yield(model(X.to(device), w.to(device)), scale)
        metrics = compute_metrics(y_np, y_pred.cpu().numpy())
        results['LSTM'] = {
            'metrics': metrics,
            'predictions': y_pred.cpu().numpy().tolist(),
        }
        print(f"LSTM: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.1f}")
    except FileNotFoundError:
        print("LSTM model not found, skipping.")
    
    # RF
    try:
        rf = joblib.load(os.path.join(models_dir, 'rf_model.pkl'))
        X_tab, _ = extract_tabular_features(X, w)
        y_pred = rf.predict(X_tab)
        metrics = compute_metrics(y_np, y_pred)
        results['Random Forest'] = {
            'metrics': metrics,
            'predictions': y_pred.tolist(),
        }
        print(f"Random Forest: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.1f}")
    except FileNotFoundError:
        print("RF model not found, skipping.")
    
    # XGBoost
    try:
        xgb = joblib.load(os.path.join(models_dir, 'xgb_model.pkl'))
        X_tab, _ = extract_tabular_features(X, w)
        y_pred = xgb.predict(X_tab)
        metrics = compute_metrics(y_np, y_pred)
        results['XGBoost'] = {
            'metrics': metrics,
            'predictions': y_pred.tolist(),
        }
        print(f"XGBoost: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.1f}")
    except FileNotFoundError:
        print("XGBoost model not found, skipping.")
    
    results['actuals'] = y_np.tolist()
    results['meta'] = test_ds.meta
    
    save_json(results, os.path.join(results_dir, 'full_evaluation.json'))
    return results


def evaluate_by_district(test_samples, models_dir, results_dir, device='cpu'):
    """District-wise evaluation."""
    print("\n" + "=" * 60)
    print("DISTRICT-WISE EVALUATION")
    print("=" * 60)
    
    model, scale = load_transformer(models_dir, device)
    districts = ['Thanjavur', 'Thiruvarur', 'Nagapattinam']
    
    district_results = {}
    for dist in districts:
        dist_samples = [s for s in test_samples if s['district'] == dist]
        if not dist_samples:
            # Use all samples for the district from any split
            continue
        
        ds = PaddyYieldDataset(dist_samples)
        with torch.no_grad():
            y_pred, _, _ = model(ds.X.to(device), ds.weights.to(device))
            y_pred = inverse_yield(y_pred, scale)
        
        metrics = compute_metrics(ds.y.numpy(), y_pred.cpu().numpy())
        district_results[dist] = metrics
        print(f"{dist}: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.1f}, MAPE={metrics['MAPE']:.2f}%")
    
    save_json(district_results, os.path.join(results_dir, 'district_evaluation.json'))
    return district_results


def evaluate_by_season(test_samples, models_dir, results_dir, device='cpu'):
    """Season-wise evaluation."""
    print("\n" + "=" * 60)
    print("SEASON-WISE EVALUATION")
    print("=" * 60)
    
    model, scale = load_transformer(models_dir, device)
    seasons = ['Kuruvai', 'Samba', 'Thaladi']
    
    season_results = {}
    for season in seasons:
        season_samples = [s for s in test_samples if s['season'] == season]
        if not season_samples:
            continue
        
        ds = PaddyYieldDataset(season_samples)
        with torch.no_grad():
            y_pred, _, _ = model(ds.X.to(device), ds.weights.to(device))
            y_pred = inverse_yield(y_pred, scale)
        
        metrics = compute_metrics(ds.y.numpy(), y_pred.cpu().numpy())
        season_results[season] = metrics
        print(f"{season}: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.1f}")
    
    save_json(season_results, os.path.join(results_dir, 'season_evaluation.json'))
    return season_results


def evaluate_early_forecast(test_samples, models_dir, results_dir, device='cpu'):
    """Early-season forecasting evaluation at different forecast weeks."""
    print("\n" + "=" * 60)
    print("EARLY-SEASON FORECASTING")
    print("=" * 60)
    
    model, scale = load_transformer(models_dir, device)
    ds = PaddyYieldDataset(test_samples)
    X, w, y = ds.X.to(device), ds.weights.to(device), ds.y.numpy()
    
    forecast_weeks = [4, 6, 8, 10, 12, 14, 16]
    forecast_results = {}
    
    for week in forecast_weeks:
        with torch.no_grad():
            y_pred, _, _ = model.predict_at_week(X, w, week)
            y_pred = inverse_yield(y_pred, scale)
        
        metrics = compute_metrics(y, y_pred.cpu().numpy())
        forecast_results[f'week_{week}'] = metrics
        marker = " ← Operational milestone" if week == 12 else ""
        print(f"Week {week:2d}: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.1f}{marker}")
    
    save_json(forecast_results, os.path.join(results_dir, 'early_forecast.json'))
    return forecast_results


def ablation_study(train_samples, val_samples, test_samples, 
                   models_dir, results_dir, device='cpu'):
    """Ablation study: remove SAR, remove grid, replace attention pooling."""
    print("\n" + "=" * 60)
    print("ABLATION STUDY")
    print("=" * 60)
    
    from torch.utils.data import DataLoader
    
    ablation_results = {}
    
    # Full model (baseline)
    model, scale = load_transformer(models_dir, device)
    ds = PaddyYieldDataset(test_samples)
    with torch.no_grad():
        y_pred, _, _ = model(ds.X.to(device), ds.weights.to(device))
        y_pred = inverse_yield(y_pred, scale)
    full_metrics = compute_metrics(ds.y.numpy(), y_pred.cpu().numpy())
    ablation_results['Full Model'] = full_metrics
    print(f"Full Model: R²={full_metrics['R2']:.4f}, RMSE={full_metrics['RMSE']:.1f}")
    
    # Without SAR (zero out VH and VV channels - indices 0,1)
    X_no_sar = ds.X.clone()
    X_no_sar[:, :, :, 0] = 0  # VH
    X_no_sar[:, :, :, 1] = 0  # VV
    with torch.no_grad():
        y_pred, _, _ = model(X_no_sar.to(device), ds.weights.to(device))
        y_pred = inverse_yield(y_pred, scale)
    no_sar_metrics = compute_metrics(ds.y.numpy(), y_pred.cpu().numpy())
    ablation_results['Without SAR'] = no_sar_metrics
    delta_r2 = no_sar_metrics['R2'] - full_metrics['R2']
    print(f"Without SAR: R²={no_sar_metrics['R2']:.4f} (ΔR²={delta_r2:+.4f}), "
          f"RMSE={no_sar_metrics['RMSE']:.1f}")
    
    # Without Grid (use uniform weights instead of cropland-weighted)
    uniform_weights = torch.ones_like(ds.weights) / 15.0
    with torch.no_grad():
        y_pred, _, _ = model(ds.X.to(device), uniform_weights.to(device))
        y_pred = inverse_yield(y_pred, scale)
    no_grid_metrics = compute_metrics(ds.y.numpy(), y_pred.cpu().numpy())
    ablation_results['Without Grid Weighting'] = no_grid_metrics
    delta_r2 = no_grid_metrics['R2'] - full_metrics['R2']
    print(f"Without Grid: R²={no_grid_metrics['R2']:.4f} (ΔR²={delta_r2:+.4f}), "
          f"RMSE={no_grid_metrics['RMSE']:.1f}")
    
    # Without Attention Pooling (use mean pooling)
    # We modify the model's temporal pooling temporarily
    original_pool = model.temporal_pool
    
    class MeanPooling(torch.nn.Module):
        def forward(self, h, mask=None):
            if mask is not None:
                h = h * mask.unsqueeze(-1).float()
                pooled = h.sum(dim=1) / mask.float().sum(dim=1, keepdim=True).clamp(min=1)
            else:
                pooled = h.mean(dim=1)
            alpha = torch.ones(h.size(0), h.size(1), device=h.device) / h.size(1)
            return pooled, alpha
    
    model.temporal_pool = MeanPooling()
    with torch.no_grad():
        y_pred, _, _ = model(ds.X.to(device), ds.weights.to(device))
        y_pred = inverse_yield(y_pred, scale)
    no_attn_metrics = compute_metrics(ds.y.numpy(), y_pred.cpu().numpy())
    ablation_results['Mean Pooling'] = no_attn_metrics
    delta_r2 = no_attn_metrics['R2'] - full_metrics['R2']
    print(f"Mean Pooling: R²={no_attn_metrics['R2']:.4f} (ΔR²={delta_r2:+.4f}), "
          f"RMSE={no_attn_metrics['RMSE']:.1f}")
    
    model.temporal_pool = original_pool  # Restore
    
    save_json(ablation_results, os.path.join(results_dir, 'ablation_study.json'))
    return ablation_results


def run_full_evaluation(data_dir, models_dir, results_dir):
    """Run all evaluations."""
    set_seed(42)
    device = get_device()
    
    # Load data
    tensor_path = os.path.join(data_dir, 'processed', 'tensors.pt')
    data = torch.load(tensor_path, weights_only=False)
    
    train_samples = data['train']
    val_samples = data['val']
    test_samples = data['test']
    
    # Use all samples for evaluation when test set is small
    all_samples = train_samples + val_samples + test_samples
    
    print(f"Total samples: {len(all_samples)} (Train: {len(train_samples)}, "
          f"Val: {len(val_samples)}, Test: {len(test_samples)})")
    
    # Run evaluations
    eval_results = evaluate_all_models(all_samples, models_dir, results_dir, device)
    district_results = evaluate_by_district(all_samples, models_dir, results_dir, device)
    season_results = evaluate_by_season(all_samples, models_dir, results_dir, device)
    forecast_results = evaluate_early_forecast(all_samples, models_dir, results_dir, device)
    ablation_results = ablation_study(train_samples, val_samples, all_samples,
                                      models_dir, results_dir, device)
    
    # Combined results
    combined = {
        'model_comparison': {k: v.get('metrics', v) for k, v in eval_results.items() 
                            if k not in ['actuals', 'meta']},
        'district_wise': district_results,
        'season_wise': season_results,
        'early_forecast': forecast_results,
        'ablation': ablation_results,
    }
    save_json(combined, os.path.join(results_dir, 'combined_results.json'))
    
    print("\n" + "=" * 60)
    print("ALL EVALUATIONS COMPLETE")
    print(f"Results saved to {results_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    models_dir = os.path.join(project_root, 'models')
    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    run_full_evaluation(data_dir, models_dir, results_dir)
