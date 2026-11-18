"""
Model architectures for Cauvery Delta paddy yield prediction.

Models:
1. AWMILTransformer - 15-Box Area-Weighted MIL Temporal Transformer (main model)
2. LSTMBaseline - 2-layer LSTM baseline
3. extract_tabular_features - Feature extraction for RF/XGBoost baselines
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""
    
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)
    
    def forward(self, x):
        """x: (B, T, d_model)"""
        return x + self.pe[:, :x.size(1)]


class TemporalAttentionPooling(nn.Module):
    """Learnable attention pooling over temporal dimension.
    
    α_t = softmax(w_p^T * tanh(W_p * h_t + b_p))
    h_pool = Σ α_t * h_t
    """
    
    def __init__(self, d_model):
        super().__init__()
        self.W_p = nn.Linear(d_model, d_model)
        self.w_p = nn.Linear(d_model, 1, bias=False)
    
    def forward(self, h, mask=None):
        """
        h: (B, T, d_model)
        mask: (B, T) boolean, True for valid positions
        Returns: pooled (B, d_model), attention_weights (B, T)
        """
        # Compute attention scores
        scores = self.w_p(torch.tanh(self.W_p(h)))  # (B, T, 1)
        scores = scores.squeeze(-1)  # (B, T)
        
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        alpha = F.softmax(scores, dim=-1)  # (B, T)
        
        # Weighted sum
        pooled = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)  # (B, d_model)
        
        return pooled, alpha


class AWMILTransformer(nn.Module):
    """15-Box Area-Weighted Multiple Instance Learning Temporal Transformer.
    
    Architecture:
        - Input projection: Linear(input_dim, d_model)
        - Sinusoidal positional encoding
        - L Transformer encoder layers (d_model, nhead, d_ff, dropout)
        - Temporal attention pooling
        - Regression head: Linear(d_model, 1)
        - Area-weighted aggregation across boxes
    
    Args:
        input_dim: Number of input features (default: 6)
        d_model: Hidden dimension (default: 32)
        nhead: Number of attention heads (default: 4)
        num_layers: Number of encoder layers (default: 2)
        d_ff: Feed-forward dimension (default: 64)
        dropout: Dropout probability (default: 0.35)
        num_boxes: Number of spatial grid boxes (default: 15)
        max_seq_len: Maximum sequence length (default: 20)
    """
    
    def __init__(self, input_dim=6, d_model=32, nhead=4, num_layers=2,
                 d_ff=64, dropout=0.35, num_boxes=15, max_seq_len=20):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_boxes = num_boxes
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Temporal attention pooling
        self.temporal_pool = TemporalAttentionPooling(d_model)
        
        # Regression head (per-box yield prediction)
        self.regression_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x, weights, mask=None):
        """
        Args:
            x: (B, num_boxes, T, input_dim) - Multi-modal features per box
            weights: (B, num_boxes) - Cropland area fractions
            mask: (B, T) - Valid timestep mask (optional)
        
        Returns:
            district_pred: (B,) - Area-weighted district yield prediction
            box_preds: (B, num_boxes) - Per-box yield predictions
            attn_weights: (B, num_boxes, T) - Temporal attention weights
        """
        B, N, T, D = x.shape
        
        # Process each box through the transformer
        # Reshape: (B * N, T, D)
        x_flat = x.view(B * N, T, D)
        
        # Input projection
        h = self.input_proj(x_flat)  # (B*N, T, d_model)
        
        # Positional encoding
        h = self.pos_enc(h)
        
        # Transformer encoding
        h = self.transformer(h)  # (B*N, T, d_model)
        
        # Temporal attention pooling
        h_reshaped = h.view(B, N, T, self.d_model)
        
        box_preds = []
        attn_weights = []
        
        for i in range(N):
            box_h = h_reshaped[:, i]  # (B, T, d_model)
            pooled, alpha = self.temporal_pool(box_h, mask)  # (B, d_model), (B, T)
            pred = self.regression_head(pooled).squeeze(-1)  # (B,)
            box_preds.append(pred)
            attn_weights.append(alpha)
        
        box_preds = torch.stack(box_preds, dim=1)      # (B, N)
        attn_weights = torch.stack(attn_weights, dim=1)  # (B, N, T)
        
        # Area-weighted aggregation
        # ŷ_district = Σ(w_i * ŷ_i) / Σ(w_i)
        w_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)  # (B, 1)
        district_pred = (weights * box_preds).sum(dim=1) / w_sum.squeeze(1)  # (B,)
        
        return district_pred, box_preds, attn_weights
    
    def predict_at_week(self, x, weights, week):
        """Predict using only data up to a specific week (for early forecasting).
        
        Args:
            x: (B, num_boxes, T, input_dim)
            weights: (B, num_boxes)
            week: int, number of weeks to use (1-indexed)
        """
        x_truncated = x[:, :, :week, :]
        # Zero-pad back to original length for positional encoding consistency
        pad_len = x.size(2) - week
        if pad_len > 0:
            padding = torch.zeros(x.size(0), x.size(1), pad_len, x.size(3), device=x.device)
            x_padded = torch.cat([x_truncated, padding], dim=2)
        else:
            x_padded = x_truncated
        
        # Create mask: True for valid positions
        mask = torch.zeros(x.size(0), x.size(2), dtype=torch.bool, device=x.device)
        mask[:, :week] = True
        
        return self.forward(x_padded, weights, mask)


class LSTMBaseline(nn.Module):
    """LSTM baseline for paddy yield prediction.
    
    Uses district-averaged features (mean across 15 boxes).
    Two-layer stacked LSTM with final hidden state → regression.
    """
    
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, weights=None):
        """
        x: (B, T, input_dim) - District-averaged weekly features
           OR (B, 15, T, input_dim) - will be averaged across boxes
        weights: optional (B, 15) for weighted averaging
        Returns: (B,) predictions
        """
        if x.dim() == 4:
            # Average across boxes
            if weights is not None:
                w = weights.unsqueeze(-1).unsqueeze(-1)  # (B, 15, 1, 1)
                w_sum = weights.sum(dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1).clamp(min=1e-8)
                x = (x * w).sum(dim=1) / w_sum.squeeze(1)  # (B, T, D)
            else:
                x = x.mean(dim=1)  # (B, T, D)
        
        output, (h_n, _) = self.lstm(x)  # h_n: (num_layers, B, hidden)
        h_last = self.dropout(h_n[-1])    # (B, hidden)
        pred = self.fc(h_last).squeeze(-1)  # (B,)
        return pred


def extract_tabular_features(X, weights=None):
    """Extract tabular features from time-series for RF/XGBoost.
    
    Args:
        X: (N, 15, T, 6) or (N, T, 6) tensor
        weights: (N, 15) optional cropland weights
    
    Returns:
        features: (N, num_features) numpy array
        feature_names: list of feature names
    """
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    if weights is not None and isinstance(weights, torch.Tensor):
        weights = weights.numpy()
    
    # If 4D (with boxes), average across boxes
    if X.ndim == 4:
        if weights is not None:
            w = weights[:, :, np.newaxis, np.newaxis]  # (N, 15, 1, 1)
            w_sum = weights.sum(axis=1, keepdims=True)[:, :, np.newaxis, np.newaxis]
            w_sum = np.clip(w_sum, 1e-8, None)
            X = (X * w).sum(axis=1) / w_sum.squeeze(1)  # (N, T, 6)
        else:
            X = X.mean(axis=1)
    
    N = X.shape[0]
    feat_names = ['VH_dB', 'VV_dB', 'NDVI', 'LSWI', 'rainfall_mm', 'temperature_C']
    all_features = []
    all_names = []
    
    for i, name in enumerate(feat_names):
        col = X[:, :, i]  # (N, T)
        
        # Statistical summaries
        all_features.append(np.nanmean(col, axis=1))
        all_names.append(f'{name}_mean')
        
        all_features.append(np.nanstd(col, axis=1))
        all_names.append(f'{name}_std')
        
        all_features.append(np.nanmax(col, axis=1))
        all_names.append(f'{name}_max')
        
        all_features.append(np.nanmin(col, axis=1))
        all_names.append(f'{name}_min')
        
        # Range
        all_features.append(np.nanmax(col, axis=1) - np.nanmin(col, axis=1))
        all_names.append(f'{name}_range')
    
    # Total rainfall
    rain_col = X[:, :, 4]
    all_features.append(np.nansum(rain_col, axis=1))
    all_names.append('rainfall_total')
    
    features = np.stack(all_features, axis=1)  # (N, num_features)
    
    # Replace any NaN with 0
    features = np.nan_to_num(features, nan=0.0)
    
    return features, all_names
