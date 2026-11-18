"""
Preprocessing pipeline for Cauvery Delta paddy yield prediction.
Handles data loading, cleaning, feature engineering, normalization,
tensor creation, and train/val/test splitting.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import StandardScaler


# Constants
FEATURES = ['VH_dB', 'VV_dB', 'NDVI', 'LSWI', 'rainfall_mm', 'temperature_C']
MAX_WEEKS = 20  # Pad/truncate to this length
NUM_BOXES = 15
DISTRICTS = ['Thanjavur', 'Thiruvarur', 'Nagapattinam']
SEASONS = ['Kuruvai', 'Samba', 'Thaladi']

# Chronological split
TRAIN_YEARS = list(range(2015, 2022))   # 2015-2021
VAL_YEARS = [2022, 2023]                # 2022-2023
TEST_YEARS = [2024, 2025]               # 2024-2025


def load_raw_data(data_dir):
    """Load all raw CSV files."""
    dfs = {}
    
    # Load yield data
    yield_path = os.path.join(data_dir, 'yield.csv')
    dfs['yield'] = pd.read_csv(yield_path, comment='#')
    print(f"Loaded yield: {len(dfs['yield'])} records")
    
    # Load satellite data
    sat_path = os.path.join(data_dir, 'satellite.csv')
    if os.path.exists(sat_path):
        dfs['satellite'] = pd.read_csv(sat_path)
        print(f"Loaded satellite: {len(dfs['satellite'])} records")
    
    # Load weather data
    weather_path = os.path.join(data_dir, 'weather.csv')
    if os.path.exists(weather_path):
        dfs['weather'] = pd.read_csv(weather_path)
        print(f"Loaded weather: {len(dfs['weather'])} records")
    
    # Load grid metadata
    grid_path = os.path.join(data_dir, 'grid_metadata.csv')
    if os.path.exists(grid_path):
        dfs['grid'] = pd.read_csv(grid_path)
        print(f"Loaded grid: {len(dfs['grid'])} boxes")
    
    # Load merged if available
    merged_path = os.path.join(data_dir, 'merged_dataset.csv')
    if os.path.exists(merged_path):
        dfs['merged'] = pd.read_csv(merged_path)
        print(f"Loaded merged: {len(dfs['merged'])} records")
    
    return dfs


def clean_data(dfs):
    """Clean and merge raw data."""
    # Create merged if not available
    if 'merged' not in dfs and 'satellite' in dfs and 'weather' in dfs:
        merged = dfs['satellite'].merge(
            dfs['weather'], on=['year', 'season', 'week', 'box_id', 'district']
        )
        if 'grid' in dfs:
            merged = merged.merge(
                dfs['grid'][['box_id', 'cropland_fraction']], on='box_id'
            )
        dfs['merged'] = merged
    
    df = dfs['merged'].copy()
    
    # Filter low cropland boxes
    if 'cropland_fraction' in df.columns:
        before = len(df)
        df = df[df['cropland_fraction'] >= 0.05].copy()
        print(f"Filtered cropland < 5%: {before} -> {len(df)} rows")
    
    # Handle missing values
    # Forward-fill then backward-fill within each box-season-year group
    group_cols = ['box_id', 'year', 'season']
    for feat in FEATURES:
        if feat in df.columns:
            df[feat] = df.groupby(group_cols)[feat].transform(
                lambda x: x.ffill().bfill()
            )
    
    # Fill remaining NaN with median of same week across years
    for feat in FEATURES:
        if feat in df.columns:
            medians = df.groupby(['box_id', 'season', 'week'])[feat].transform('median')
            df[feat] = df[feat].fillna(medians)
    
    # Final fill with global median
    for feat in FEATURES:
        if feat in df.columns:
            df[feat] = df[feat].fillna(df[feat].median())
    
    # Clip extreme outliers (beyond 3σ)
    for feat in FEATURES:
        if feat in df.columns:
            mean, std = df[feat].mean(), df[feat].std()
            df[feat] = df[feat].clip(mean - 3*std, mean + 3*std)
    
    dfs['merged_clean'] = df
    print(f"Clean data: {len(df)} rows, missing values: {df[FEATURES].isna().sum().sum()}")
    return dfs


def normalize_features(df, scaler=None, fit=False):
    """Normalize features using StandardScaler."""
    feat_cols = [f for f in FEATURES if f in df.columns]
    
    if fit or scaler is None:
        scaler = StandardScaler()
        df[feat_cols] = scaler.fit_transform(df[feat_cols])
    else:
        df[feat_cols] = scaler.transform(df[feat_cols])
    
    return df, scaler


def create_tensors(dfs):
    """Create PyTorch tensors for model training.
    
    Returns:
        samples: list of dicts with keys:
            - X: (15, T, 6) tensor
            - weights: (15,) tensor of cropland fractions
            - y: scalar yield
            - year: int
            - season: str
            - district: str
    """
    df = dfs['merged_clean'].copy()
    yield_df = dfs['yield']
    grid_df = dfs.get('grid')
    
    samples = []
    
    for _, yrow in yield_df.iterrows():
        year = yrow['year']
        season = yrow['season']
        district = yrow['district']
        yield_val = yrow['yield_kg_ha']
        
        # Get district prefix for box IDs
        prefix = district[:3].upper()
        
        # Get data for this district-season-year
        mask = (df['year'] == year) & (df['season'] == season) & (df['district'] == district)
        district_data = df[mask]
        
        if len(district_data) == 0:
            continue
        
        # Get all boxes for this district
        district_boxes = sorted(district_data['box_id'].unique())
        
        # Create tensor: (num_boxes, max_weeks, num_features)
        X = np.zeros((NUM_BOXES, MAX_WEEKS, len(FEATURES)))
        weights = np.zeros(NUM_BOXES)
        
        feat_cols = [f for f in FEATURES if f in df.columns]
        
        for i, box_id in enumerate(district_boxes[:NUM_BOXES]):
            box_data = district_data[district_data['box_id'] == box_id].sort_values('week')
            
            # Get cropland fraction
            if grid_df is not None:
                box_meta = grid_df[grid_df['box_id'] == box_id]
                if len(box_meta) > 0:
                    weights[i] = box_meta['cropland_fraction'].values[0]
            elif 'cropland_fraction' in box_data.columns:
                weights[i] = box_data['cropland_fraction'].values[0]
            
            # Fill features
            n_weeks = min(len(box_data), MAX_WEEKS)
            for j, feat in enumerate(feat_cols):
                if feat in box_data.columns:
                    vals = box_data[feat].values[:MAX_WEEKS]
                    X[i, :len(vals), j] = vals
        
        # Normalize weights to sum to 1
        if weights.sum() > 0:
            weights_norm = weights / weights.sum()
        else:
            weights_norm = np.ones(NUM_BOXES) / NUM_BOXES
        
        samples.append({
            'X': torch.FloatTensor(X),
            'weights': torch.FloatTensor(weights_norm),
            'weights_raw': torch.FloatTensor(weights),
            'y': torch.FloatTensor([yield_val]),
            'year': year,
            'season': season,
            'district': district,
        })
    
    print(f"Created {len(samples)} tensor samples")
    return samples


def create_splits(samples):
    """Split samples chronologically into train/val/test."""
    train = [s for s in samples if s['year'] in TRAIN_YEARS]
    val = [s for s in samples if s['year'] in VAL_YEARS]
    test = [s for s in samples if s['year'] in TEST_YEARS]
    
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    
    split_info = {
        'train_years': TRAIN_YEARS,
        'val_years': VAL_YEARS,
        'test_years': TEST_YEARS,
        'train_count': len(train),
        'val_count': len(val),
        'test_count': len(test),
    }
    
    return train, val, test, split_info


def run_preprocessing(data_dir, output_dir=None, models_dir=None):
    """Run full preprocessing pipeline."""
    if output_dir is None:
        output_dir = os.path.join(data_dir, 'processed')
    if models_dir is None:
        models_dir = os.path.join(os.path.dirname(data_dir), 'models')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Load data
    print("=" * 50)
    print("Loading raw data...")
    dfs = load_raw_data(data_dir)
    
    # Clean
    print("\nCleaning data...")
    dfs = clean_data(dfs)
    
    # Normalize (fit on training years only)
    print("\nNormalizing features...")
    df = dfs['merged_clean']
    train_mask = df['year'].isin(TRAIN_YEARS)
    
    # Fit scaler on training data
    feat_cols = [f for f in FEATURES if f in df.columns]
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, feat_cols])
    
    # Transform all data
    df[feat_cols] = scaler.transform(df[feat_cols])
    dfs['merged_clean'] = df
    
    # Save scaler
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")
    
    # Create tensors
    print("\nCreating tensors...")
    samples = create_tensors(dfs)
    
    # Split
    print("\nSplitting data...")
    train, val, test, split_info = create_splits(samples)
    
    # Save
    data = {
        'train': train, 'val': val, 'test': test,
        'features': FEATURES,
        'split_info': split_info,
    }
    tensor_path = os.path.join(output_dir, 'tensors.pt')
    torch.save(data, tensor_path)
    print(f"Saved tensors to {tensor_path}")
    
    split_path = os.path.join(output_dir, 'split_indices.json')
    with open(split_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # Save merged clean data
    merged_path = os.path.join(output_dir, 'merged_clean.csv')
    df.to_csv(merged_path, index=False)
    
    print("\n" + "=" * 50)
    print("Preprocessing complete!")
    print(f"  Train samples: {len(train)}")
    print(f"  Val samples: {len(val)}")
    print(f"  Test samples: {len(test)}")
    print(f"  Features: {FEATURES}")
    
    return data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=None)
    args = parser.parse_args()
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = args.data_dir or os.path.join(project_root, 'data')
    models_dir = os.path.join(project_root, 'models')
    
    run_preprocessing(data_dir, models_dir=models_dir)
