"""
Google Earth Engine data collection script for Cauvery Delta paddy yield prediction.
Collects: Sentinel-1 SAR (VH/VV), MODIS NDVI/LSWI, CHIRPS rainfall, ERA5-Land temperature.
Study area: Thanjavur, Thiruvarur, Nagapattinam districts (2015-2025).

Usage:
    python src/collect_gee_data.py                    # Full GEE collection
    python src/collect_gee_data.py --dry-run           # Generate grid metadata only
    python src/collect_gee_data.py --synthetic          # Generate synthetic satellite/weather data for testing

Data Sources:
    - Sentinel-1 C-band IW GRD: COPERNICUS/S1_GRD (VH, VV backscatter)
    - MODIS: MODIS/061/MOD13Q1 (NDVI, LSWI via bands)
    - CHIRPS: UCSB-CHG/CHIRPS/DAILY (precipitation)
    - ERA5-Land: ECMWF/ERA5_LAND/DAILY_AGGR (2m temperature)
    - ESA WorldCover: ESA/WorldCover/v200 (cropland fraction)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add parent dir to path
sys.path.insert(0, os.path.dirname(__file__))
from grid_utils import generate_grid_metadata, DISTRICT_CENTERS, BOX_SIZE

# Season definitions
SEASONS = {
    'Kuruvai':  {'start_month': 6, 'start_day': 1, 'weeks': 16},   # Jun-Sep
    'Samba':    {'start_month': 8, 'start_day': 1, 'weeks': 20},    # Aug-Jan
    'Thaladi':  {'start_month': 10, 'start_day': 1, 'weeks': 20},   # Oct-Feb
}

YEARS = list(range(2015, 2026))
DISTRICTS = ['Thanjavur', 'Thiruvarur', 'Nagapattinam']


def get_season_dates(year, season):
    """Get start and end dates for a season-year."""
    s = SEASONS[season]
    start = datetime(year, s['start_month'], s['start_day'])
    end = start + timedelta(weeks=s['weeks'])
    # Handle year rollover for Samba/Thaladi
    return start, end


def collect_with_gee(grid_df, output_dir):
    """Collect satellite and weather data using Google Earth Engine."""
    try:
        import ee
        ee.Authenticate()
        ee.Initialize(project='ee-project')
    except Exception as e:
        print(f"GEE initialization failed: {e}")
        print("Use --synthetic flag to generate test data without GEE.")
        return None, None
    
    satellite_rows = []
    weather_rows = []
    
    for year in YEARS:
        for season_name, season_def in SEASONS.items():
            start_date, end_date = get_season_dates(year, season_name)
            print(f"Processing {year} {season_name}: {start_date.date()} to {end_date.date()}")
            
            for _, box in grid_df.iterrows():
                if box['cropland_fraction'] < 0.05:
                    continue
                    
                region = ee.Geometry.Rectangle([
                    box['lon_min'], box['lat_min'],
                    box['lon_max'], box['lat_max']
                ])
                
                # Weekly aggregation
                for week in range(season_def['weeks']):
                    w_start = start_date + timedelta(weeks=week)
                    w_end = w_start + timedelta(days=7)
                    w_start_str = w_start.strftime('%Y-%m-%d')
                    w_end_str = w_end.strftime('%Y-%m-%d')
                    
                    # --- Sentinel-1 SAR ---
                    try:
                        s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
                              .filterBounds(region)
                              .filterDate(w_start_str, w_end_str)
                              .filter(ee.Filter.eq('instrumentMode', 'IW'))
                              .select(['VH', 'VV']))
                        s1_mean = s1.mean().reduceRegion(
                            reducer=ee.Reducer.mean(), geometry=region, scale=1000
                        ).getInfo()
                        vh = s1_mean.get('VH', np.nan)
                        vv = s1_mean.get('VV', np.nan)
                    except:
                        vh, vv = np.nan, np.nan
                    
                    # --- MODIS NDVI/LSWI ---
                    try:
                        modis = (ee.ImageCollection('MODIS/061/MOD13Q1')
                                 .filterBounds(region)
                                 .filterDate(w_start_str, w_end_str))
                        modis_mean = modis.mean().reduceRegion(
                            reducer=ee.Reducer.mean(), geometry=region, scale=1000
                        ).getInfo()
                        ndvi = modis_mean.get('NDVI', np.nan)
                        if ndvi is not None and ndvi != np.nan:
                            ndvi = ndvi * 0.0001  # Scale factor
                        # LSWI from NIR and SWIR bands
                        nir = modis_mean.get('sur_refl_b02', np.nan)
                        swir = modis_mean.get('sur_refl_b06', np.nan)
                        if nir and swir and nir + swir != 0:
                            lswi = (nir - swir) / (nir + swir)
                        else:
                            lswi = np.nan
                    except:
                        ndvi, lswi = np.nan, np.nan
                    
                    satellite_rows.append({
                        'year': year, 'season': season_name, 'week': week + 1,
                        'box_id': box['box_id'], 'district': box['district'],
                        'VH_dB': vh, 'VV_dB': vv, 'NDVI': ndvi, 'LSWI': lswi,
                    })
                    
                    # --- CHIRPS Rainfall ---
                    try:
                        chirps = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
                                  .filterBounds(region)
                                  .filterDate(w_start_str, w_end_str))
                        rain = chirps.sum().reduceRegion(
                            reducer=ee.Reducer.mean(), geometry=region, scale=5000
                        ).getInfo()
                        rainfall = rain.get('precipitation', np.nan)
                    except:
                        rainfall = np.nan
                    
                    # --- ERA5-Land Temperature ---
                    try:
                        era5 = (ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
                                .filterBounds(region)
                                .filterDate(w_start_str, w_end_str)
                                .select('temperature_2m'))
                        temp = era5.mean().reduceRegion(
                            reducer=ee.Reducer.mean(), geometry=region, scale=10000
                        ).getInfo()
                        temperature = temp.get('temperature_2m', np.nan)
                        if temperature is not None and temperature != np.nan:
                            temperature = temperature - 273.15  # K to °C
                    except:
                        temperature = np.nan
                    
                    weather_rows.append({
                        'year': year, 'season': season_name, 'week': week + 1,
                        'box_id': box['box_id'], 'district': box['district'],
                        'rainfall_mm': rainfall, 'temperature_C': temperature,
                    })
    
    sat_df = pd.DataFrame(satellite_rows)
    weather_df = pd.DataFrame(weather_rows)
    return sat_df, weather_df


def generate_synthetic_data(grid_df):
    """Generate realistic synthetic satellite and weather data for testing.
    
    Based on published phenological patterns for Cauvery Delta paddy:
    - SAR VH increases from -18dB (transplanting) to -12dB (peak tillering)
    - SAR VV ranges -10 to -6 dB
    - NDVI follows sigmoid growth: 0.2 (bare) → 0.7-0.85 (peak) → 0.4 (harvest)
    - LSWI high during flooding (0.3-0.5), decreases during growth
    - Rainfall: monsoon peak during Samba (Oct-Dec)
    - Temperature: 26-33°C range
    """
    np.random.seed(42)
    
    satellite_rows = []
    weather_rows = []
    
    for year in YEARS:
        # Year effect: slight trend + random
        year_effect = (year - 2020) * 0.01 + np.random.normal(0, 0.02)
        
        for season_name, season_def in SEASONS.items():
            n_weeks = season_def['weeks']
            
            # Season-specific parameters
            if season_name == 'Kuruvai':
                ndvi_peak = 0.78 + np.random.normal(0, 0.03)
                rain_base = 15.0
                temp_base = 30.5
            elif season_name == 'Samba':
                ndvi_peak = 0.72 + np.random.normal(0, 0.04)
                rain_base = 35.0  # Monsoon
                temp_base = 28.0
            else:  # Thaladi
                ndvi_peak = 0.68 + np.random.normal(0, 0.04)
                rain_base = 25.0
                temp_base = 27.0
            
            for _, box in grid_df.iterrows():
                if box['cropland_fraction'] < 0.05:
                    continue
                
                crop_frac = box['cropland_fraction']
                # Spatial variation based on position
                spatial_factor = 0.95 + 0.1 * crop_frac
                
                for week in range(1, n_weeks + 1):
                    t = week / n_weeks  # Normalized time [0, 1]
                    
                    # --- SAR Backscatter ---
                    # VH: increases with crop growth (-18 to -12 dB)
                    vh_base = -18 + 6 * np.sin(np.pi * t) * spatial_factor
                    vh = vh_base + np.random.normal(0, 0.8) + year_effect
                    
                    # VV: standing water early, increases later (-10 to -6 dB)
                    vv_base = -10 + 4 * (1 - np.exp(-3 * t)) * spatial_factor
                    vv = vv_base + np.random.normal(0, 0.6) + year_effect
                    
                    # --- NDVI ---
                    # Sigmoid growth curve
                    ndvi = 0.15 + ndvi_peak * (1 / (1 + np.exp(-8 * (t - 0.4))))
                    if t > 0.8:  # Senescence
                        ndvi *= (1 - 0.5 * (t - 0.8) / 0.2)
                    ndvi = ndvi * spatial_factor + np.random.normal(0, 0.03)
                    ndvi = np.clip(ndvi, 0.0, 1.0)
                    
                    # Cloud gaps during Samba monsoon (weeks 6-14)
                    if season_name == 'Samba' and 6 <= week <= 14:
                        if np.random.random() < 0.3:  # 30% cloud contamination
                            ndvi = np.nan
                    
                    # --- LSWI ---
                    lswi = 0.4 * (1 - t) + 0.1 * np.sin(2 * np.pi * t)
                    lswi += np.random.normal(0, 0.05)
                    lswi = np.clip(lswi, -0.3, 0.6)
                    if season_name == 'Samba' and 6 <= week <= 14:
                        if np.random.random() < 0.3:
                            lswi = np.nan
                    
                    # --- Rainfall ---
                    if season_name == 'Samba':
                        # Monsoon rainfall pattern
                        rain_seasonal = rain_base * (1 + 1.5 * np.sin(np.pi * (t - 0.2)))
                    else:
                        rain_seasonal = rain_base * (1 + 0.5 * np.sin(np.pi * t))
                    rainfall = max(0, rain_seasonal + np.random.normal(0, rain_base * 0.3))
                    
                    # Cyclone events (Gaja 2018, Nivar 2020)
                    if year == 2018 and season_name == 'Samba' and 10 <= week <= 12:
                        rainfall += np.random.uniform(50, 120)
                    if year == 2020 and season_name == 'Samba' and 12 <= week <= 14:
                        rainfall += np.random.uniform(40, 80)
                    
                    # --- Temperature ---
                    temp = temp_base + 2 * np.sin(np.pi * t) + np.random.normal(0, 0.8)
                    
                    satellite_rows.append({
                        'year': year, 'season': season_name, 'week': week,
                        'box_id': box['box_id'], 'district': box['district'],
                        'VH_dB': round(vh, 3), 'VV_dB': round(vv, 3),
                        'NDVI': round(ndvi, 4) if not np.isnan(ndvi) else np.nan,
                        'LSWI': round(lswi, 4) if not np.isnan(lswi) else np.nan,
                    })
                    
                    weather_rows.append({
                        'year': year, 'season': season_name, 'week': week,
                        'box_id': box['box_id'], 'district': box['district'],
                        'rainfall_mm': round(rainfall, 2),
                        'temperature_C': round(temp, 2),
                    })
    
    sat_df = pd.DataFrame(satellite_rows)
    weather_df = pd.DataFrame(weather_rows)
    return sat_df, weather_df


def main():
    parser = argparse.ArgumentParser(description='Collect GEE data for Cauvery Delta')
    parser.add_argument('--dry-run', action='store_true', help='Generate grid metadata only')
    parser.add_argument('--synthetic', action='store_true', 
                        help='Generate synthetic data for testing (no GEE needed)')
    parser.add_argument('--output-dir', default=None, help='Output directory')
    args = parser.parse_args()
    
    # Determine output directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = args.output_dir or os.path.join(project_root, 'data')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate grid metadata
    print("Generating grid metadata...")
    grid_df = generate_grid_metadata()
    grid_path = os.path.join(output_dir, 'grid_metadata.csv')
    grid_df.to_csv(grid_path, index=False)
    print(f"Saved {len(grid_df)} grid boxes to {grid_path}")
    
    if args.dry_run:
        print("Dry run complete. Grid metadata saved.")
        return
    
    if args.synthetic:
        print("Generating synthetic satellite and weather data...")
        sat_df, weather_df = generate_synthetic_data(grid_df)
    else:
        print("Collecting data from Google Earth Engine...")
        sat_df, weather_df = collect_with_gee(grid_df, output_dir)
        if sat_df is None:
            print("GEE collection failed. Use --synthetic for testing.")
            return
    
    # Save satellite data
    sat_path = os.path.join(output_dir, 'satellite.csv')
    sat_df.to_csv(sat_path, index=False)
    print(f"Saved satellite data: {len(sat_df)} rows to {sat_path}")
    
    # Save weather data
    weather_path = os.path.join(output_dir, 'weather.csv')
    weather_df.to_csv(weather_path, index=False)
    print(f"Saved weather data: {len(weather_df)} rows to {weather_path}")
    
    # Create merged dataset
    merged = sat_df.merge(weather_df, on=['year', 'season', 'week', 'box_id', 'district'])
    merged = merged.merge(grid_df[['box_id', 'cropland_fraction']], on='box_id')
    merged_path = os.path.join(output_dir, 'merged_dataset.csv')
    merged.to_csv(merged_path, index=False)
    print(f"Saved merged dataset: {len(merged)} rows to {merged_path}")
    
    # Summary
    print("\n=== Data Collection Summary ===")
    print(f"Districts: {sat_df['district'].nunique()}")
    print(f"Years: {sat_df['year'].min()}-{sat_df['year'].max()}")
    print(f"Seasons: {sat_df['season'].unique().tolist()}")
    print(f"Grid boxes: {sat_df['box_id'].nunique()}")
    print(f"Satellite records: {len(sat_df)}")
    print(f"Weather records: {len(weather_df)}")
    print(f"Missing NDVI: {sat_df['NDVI'].isna().sum()} ({sat_df['NDVI'].isna().mean()*100:.1f}%)")
    print(f"Missing LSWI: {sat_df['LSWI'].isna().sum()} ({sat_df['LSWI'].isna().mean()*100:.1f}%)")


if __name__ == '__main__':
    main()
