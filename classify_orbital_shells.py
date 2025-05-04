#!/usr/bin/env python3
"""
Script to classify Starlink satellites into their orbital shells based on distance metrics.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import os

def extract_distance_features(periods_df: pd.DataFrame, features_file: str = None, force_extract: bool = False) -> pd.DataFrame:
    """Extract distance-based features from connection periods."""
    if features_file and not force_extract:
        try:
            # Try to load existing features
            features_df = pd.read_csv(features_file)
            print(f"Loaded existing features from {features_file}")
            return features_df
        except FileNotFoundError:
            print(f"Features file not found, extracting new features")
    
    print("Extracting distance features...")
    features = []
    
    # Constants
    EARTH_RADIUS = 6371  # km
    OBSERVER_ALTITUDE = 234.59393108338304 / 1000  # Convert meters to kilometers
    OBSERVER_LAT = 44.99114849999998  # degrees
    OBSERVER_LON = -93.22254183333334  # degrees
    
    def load_tle_data(satellite: str) -> pd.DataFrame:
        """Load TLE data for a specific satellite."""
        tle_file = f"output/tle_{satellite}.txt"
        try:
            with open(tle_file, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    return lines[-2:]  # Return the most recent TLE
            return None
        except FileNotFoundError:
            print(f"Warning: TLE file not found for satellite {satellite}")
            return None
    
    def calculate_satellite_position(tle_lines, timestamp):
        """Calculate satellite position using TLE data."""
        from skyfield.api import load, wgs84
        from datetime import datetime
        
        # Load TLE data
        satellite = load.tle(tle_lines[0], tle_lines[1])
        
        # Convert timestamp to skyfield time
        ts = load.timescale()
        t = ts.utc(datetime.fromtimestamp(timestamp.timestamp()))
        
        # Calculate position
        geocentric = satellite.at(t)
        subpoint = wgs84.subpoint(geocentric)
        
        return {
            'latitude': subpoint.latitude.degrees,
            'longitude': subpoint.longitude.degrees,
            'altitude': subpoint.elevation.km
        }
    
    def calculate_elevation_angle(sat_pos, obs_pos):
        """Calculate elevation angle from observer to satellite."""
        from math import sin, cos, asin, sqrt, radians
        
        # Convert positions to radians
        sat_lat = radians(sat_pos['latitude'])
        sat_lon = radians(sat_pos['longitude'])
        obs_lat = radians(obs_pos['latitude'])
        obs_lon = radians(obs_pos['longitude'])
        
        # Calculate great circle distance
        d_lon = sat_lon - obs_lon
        d_lat = sat_lat - obs_lat
        a = sin(d_lat/2)**2 + cos(obs_lat) * cos(sat_lat) * sin(d_lon/2)**2
        c = 2 * asin(sqrt(a))
        distance = EARTH_RADIUS * c
        
        # Calculate elevation angle
        # Using law of cosines: c² = a² + b² - 2ab*cos(C)
        # Where:
        # c = slant_range (measured distance)
        # a = Earth radius + observer altitude
        # b = Earth radius + satellite altitude
        # C = 90° + elevation_angle
        
        a = EARTH_RADIUS + OBSERVER_ALTITUDE
        b = EARTH_RADIUS + sat_pos['altitude']
        c = distance
        
        # Solve for elevation angle
        cos_elev = (a**2 + c**2 - b**2) / (2 * a * c)
        cos_elev = max(min(cos_elev, 1.0), -1.0)  # Clamp to valid range
        elevation = 90 - np.degrees(np.arccos(cos_elev))
        
        return elevation
    
    # Group by satellite and calculate features
    for satellite, group in tqdm(periods_df.groupby('Satellite'), desc="Extracting features"):
        # Load TLE data for this satellite
        tle_lines = load_tle_data(satellite)
        if tle_lines is None:
            continue
            
        # Calculate true elevation angles and altitudes
        true_elevations = []
        true_altitudes = []
        
        for _, row in group.iterrows():
            # Calculate satellite position at this time
            sat_pos = calculate_satellite_position(tle_lines, row['Start_Time'])
            
            # Calculate true elevation angle
            obs_pos = {'latitude': OBSERVER_LAT, 'longitude': OBSERVER_LON}
            true_elev = calculate_elevation_angle(sat_pos, obs_pos)
            
            true_elevations.append(true_elev)
            true_altitudes.append(sat_pos['altitude'])
        
        # Calculate features using true altitudes
        mean_altitude = np.mean(true_altitudes)
        min_altitude = np.min(true_altitudes)
        max_altitude = np.max(true_altitudes)
        std_altitude = np.std(true_altitudes)
        
        # Calculate altitude stability (inverse of standard deviation)
        altitude_stability = 1 / (std_altitude + 1e-6)  # Add small epsilon to avoid division by zero
        
        features.append({
            'Satellite': satellite,
            'Mean_Altitude': mean_altitude,
            'Altitude_Stability': altitude_stability,
            'Min_Altitude': min_altitude,
            'Max_Altitude': max_altitude,
            'Altitude_Range': max_altitude - min_altitude,
            'Mean_True_Elevation': np.mean(true_elevations),
            'Min_True_Elevation': np.min(true_elevations),
            'Max_True_Elevation': np.max(true_elevations)
        })
    
    features_df = pd.DataFrame(features)
    
    # Handle any NaN values by replacing with median
    for col in features_df.columns:
        if col != 'Satellite':
            features_df[col] = features_df[col].fillna(features_df[col].median())
    
    # Save features if file path provided
    if features_file:
        features_df.to_csv(features_file, index=False)
        print(f"Saved extracted features to {features_file}")
    
    return features_df

def find_optimal_clusters(features_df: pd.DataFrame, max_clusters: int = 5) -> int:
    """Find the optimal number of clusters using elbow method and silhouette score."""
    print("Finding optimal number of clusters...")
    
    # Prepare features for clustering (exclude satellite name)
    X = features_df[['Mean_Altitude', 'Altitude_Stability', 'Min_Altitude', 'Max_Altitude', 'Altitude_Range']].values
    
    # Test both 3-shell (Gen 2) and 5-shell (Gen 1) configurations
    n_clusters_list = [3, 5]
    inertia = []
    silhouette_scores = []
    
    for n_clusters in tqdm(n_clusters_list, desc="Analyzing clusters"):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    
    # Choose the configuration with the best silhouette score
    best_idx = np.argmax(silhouette_scores)
    optimal_clusters = n_clusters_list[best_idx]
    
    print(f"Optimal number of clusters: {optimal_clusters}")
    print(f"Silhouette scores: {dict(zip(n_clusters_list, silhouette_scores))}")
    
    return optimal_clusters

def classify_orbital_shells(periods_df: pd.DataFrame, features_df: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    """Classify satellites into orbital shells based on altitude features."""
    print("Classifying satellites into orbital shells...")
    
    # Prepare features for clustering
    X = features_df[['Mean_Altitude', 'Altitude_Stability', 'Min_Altitude', 'Max_Altitude', 'Altitude_Range']].values
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    features_df['Shell'] = kmeans.fit_predict(X)
    
    # Sort shells by mean altitude
    shell_order = features_df.groupby('Shell')['Mean_Altitude'].mean().sort_values().index
    shell_mapping = {old: new for new, old in enumerate(shell_order)}
    features_df['Shell'] = features_df['Shell'].map(shell_mapping)
    
    # Add generation classification based on altitude
    features_df['Generation'] = features_df['Mean_Altitude'].apply(
        lambda x: 'Gen 1' if x < 550 else 'Gen 2'
    )
    
    # Print shell information
    print("\nShell Information:")
    for shell in sorted(features_df['Shell'].unique()):
        shell_sats = features_df[features_df['Shell'] == shell]
        print(f"\nShell {shell}:")
        print(f"Number of satellites: {len(shell_sats)}")
        print(f"Mean altitude: {shell_sats['Mean_Altitude'].mean():.2f} km")
        print(f"Altitude range: {shell_sats['Min_Altitude'].min():.2f} - {shell_sats['Max_Altitude'].max():.2f} km")
        print(f"Generation: {shell_sats['Generation'].iloc[0]}")
    
    return features_df

def save_results(features_df: pd.DataFrame, output_file: str):
    """Save classification results to CSV file."""
    # Calculate statistics for each shell
    shell_stats = features_df.groupby(['Shell', 'Generation']).agg({
        'Mean_Altitude': ['mean', 'std', 'min', 'max'],
        'Altitude_Stability': 'mean',
        'Satellite': 'count'
    }).round(2)
    
    # Save results
    features_df.to_csv(output_file, index=False)
    print(f"\nSaved classification results to {output_file}")
    print("\nShell Statistics:")
    print(shell_stats)

def main():
    parser = argparse.ArgumentParser(description='Classify Starlink satellites into orbital shells')
    parser.add_argument('--periods', required=True, help='Path to connection periods CSV file')
    parser.add_argument('--output', required=True, help='Path to output CSV file')
    parser.add_argument('--features', default='extracted_features.csv', help='Path to save/load features')
    parser.add_argument('--force_extract', action='store_true', help='Force re-extraction of features')
    parser.add_argument('--max_clusters', type=int, default=5, help='Maximum number of clusters to test')
    
    args = parser.parse_args()
    
    # Load connection periods
    print(f"Loading connection periods from {args.periods}")
    periods_df = pd.read_csv(args.periods)
    periods_df['Start_Time'] = pd.to_datetime(periods_df['Start_Time'])
    periods_df['End_Time'] = pd.to_datetime(periods_df['End_Time'])
    
    # Check if we need to force re-extraction
    if not args.force_extract and os.path.exists(args.features):
        try:
            # Try to load existing features and check if they have the new column names
            existing_features = pd.read_csv(args.features)
            required_columns = ['Mean_Altitude', 'Altitude_Stability', 'Min_Altitude', 'Max_Altitude', 'Altitude_Range']
            if not all(col in existing_features.columns for col in required_columns):
                print("Existing features have old column names, forcing re-extraction")
                args.force_extract = True
        except Exception as e:
            print(f"Error checking existing features: {e}")
            args.force_extract = True
    
    # Extract features
    features_df = extract_distance_features(periods_df, args.features, args.force_extract)
    
    # Find optimal number of clusters
    n_clusters = find_optimal_clusters(features_df, args.max_clusters)
    
    # Classify satellites
    classified_df = classify_orbital_shells(periods_df, features_df, n_clusters)
    
    # Save results
    save_results(classified_df, args.output)

if __name__ == "__main__":
    main() 