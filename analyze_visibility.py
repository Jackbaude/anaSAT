#!/usr/bin/env python3
"""
Script to analyze satellite visibility data from satellite_visibility.json.
Generates statistics and visualizations about satellite visibility durations.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import logging
from skyfield.api import EarthSatellite, load, wgs84
import math
from sklearn.cluster import KMeans

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set the style for all plots
plt.style.use('default')
sns.set_theme(style="whitegrid")

class VisibilityAnalyzer:
    def __init__(self, json_file: str, lat_classification_file: str = None):
        """
        Initialize the analyzer with the visibility JSON file.

        Args:
            json_file (str): Path to the satellite_visibility.json file
            lat_classification_file (str): Path to the lat_classification file
        """
        self.json_file = json_file
        self.lat_classification_file = lat_classification_file
        self.data = self.load_data()
        self.durations_df = None
        self.visibility_df = None
        self.ts = load.timescale()  # Skyfield timescale object
        self.lat_classifications = self.load_lat_classifications() if lat_classification_file else None

    def load_data(self) -> dict:
        """Load and parse the JSON data file."""
        try:
            with open(self.json_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON file: {e}")
            raise

    def load_lat_classifications(self) -> pd.DataFrame:
        """Load and parse the lat classification file."""
        try:
            return pd.read_csv(self.lat_classification_file)
        except Exception as e:
            logger.error(f"Error loading lat classification file: {e}")
            return None

    def calculate_satellite_altitude(self, tle_line1: str, tle_line2: str, timestamp: datetime) -> float:
        """
        Calculate satellite altitude using TLE data.
        
        Args:
            tle_line1 (str): First line of TLE data
            tle_line2 (str): Second line of TLE data
            timestamp (datetime): Time for altitude calculation
            
        Returns:
            float: Satellite altitude in kilometers
        """
        try:
            sat = EarthSatellite(tle_line1, tle_line2)
            t = self.ts.from_datetime(timestamp)
            return wgs84.subpoint(sat.at(t)).elevation.km
        except Exception as e:
            logger.error(f"Error calculating satellite altitude: {e}")
            return None

    def calculate_durations(self) -> pd.DataFrame:
        """
        Calculate duration statistics for each satellite.
        
        Returns:
            pd.DataFrame: DataFrame containing duration statistics
        """
        durations = []
        
        # Process each satellite's FOV periods
        for sat, periods in self.data['fov_durations'].items():
            for start_str, end_str in periods:
                start = datetime.fromisoformat(start_str)
                end = datetime.fromisoformat(end_str)
                duration = (end - start).total_seconds()
                durations.append({
                    'satellite': sat,
                    'start_time': start,
                    'end_time': end,
                    'duration_seconds': duration,
                    'duration_minutes': duration / 60
                })
        
        self.durations_df = pd.DataFrame(durations)
        return self.durations_df

    def analyze_visibility(self) -> pd.DataFrame:
        """
        Analyze visibility data to create a timeline of visible satellites.
        
        Returns:
            pd.DataFrame: DataFrame containing visibility timeline
        """
        visibility_data = []
        
        for timestamp_str, satellites in self.data['visibility'].items():
            timestamp = datetime.fromisoformat(timestamp_str)
            for sat in satellites:
                # Calculate altitude from TLE data
                altitude = self.calculate_satellite_altitude(
                    sat['tle_line1'],
                    sat['tle_line2'],
                    timestamp
                )
                
                visibility_data.append({
                    'timestamp': timestamp,
                    'satellite': sat['satellite'],
                    'altitude_deg': sat['altitude_deg'],
                    'azimuth_deg': sat['azimuth_deg'],
                    'tilt_deg': sat['tilt_deg'],
                    'rotation_deg': sat['rotation_deg'],
                    'altitude_km': altitude
                })
        
        self.visibility_df = pd.DataFrame(visibility_data)
        return self.visibility_df

    def generate_statistics(self) -> dict:
        """
        Generate summary statistics for satellite visibility durations.
        
        Returns:
            dict: Dictionary containing summary statistics
        """
        if self.durations_df is None:
            self.calculate_durations()
        
        stats = {
            'total_satellites': self.durations_df['satellite'].nunique(),
            'total_visibility_periods': len(self.durations_df),
            'mean_duration_minutes': self.durations_df['duration_minutes'].mean(),
            'median_duration_minutes': self.durations_df['duration_minutes'].median(),
            'min_duration_minutes': self.durations_df['duration_minutes'].min(),
            'max_duration_minutes': self.durations_df['duration_minutes'].max(),
            'std_duration_minutes': self.durations_df['duration_minutes'].std(),
            'satellites_per_timestamp': self.visibility_df.groupby('timestamp')['satellite'].nunique().mean()
        }
        
        return stats

    def plot_altitude_timeline(self, output_dir: str):
        """
        Create a plot showing the relationship between satellite altitude and visibility duration.
        
        Args:
            output_dir (str): Directory to save the plots
        """
        if self.visibility_df is None:
            self.analyze_visibility()
        if self.durations_df is None:
            self.calculate_durations()
        
        # Merge visibility and duration data
        merged_df = pd.merge(
            self.visibility_df[['satellite', 'altitude_km']].drop_duplicates(),
            self.durations_df.groupby('satellite')['duration_minutes'].mean().reset_index(),
            on='satellite',
            how='inner'
        )
        
        plt.figure(figsize=(12, 6))
        
        # Create scatter plot
        sns.scatterplot(
            data=merged_df,
            x='altitude_km',
            y='duration_minutes',
            alpha=0.6
        )
        
        # Calculate and plot regression line
        x = merged_df['altitude_km']
        y = merged_df['duration_minutes']
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), "r--", linewidth=2)
        
        # Add R-squared value
        r2 = np.corrcoef(x, y)[0,1]**2
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', 
                transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.title('Relationship Between Satellite Altitude and Visibility Duration')
        plt.xlabel('Altitude (km)')
        plt.ylabel('Average Visibility Duration (minutes)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/altitude_vs_duration.png')
        plt.close()
        
        # Print correlation statistics
        correlation = merged_df['altitude_km'].corr(merged_df['duration_minutes'])
        logger.info(f"\nAltitude vs Duration Correlation: {correlation:.3f}")
        logger.info(f"R-squared value: {r2:.3f}")
        logger.info(f"Regression equation: y = {z[0]:.3f}x + {z[1]:.3f}")

    def plot_classified_distribution(self, output_dir: str):
        """
        Create plots showing the distribution of visibility durations by classification.
        
        Args:
            output_dir (str): Directory to save the plots
        """
        if self.visibility_df is None or self.lat_classifications is None:
            return
        
        # Merge visibility data with classifications
        merged_df = pd.merge(
            self.visibility_df,
            self.lat_classifications,
            on='satellite',
            how='left'
        )
        
        # Create distribution plots
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=merged_df, x='classification', y='altitude_km')
        plt.title('Satellite Altitude Distribution by Classification')
        plt.xlabel('Classification')
        plt.ylabel('Altitude (km)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/classified_altitude_distribution.png')
        plt.close()
        
        # Create duration distribution by classification
        if self.durations_df is not None:
            merged_durations = pd.merge(
                self.durations_df,
                self.lat_classifications,
                on='satellite',
                how='left'
            )
            
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=merged_durations, x='classification', y='duration_minutes')
            plt.title('Visibility Duration Distribution by Classification')
            plt.xlabel('Classification')
            plt.ylabel('Duration (minutes)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/classified_duration_distribution.png')
            plt.close()

    def plot_duration_distribution(self, output_dir: str):
        """
        Create plots showing the distribution of visibility durations.
        
        Args:
            output_dir (str): Directory to save the plots
        """
        if self.durations_df is None:
            self.calculate_durations()
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 1. Duration Distribution Histogram
        plt.figure(figsize=(12, 6))
        sns.histplot(data=self.durations_df, x='duration_minutes', bins=50)
        plt.title('Distribution of Satellite Visibility Durations')
        plt.xlabel('Duration (minutes)')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/duration_distribution.png')
        plt.close()
        
        # 2. Timeline Plot of Satellite Visibility
        plt.figure(figsize=(15, 8))
        
        # Get unique satellites and sort them
        satellites = sorted(self.durations_df['satellite'].unique())
        
        # Create y-axis positions for each satellite
        y_positions = {sat: i for i, sat in enumerate(satellites)}
        
        # Plot each satellite's visibility periods
        for sat in satellites:
            sat_data = self.durations_df[self.durations_df['satellite'] == sat]
            y_pos = y_positions[sat]
            
            # Plot each visibility period as a horizontal line
            for _, row in sat_data.iterrows():
                plt.hlines(y=y_pos, 
                          xmin=row['start_time'], 
                          xmax=row['end_time'], 
                          linewidth=2)
        
        # Customize the plot
        plt.yticks(range(len(satellites)), satellites, fontsize=8)
        plt.title('Satellite Visibility Timeline')
        plt.xlabel('Time')
        plt.ylabel('Satellite')
        plt.grid(True, axis='x')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, fontsize=8)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        plt.savefig(f'{output_dir}/satellite_timeline.png')
        plt.close()
        
        # 3. Time Series of Number of Visible Satellites
        if self.visibility_df is None:
            self.analyze_visibility()
        
        visible_counts = self.visibility_df.groupby('timestamp')['satellite'].nunique()
        plt.figure(figsize=(15, 6))
        visible_counts.plot()
        plt.title('Number of Visible Satellites Over Time')
        plt.xlabel('Time')
        plt.ylabel('Number of Visible Satellites')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/visible_satellites_over_time.png')
        plt.close()

    def save_statistics(self, output_dir: str):
        """
        Save statistics to a CSV file.
        
        Args:
            output_dir (str): Directory to save the statistics
        """
        if self.durations_df is None:
            self.calculate_durations()
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save duration statistics
        duration_stats = self.durations_df.groupby('satellite').agg({
            'duration_seconds': ['count', 'mean', 'std', 'min', 'max'],
            'duration_minutes': ['mean', 'std', 'min', 'max']
        }).round(2)
        
        duration_stats.to_csv(f'{output_dir}/duration_statistics.csv')
        
        # Save summary statistics
        summary_stats = self.generate_statistics()
        pd.Series(summary_stats).to_csv(f'{output_dir}/summary_statistics.csv')

    def classify_altitudes(self, n_clusters: int = 3) -> pd.DataFrame:
        """
        Classify satellites into altitude bands using KMeans clustering.
        
        Args:
            n_clusters (int): Number of altitude bands to create
            
        Returns:
            pd.DataFrame: DataFrame with added altitude classification
        """
        if self.visibility_df is None:
            self.analyze_visibility()
        
        # Filter out any rows with missing altitude data
        df_filtered = self.visibility_df[['satellite', 'altitude_km']].dropna().copy()
        
        # Reshape altitude data for KMeans
        altitudes = df_filtered['altitude_km'].values.reshape(-1, 1)
        
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
        labels = kmeans.fit_predict(altitudes)
        
        # Sort by cluster centers and relabel
        sorted_centers = np.argsort(kmeans.cluster_centers_.flatten())
        label_map = {old: new for new, old in enumerate(sorted_centers)}
        df_filtered['altitude_band'] = [label_map[l] for l in labels]
        
        # Create descriptive labels
        band_names = {0: 'Low Altitude', 1: 'Mid Altitude', 2: 'High Altitude'}
        df_filtered['altitude_band_label'] = df_filtered['altitude_band'].map(band_names)
        
        # Merge back with original data
        self.visibility_df = pd.merge(
            self.visibility_df,
            df_filtered[['satellite', 'altitude_band', 'altitude_band_label']],
            on='satellite',
            how='left'
        )
        
        return self.visibility_df

    def plot_altitude_distribution(self, output_dir: str):
        """
        Plot the distribution of satellite altitudes with mean and median lines.
        
        Args:
            output_dir (str): Directory to save the plots
        """
        if self.visibility_df is None:
            self.analyze_visibility()
        
        plt.figure(figsize=(12, 6))
        
        # Create a combined histogram and KDE plot
        sns.histplot(data=self.visibility_df, x='altitude_km', bins=30, kde=True)
        
        # Add vertical lines for mean and median
        mean_alt = self.visibility_df['altitude_km'].mean()
        median_alt = self.visibility_df['altitude_km'].median()
        
        plt.axvline(mean_alt, color='red', linestyle='--', label=f'Mean: {mean_alt:.1f} km')
        plt.axvline(median_alt, color='green', linestyle='--', label=f'Median: {median_alt:.1f} km')
        
        plt.title('Distribution of Satellite Altitudes')
        plt.xlabel('Altitude (km)')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/altitude_distribution.png')
        plt.close()
        
        # Print altitude statistics
        logger.info("\nAltitude Statistics:")
        logger.info(f"Mean altitude: {mean_alt:.2f} km")
        logger.info(f"Median altitude: {median_alt:.2f} km")
        logger.info(f"Standard deviation: {self.visibility_df['altitude_km'].std():.2f} km")
        logger.info(f"Min altitude: {self.visibility_df['altitude_km'].min():.2f} km")
        logger.info(f"Max altitude: {self.visibility_df['altitude_km'].max():.2f} km")

    def plot_altitude_bands(self, output_dir: str):
        """
        Create plots showing satellite visibility by altitude bands.
        
        Args:
            output_dir (str): Directory to save the plots
        """
        if 'altitude_band_label' not in self.visibility_df.columns:
            self.classify_altitudes()
        
        # Plot 1: Altitude bands over time
        plt.figure(figsize=(15, 8))
        sns.scatterplot(
            data=self.visibility_df,
            x='timestamp',
            y='altitude_km',
            hue='altitude_band_label',
            palette='Set2'
        )
        plt.title('Satellite Altitudes Over Time by Classification')
        plt.xlabel('Time')
        plt.ylabel('Altitude (km)')
        plt.legend(title='Altitude Band')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/altitude_bands_timeline.png')
        plt.close()
        
        # Plot 2: Box plot of altitudes by band
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=self.visibility_df,
            x='altitude_band_label',
            y='altitude_km',
            palette='Set2'
        )
        plt.title('Altitude Distribution by Band')
        plt.xlabel('Altitude Band')
        plt.ylabel('Altitude (km)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/altitude_bands_distribution.png')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze satellite visibility data')
    parser.add_argument('--input', required=True, help='Path to satellite_visibility.json')
    parser.add_argument('--output_dir', default='analysis_results', help='Directory for output files')
    parser.add_argument('--lat_classification', help='Path to lat classification file')
    parser.add_argument('--n_clusters', type=int, default=3, help='Number of altitude bands for classification')
    
    args = parser.parse_args()
    
    analyzer = VisibilityAnalyzer(args.input, args.lat_classification)
    
    # Calculate durations and analyze visibility
    analyzer.calculate_durations()
    analyzer.analyze_visibility()
    
    # Generate and print summary statistics
    stats = analyzer.generate_statistics()
    logger.info("Summary Statistics:")
    for key, value in stats.items():
        logger.info(f"{key}: {value}")
    
    # Generate plots and save statistics
    analyzer.plot_duration_distribution(args.output_dir)
    analyzer.plot_altitude_timeline(args.output_dir)
    analyzer.plot_altitude_distribution(args.output_dir)
    analyzer.plot_altitude_bands(args.output_dir)
    if args.lat_classification:
        analyzer.plot_classified_distribution(args.output_dir)
    analyzer.save_statistics(args.output_dir)
    
    logger.info(f"Analysis complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 