#!/usr/bin/env python3
"""
Script to analyze satellite connections and latency data.

This script processes satellite connection data, including:
- Satellite position and connection data
- Network latency measurements (ping)
- gRPC status information
- TLE (Two-Line Element) data for satellite positions

The script combines these data sources to analyze satellite connections,
calculate altitudes, and track connection periods with various metrics.
"""

import pandas as pd
import os
import re
from datetime import datetime, timezone, timedelta
import logging
import argparse
import numpy as np
import glob
from typing import List, Dict, Any, Optional, Callable
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from skyfield.api import EarthSatellite, load, wgs84
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import math
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class SatelliteDataProcessor:
    """
    A class to process and analyze satellite connection data.

    This class handles the loading, processing, and analysis of various data sources:
    - Satellite connection data (position, elevation, azimuth)
    - Network latency measurements
    - gRPC status information
    - TLE data for satellite positions

    The class combines these data sources to create comprehensive analysis of satellite
    connections, including connection periods, altitudes, and performance metrics.

    Attributes:
        output_dir (str): Directory where processed data will be saved
        ts (Timescale): Skyfield timescale object for satellite calculations
        file_regex (Pattern): Compiled regex for matching satellite data files
        max_workers (int): Number of workers for parallel processing
        merge_tolerance (Timedelta): Time tolerance for merging dataframes
    """

    def __init__(self, start_time: datetime, end_time: datetime, latitude: float, longitude:float, altitude: float, output_dir: str = 'output'):
        """
        Initialize the SatelliteDataProcessor.

        Args:
            output_dir (str): Directory where processed data will be saved
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize commonly used values
        self.ts = load.timescale()  # Skyfield timescale object used in multiple methods
        self.file_regex = re.compile(r'serving_satellite_data-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.csv')
        self.max_workers = 8  # Number of workers for parallel processing
        self.merge_tolerance = pd.Timedelta(seconds=1)  # Tolerance for merging dataframes

        self.tle_cache = {}  # In-memory cache for TLE data

        self.start_time = start_time
        self.end_time = end_time
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
    

    @staticmethod
    def parse_timestamp(ts: str) -> datetime:
        """
        Parse timestamp string into datetime object.

        Args:
            ts (str): Timestamp string in format 'YYYY-MM-DD-HH-MM-SS'

        Returns:
            datetime: Parsed datetime object with UTC timezone
        """
        return datetime.strptime(ts, "%Y-%m-%d-%H-%M-%S").replace(tzinfo=timezone.utc)
    
    def generate_handover_timestamps(self):
        """
        Generate timestamps at 12, 27, 42, and 57 seconds of each minute between start and end time.
        """
        current_time = self.start_time
        handover_seconds = [12, 27, 42, 57]
        timestamps = []

        while current_time <= self.end_time:
            for sec in handover_seconds:
                ts = current_time.replace(second=sec, microsecond=0)
                if ts <= self.end_time:
                    timestamps.append(ts)
            current_time += timedelta(minutes=1)

        logger.info(f"Generated {len(timestamps)} handover timestamps")
        return timestamps
    
    def load_tle_data(self, timestamp: datetime) -> pd.DataFrame:
        """
        Load TLE (Two-Line Element) data for a specific hour.

        Args:
            timestamp (datetime): The timestamp for which to load TLE data

        Returns:
            pd.DataFrame: DataFrame containing TLE data with columns:
                - satellite_name: Name of the satellite
                - tle_line1: First line of TLE data
                - tle_line2: Second line of TLE data
                - timestamp: When the TLE data was recorded
        """
        # Round down to the nearest hour
        hour_timestamp = timestamp.replace(minute=0, second=0, microsecond=0)
        date_str = hour_timestamp.strftime('%Y-%m-%d')
        tle_file = os.path.join('data', 'TLE', date_str, f"starlink-tle-{date_str}-{hour_timestamp.strftime('%H-%M-%S')}.txt")
        tle_records = []
        
        if os.path.exists(tle_file):
            try:
                with open(tle_file, 'r') as f:
                    lines = f.readlines()
                    for i in range(0, len(lines), 3):
                        if i + 2 < len(lines):
                            tle_records.append({
                                'satellite_name': lines[i].strip(),
                                'tle_line1': lines[i+1].strip(),
                                'tle_line2': lines[i+2].strip(),
                                'timestamp': hour_timestamp
                            })
            except Exception as e:
                logger.error(f"Error loading TLE file {tle_file}: {e}")
        
        return pd.DataFrame(tle_records) if tle_records else pd.DataFrame(columns=['satellite_name', 'tle_line1', 'tle_line2', 'timestamp'])

    def get_matching_serving_satellite_files(self, start_time: datetime, end_time: datetime) -> List[tuple[datetime, Path]]:
        """
        Find satellite data files that fall within the specified time window.

        Args:
            start_time (datetime): Start of the time window
            end_time (datetime): End of the time window

        Returns:
            List[tuple[datetime, Path]]: List of tuples containing (file_time, file_path)
            for each matching file
        """
        files = sorted(Path("data").glob("serving_satellite_data-*.csv"))
        matched = []
        for f in files:
            match = self.file_regex.search(f.name)
            if match:
                file_time = datetime.strptime(match.group(1), "%Y-%m-%d-%H-%M-%S").replace(tzinfo=timezone.utc)
                if start_time <= file_time <= end_time:
                    matched.append((file_time, f))
        return matched

    def process_satellite_file(self, file_time: datetime, file_path: Path, ts: Any, load_tle_data_fn: Callable[[datetime], pd.DataFrame]) -> pd.DataFrame:
        """
        Process a single satellite data file, adding TLE data and computing altitudes.

        Args:
            file_time (datetime): Timestamp associated with the file
            file_path (Path): Path to the satellite data file
            ts (Any): Skyfield timescale object
            load_tle_data_fn (Callable): Function to load TLE data for a timestamp

        Returns:
            pd.DataFrame: Processed satellite data with added TLE and altitude information
        """
        tle_data = load_tle_data_fn(file_time)

        try:
            # Read CSV file with proper datetime parsing
            df = pd.read_csv(file_path, parse_dates=['Timestamp'])
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)

            # Add TLE data columns using merge
            if not tle_data.empty:
                tle_data = tle_data.rename(columns={
                    'satellite_name': 'Connected_Satellite',
                    'tle_line1': 'TLE_Line1',
                    'tle_line2': 'TLE_Line2',
                    'timestamp': 'TLE_Timestamp'
                })
                df = df.merge(tle_data, on='Connected_Satellite', how='left')

            def compute_altitude(row) -> Optional[float]:
                """
                Compute satellite altitude using TLE data.

                Args:
                    row: DataFrame row containing satellite and TLE data

                Returns:
                    Optional[float]: Satellite altitude in kilometers, or None if calculation fails
                """
                try:
                    if pd.notna(row.TLE_Line1) and pd.notna(row.TLE_Line2):
                        sat = EarthSatellite(row.TLE_Line1, row.TLE_Line2, row.Connected_Satellite, ts)
                        t = ts.from_datetime(row.Timestamp.to_pydatetime())
                        return wgs84.subpoint(sat.at(t)).elevation.km
                except Exception as e:
                    print(f"Altitude error for {row.Connected_Satellite}: {e}")
                return None

            # Compute altitudes in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                df['Altitude_km'] = list(executor.map(compute_altitude, df.itertuples(index=False)))

            return df

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return pd.DataFrame()

    def load_serving_satellite_data(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Load and combine satellite data files within the time window.

        Args:
            start_time (datetime): Start of the time window
            end_time (datetime): End of the time window

        Returns:
            pd.DataFrame: Combined satellite data
        """
        print("Loading serving satellite data...")
        matched_files = self.get_matching_serving_satellite_files(start_time, end_time)

        if not matched_files:
            print("No matching serving satellite files found")
            return pd.DataFrame()

        all_data = []
        for file_time, file_path in tqdm(matched_files, desc="Loading satellite files"):
            df = self.process_satellite_file(file_time, file_path, self.ts, self.load_tle_data)
            if not df.empty:
                # Filter data to only include rows within the time window
                df = df[(df['Timestamp'] >= start_time) & (df['Timestamp'] <= end_time)]
                all_data.append(df)

        if not all_data:
            return pd.DataFrame()

        combined_df = pd.concat(all_data, ignore_index=True).sort_values('Timestamp')
        return combined_df

    def merge_satellite_data(self, satellite_df: pd.DataFrame, ping_df: pd.DataFrame, grpc_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge satellite data with ping and gRPC data.

        Args:
            satellite_df (pd.DataFrame): DataFrame containing satellite data
            ping_df (pd.DataFrame): DataFrame containing ping latency data
            grpc_df (pd.DataFrame): DataFrame containing gRPC status data

        Returns:
            pd.DataFrame: Combined satellite data with ping and gRPC information
        """
        if satellite_df.empty:
            return pd.DataFrame()

        # Merge ping data
        if not ping_df.empty:
            print("Merging ping data...")
            satellite_df = pd.merge_asof(
                satellite_df.sort_values('Timestamp'),
                ping_df.sort_values('Timestamp'),
                on='Timestamp',
                direction='nearest',
                tolerance=self.merge_tolerance
            )

        # Merge gRPC data
        if not grpc_df.empty:
            print("Merging GRPC data...")
            grpc_df = grpc_df.rename(columns={'timestamp': 'Timestamp'})
            grpc_columns = [col for col in grpc_df.columns if col != 'Timestamp']
            satellite_df = pd.merge_asof(
                satellite_df.sort_values('Timestamp'),
                grpc_df[['Timestamp'] + grpc_columns].sort_values('Timestamp'),
                on='Timestamp',
                direction='nearest',
                tolerance=self.merge_tolerance
            )

        return satellite_df

    def load_single_ping_file(self, file_path: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Load ping data from a single file, filtering timestamps within the range.

        Args:
            file_path (str): Path to the ping data file
            start_time (datetime): Start of the time window
            end_time (datetime): End of the time window

        Returns:
            pd.DataFrame: DataFrame containing ping latency data
        """
        try:
            # Pre-compile regex pattern
            pattern = re.compile(r'\[(\d+\.\d+)\].*time=(\d+\.\d+)\s+ms')
            
            # Read file content
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Find all matches at once
            matches = pattern.findall(content)
            
            if not matches:
                return pd.DataFrame(columns=['Timestamp', 'Latency_ms'])
            
            # Convert matches to DataFrame using vectorized operations
            df = pd.DataFrame(matches, columns=['timestamp', 'latency'])
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            df['latency'] = pd.to_numeric(df['latency'])
            
            # Convert timestamps to datetime using vectorized operation
            df['Timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
            df['Latency_ms'] = df['latency']
            
            # Filter by time range (inclusive)
            mask = (df['Timestamp'] >= start_time) & (df['Timestamp'] <= end_time)
            df = df[mask][['Timestamp', 'Latency_ms']]
            
            return df
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return pd.DataFrame(columns=['Timestamp', 'Latency_ms'])

    def load_ping_data(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Load and combine all ping data files within the time window.

        Args:
            start_time (datetime): Start of the time window
            end_time (datetime): End of the time window

        Returns:
            pd.DataFrame: Combined ping latency data
        """
        all_data = []
        total_hours = int((end_time - start_time).total_seconds() / 3600) + 1

        current_time = start_time
        with tqdm(total=total_hours, desc="Loading ping files") as pbar:
            while current_time <= end_time:
                date_str = current_time.strftime('%Y-%m-%d')
                ping_file = os.path.join('data', 'latency', date_str, f"ping-100ms-{date_str}-{current_time.strftime('%H-%M-%S')}.txt")
                
                if os.path.exists(ping_file):
                    df = self.load_single_ping_file(ping_file, start_time, end_time)
                    if not df.empty:
                        all_data.append(df)
                        pbar.set_postfix({'file': os.path.basename(ping_file)})

                current_time += timedelta(hours=1)
                pbar.update(1)

        if not all_data:
            print("Warning: No ping data files found in the specified time window")
            return pd.DataFrame(columns=['Timestamp', 'Latency_ms'])

        print("Combining ping data...")
        combined_df = pd.concat(all_data, ignore_index=True).sort_values('Timestamp')
        # Ensure final data is within time range (inclusive)
        combined_df = combined_df[(combined_df['Timestamp'] >= start_time) & (combined_df['Timestamp'] <= end_time)]
        return combined_df

    def load_single_grpc_file(self, timestamp: datetime, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Load a single GRPC status file for a specific hour.

        Args:
            timestamp (datetime): The timestamp for the hour to load
            start_time (datetime): Start of the time window
            end_time (datetime): End of the time window

        Returns:
            pd.DataFrame: GRPC status data for the specified hour
        """
        date_str = timestamp.strftime('%Y-%m-%d')
        grpc_file = os.path.join('data', 'grpc', date_str, f"GRPC_STATUS-{date_str}-{timestamp.strftime('%H-%M-%S')}.csv")
        
        if os.path.exists(grpc_file):
            try:
                df = pd.read_csv(grpc_file)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
                # Filter data to only include rows within the time window (inclusive)
                df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
                return df
            except Exception as e:
                print(f"Error loading {grpc_file}: {e}")
                return pd.DataFrame()
        return pd.DataFrame()

    def load_grpc_data(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Load and combine all GRPC status files within the time window.

        Args:
            start_time (datetime): Start of the time window
            end_time (datetime): End of the time window

        Returns:
            pd.DataFrame: Combined gRPC status data
        """
        all_data = []
        total_hours = int((end_time - start_time).total_seconds() / 3600) + 1
        
        current_time = start_time
        with tqdm(total=total_hours, desc="Loading GRPC files") as pbar:
            while current_time <= end_time:
                df = self.load_single_grpc_file(current_time, start_time, end_time)
                if not df.empty:
                    all_data.append(df)
                    pbar.set_postfix({'file': current_time.strftime('%Y-%m-%d-%H-%M-%S')})
                
                current_time += timedelta(hours=1)
                pbar.update(1)
        
        print("Combining GRPC data...")
        combined_df = pd.concat(all_data, ignore_index=True).sort_values('timestamp')
        # Ensure final data is within time range (inclusive)
        combined_df = combined_df[(combined_df['timestamp'] >= start_time) & (combined_df['timestamp'] <= end_time)]
        return combined_df

    def create_connection_periods_csv(self, serving_df: pd.DataFrame, output_file: str):
        """
        Create a CSV file with unique satellite connection periods.

        This method analyzes the serving satellite data to identify distinct connection
        periods for each satellite, calculating various metrics for each period.

        Args:
            serving_df (pd.DataFrame): DataFrame containing serving satellite data
            output_file (str): Path where the connection periods CSV will be saved

        Returns:
            pd.DataFrame: DataFrame containing connection period data
        """
        print("Creating connection periods CSV...")
        periods = []
        current_sat = None
        start_time = None
        current_data = []

        rows = serving_df.to_dict('records')
        
        def append_period(satellite, start, end, data):
            """
            Create a period entry with calculated metrics.

            Args:
                satellite (str): Satellite identifier
                start (datetime): Start time of the period
                end (datetime): End time of the period
                data (list): List of data points in the period

            Returns:
                dict: Dictionary containing period metrics
            """
            distances = [d['Distance'] for d in data if pd.notna(d['Distance'])]
            altitudes = [d['Altitude_km'] for d in data if pd.notna(d['Altitude_km'])]
            latencies = [d['Latency_ms'] for d in data if pd.notna(d['Latency_ms'])]
            pop_latencies = [d['popPingLatencyMs'] for d in data if pd.notna(d['popPingLatencyMs'])]

            first_entry = data[0]
            return {
                'Satellite': satellite,
                'Start_Time': start,
                'End_Time': end,
                'Duration_Seconds': (end - start).total_seconds(),

                'Max_Distance': max(distances) if distances else None,
                'Min_Distance': min(distances) if distances else None,
                'First_Distance': distances[0] if distances else None,
                'Last_Distance': distances[-1] if distances else None,
                'Mean_Distance': np.mean(distances) if distances else None,
                'Std_Distance': np.std(distances) if distances else None,

                'Mean_Altitude_km': np.mean(altitudes) if altitudes else None,
                'Mean_Latency_ms': np.mean(latencies) if latencies else None,
                'Mean_popPingLatencyMs': np.mean(pop_latencies) if pop_latencies else None,

                'TLE_Line1': first_entry.get('TLE_Line1'),
                'TLE_Line2': first_entry.get('TLE_Line2'),
                'TLE_Timestamp': first_entry.get('TLE_Timestamp')
            }

        print("Processing connection periods...")

        with tqdm(total=len(rows), desc="Analyzing connections") as pbar:
            for i, row in enumerate(rows):
                satellite = row['Connected_Satellite']
                timestamp = row['Timestamp']

                if current_sat is None:
                    current_sat = satellite
                    start_time = timestamp
                    current_data = [row]
                elif satellite == current_sat:
                    current_data.append(row)
                else:
                    # Satellite switched, finalize current period
                    end_time = current_data[-1]['Timestamp']
                    periods.append(append_period(current_sat, start_time, end_time, current_data))

                    # Start new period
                    current_sat = satellite
                    start_time = timestamp
                    current_data = [row]

                pbar.update(1)

        # Handle final period
        if current_sat is not None and current_data:
            end_time = current_data[-1]['Timestamp']
            periods.append(append_period(current_sat, start_time, end_time, current_data))

        print("Saving connection periods...")
        periods_df = pd.DataFrame(periods)
        periods_df.to_csv(output_file, index=False)
        print(f"Saved connection periods to {output_file}")
        return periods_df
    
    def get_satellite_position(self, tle_data, timestamp):
        """
        Compute satellite position (alt, az) using TLE and Skyfield.

        Args:
            tle_data (dict): Dictionary containing TLE data
            timestamp (datetime): Timestamp for position calculation

        Returns:
            tuple: (altitude_degrees, azimuth_degrees)
        """
        try:
            sat = EarthSatellite(tle_data['tle_line1'], tle_data['tle_line2'], tle_data.get('satellite_name', 'Unknown'), self.ts)
            t = self.ts.from_datetime(timestamp)
            observer = wgs84.latlon(self.latitude, self.longitude, self.altitude)
            
            difference = sat - observer
            topocentric = difference.at(t)
            alt, az, _ = topocentric.altaz()
            
            return alt.degrees, az.degrees
        except Exception as e:
            logger.error(f"Error calculating satellite position: {e}")
            return None, None
    def is_valid_satellite(self, alt, az, tilt_deg, rotation_deg, fov_azimuth, fov_elevation):
        # """
        # Check if a satellite is within the valid field of view.

        # Args:
        #     alt (float): Satellite altitude in degrees
        #     az (float): Satellite azimuth in degrees
        #     tilt_deg (float): Dish tilt angle in degrees
        #     rotation_deg (float): Dish rotation angle in degrees
        #     fov_azimuth (float): Field of view azimuth in degrees
        #     fov_elevation (float): Field of view elevation in degrees

        # Returns:
        #     bool: True if satellite is within valid FOV, False otherwise
        # """
        # if alt is None or az is None:
        #     return False
            
        # # Rotate azimuth based on dish rotation
        # adjusted_azimuth = (az + rotation_deg) % 360
        
        # # Check if satellite is within FOV azimuth range
        # azimuth_range = 110  # Width of the FOV in degrees
        # if adjusted_azimuth < (fov_azimuth - azimuth_range) or adjusted_azimuth > (fov_azimuth + azimuth_range):
        #     return False
        
        # # Check if satellite is within FOV elevation range
        # fov_max_elevation = 90 - tilt_deg  # Max elevation based on tilt
        # fov_min_elevation = 20  # Minimum elevation (adjust as necessary)
        # if alt < fov_min_elevation or alt > fov_max_elevation:
        #     return False
        
        # return True
        base_radius = 60
        if alt is None or az is None:
            return False

        # Ellipse parameters
        center_x = tilt_deg
        center_y = 0
        x_radius = base_radius
        y_radius = math.sqrt(base_radius**2 - tilt_deg**2)
        
        # Satellite position in plot coordinates
        r = 90 - alt
        theta = math.radians(az)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        
        # Rotate point back to FOV frame
        angle_rad = math.radians(-rotation_deg)
        x_rot = x * math.cos(angle_rad) - y * math.sin(angle_rad)
        y_rot = x * math.sin(angle_rad) + y * math.cos(angle_rad)

        # Translate to ellipse center
        dx = x_rot - center_x
        dy = y_rot - center_y

        # Check if inside ellipse equation
        inside = (dx**2 / x_radius**2) + (dy**2 / y_radius**2) <= 1
        return inside
 

    def compute_visibility_at_handovers(self, frame_type=2):
        """
        For each timestamp, compute visible satellites using dish parameters from GRPC data.
        Loads combined serving satellite data if not already created.

        Returns:
            dict: Dictionary mapping timestamps to lists of visible satellites with their positions
        """
        # First, ensure we have the combined serving satellite data
        combined_file = os.path.join(self.output_dir, 'combined_serving_satellite.csv')
        if not os.path.exists(combined_file):
            logger.info("Combined serving satellite file not found, processing data...")
            self.process_data(self.start_time, self.end_time)
        
        # Load the combined data
        try:
            serving_df = pd.read_csv(combined_file)
            serving_df['Timestamp'] = pd.to_datetime(serving_df['Timestamp'])
            logger.info(f"Loaded combined serving satellite data with {len(serving_df)} records")
        except Exception as e:
            logger.error(f"Error loading combined serving satellite data: {e}")
            return {}

        timestamps = self.generate_handover_timestamps()
        visibility_results = {}
        
        # Process each handover timestamp
        for ts in tqdm(timestamps, desc="Computing visibility"):
            # Get the closest GRPC data point for this timestamp
            mask = serving_df['Timestamp'] <= ts
            if not mask.any():
                logger.warning(f"No GRPC data available before timestamp {ts}")
                continue
                
            grpc_data = serving_df[mask].iloc[-1]
            
            # Get dish parameters from GRPC data
            tilt_deg = grpc_data.get('tiltAngleDeg', 0)
            rotation_deg = grpc_data.get('boresightAzimuthDeg', 0)

            fov_azimuth = grpc_data.get('desiredBoresightAzimuthDeg', 0)
            fov_elevation = grpc_data.get('desiredBoresightElevationDeg', 0)
            
            # Get TLE data for this timestamp (will load the hour's data)
            tle_data = self.load_tle_data(ts)
            if tle_data.empty:
                logger.warning(f"No TLE data available for hour containing timestamp {ts}")
                continue
            
            visible_sats = []
            for _, sat_data in tle_data.iterrows():
                # Calculate satellite position
                alt, az = self.get_satellite_position(sat_data, ts)
                
                # Check if satellite is within valid FOV
                if self.is_valid_satellite(alt, az, tilt_deg, rotation_deg, fov_azimuth, fov_elevation):
                    visible_sats.append({
                        'satellite': sat_data['satellite_name'],
                        'altitude_deg': alt,
                        'azimuth_deg': az,
                        'tilt_deg': tilt_deg,
                        'rotation_deg': rotation_deg,
                        'fov_azimuth': fov_azimuth,
                        'fov_elevation': fov_elevation,
                        'tle_line1': sat_data['tle_line1'],
                        'tle_line2': sat_data['tle_line2']
                    })
            
            if visible_sats:
                visibility_results[ts.isoformat()] = visible_sats
                logger.info(f"{ts}: {len(visible_sats)} satellites visible")

        # Save visibility results to JSON
        if visibility_results:
            import json
            output_file = os.path.join(self.output_dir, 'satellite_visibility.json')
            with open(output_file, 'w') as f:
                json.dump(visibility_results, f, indent=2)
            logger.info(f"Saved visibility results to {output_file}")

        return visibility_results
     
    def process_data(self, start_time: datetime, end_time: datetime):
        """
        Process all data within the specified time window.

        This method orchestrates the entire data processing pipeline:
        1. Loads and combines ping data
        2. Loads and combines gRPC data
        3. Loads and processes satellite data
        4. Creates connection period analysis
        5. Computes satellite visibility
        6. Saves all processed data to CSV and JSON files

        Args:
            start_time (datetime): Start of the time window
            end_time (datetime): End of the time window
        """
        print(f"Processing data from {start_time} to {end_time}")
        
        try:
            # Load and combine ping files first
            print("Loading ping data...")
            ping_df = self.load_ping_data(start_time, end_time)
            if not ping_df.empty:
                ping_df.to_csv(os.path.join(self.output_dir, 'combined_ping_data.csv'), index=False)
                print(f"Saved ping data with {len(ping_df)} records")
            else:
                print("No ping data to save")
            
            # Load and combine GRPC files
            print("Loading GRPC data...")
            grpc_df = self.load_grpc_data(start_time, end_time)
            if not grpc_df.empty:
                grpc_df.to_csv(os.path.join(self.output_dir, 'combined_grpc_data.csv'), index=False)
                print(f"Saved GRPC data with {len(grpc_df)} records")
            else:
                print("No GRPC data to save")
            
            # Load and combine serving satellite files
            print("Loading serving satellite data...")
            serving_df = self.load_serving_satellite_data(start_time, end_time)
            if not serving_df.empty:
                # Merge with ping and GRPC data
                serving_df = self.merge_satellite_data(serving_df, ping_df, grpc_df)
                serving_df.to_csv(os.path.join(self.output_dir, 'combined_serving_satellite.csv'), index=False)
                print(f"Saved serving satellite data with {len(serving_df)} records")
            else:
                print("No serving satellite data to save")
            
            # Create connection periods CSV only if we have serving satellite data
            if not serving_df.empty:
                periods_df = self.create_connection_periods_csv(
                    serving_df,
                    os.path.join(self.output_dir, 'connection_periods.csv')
                )
                print(f"Found {len(periods_df)} unique connection periods")
            else:
                print("No connection periods to analyze (no serving satellite data)")
            
            # Compute and save satellite visibility
            print("Computing satellite visibility...")
            visibility_results = self.compute_visibility_at_handovers()
            if visibility_results:
                print(f"Computed visibility for {len(visibility_results)} timestamps")
            else:
                print("No visibility results to save")
            
            print("Processing complete!")
                
        except Exception as e:
            print(f"Error: {e}")
            raise

def main():
    """
    Main entry point for the script.
    
    Parses command line arguments and initiates data processing.
    """
    parser = argparse.ArgumentParser(description='Process satellite connection data')
    parser.add_argument('--start', required=True, help='Start time (YYYY-MM-DD-HH-MM-SS)')
    parser.add_argument('--end', required=True, help='End time (YYYY-MM-DD-HH-MM-SS)')
    parser.add_argument('--output_dir', default='output', help='Output directory for CSV files')
    parser.add_argument('--lat', type=float, required=True, help='Latitude of the observer')
    parser.add_argument('--lon', type=float, required=True, help='Longitude of the observer')
    parser.add_argument('--alt', type=float, required=True, help='Altitude of the observer (in meters)')

    args = parser.parse_args()
    
    # Parse time window
    start_time = SatelliteDataProcessor.parse_timestamp(args.start)
    end_time = SatelliteDataProcessor.parse_timestamp(args.end)
    
    # Create processor and process data
    processor = SatelliteDataProcessor(start_time, end_time, args.lat, args.lon, args.alt, args.output_dir)
    
    # processor.compute_visibility(args.latitude, args.longitude, args.altitude)
    
    processor.process_data(start_time, end_time)

if __name__ == "__main__":
    main()