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
import json
import threading
import queue
import multiprocessing as mp
from multiprocessing import Process, Queue, Manager
import fcntl
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def process_timestamp(ts, serving_df, processor, output_file):
    """Process a single timestamp and write results directly to file."""
    ts_str = ts.isoformat()
    
    # Skip if we already have this timestamp
    if ts_str in processor.shared_results:
        return 1
    
    # Get the closest GRPC data point for this timestamp
    mask = serving_df['Timestamp'] <= ts
    if not mask.any():
        logger.warning(f"No GRPC data available before timestamp {ts}")
        return 1
        
    grpc_data = serving_df[mask].iloc[-1]
    
    # Get dish parameters from GRPC data
    tilt_deg = grpc_data.get('tiltAngleDeg', 0)
    rotation_deg = grpc_data.get('boresightAzimuthDeg', 0)
    fov_azimuth = grpc_data.get('desiredBoresightAzimuthDeg', 0)
    fov_elevation = grpc_data.get('desiredBoresightElevationDeg', 0)
    
    # Get TLE data for this timestamp
    tle_data = processor.load_tle_data(ts)
    if tle_data.empty:
        logger.warning(f"No TLE data available for hour containing timestamp {ts}")
        return 1
    
    visible_sats_data = []
    
    for _, sat_data in tle_data.iterrows():
        # Calculate satellite position
        alt, az = processor.get_satellite_position(sat_data, ts)
        
        # Check if satellite is within valid FOV
        if processor.is_valid_satellite(alt, az, tilt_deg, rotation_deg):
            sat_name = sat_data['satellite_name']
            visible_sats_data.append({
                'satellite': sat_name,
                'sat_elevation_deg': alt,
                'sat_azimuth_deg': az,
                'UT_boresight_elevation': tilt_deg,
                'UT_boresight_azimuth': rotation_deg,
                'desired_boresight_azimuth': fov_azimuth,
                'desired_boresight_elevation': fov_elevation,
                'tle_line1': sat_data['tle_line1'],
                'tle_line2': sat_data['tle_line2']
            })
    
    if visible_sats_data:
        # Write results directly to file with file-based locking
        lock_file = f"{output_file}.lock"
        with open(lock_file, 'w') as lock_f:
            try:
                # Acquire an exclusive lock
                fcntl.flock(lock_f, fcntl.LOCK_EX)
                
                # Load current results
                try:
                    with open(output_file, 'r') as f:
                        current_results = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    current_results = {}
                
                # Update results
                current_results[ts_str] = visible_sats_data
                
                # Write back to file
                with open(output_file, 'w') as f:
                    json.dump(current_results, f)
                
                # Release the lock
                fcntl.flock(lock_f, fcntl.LOCK_UN)
            except Exception as e:
                logger.error(f"Error writing results: {e}")
                # Make sure to release the lock even if there's an error
                fcntl.flock(lock_f, fcntl.LOCK_UN)
                raise
        
        logger.info(f"{ts}: {len(visible_sats_data)} satellites visible")
    
    return 1

def file_writer(output_file, results_queue, total_count):
    """Dedicated process for handling file I/O."""
    local_results = {}
    processed = 0
    
    while processed < total_count:
        try:
            # Get results with timeout
            while True:
                try:
                    ts_str, data = results_queue.get(timeout=1)
                    local_results[ts_str] = data
                    
                    # Write to file after each successful update
                    with open(output_file, 'w') as f:
                        json.dump(local_results, f, indent=2)
                except queue.Empty:
                    break
        except Exception as e:
            logger.error(f"Error in file writer: {e}")
            continue

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

    def __init__(self, start_time: datetime, end_time: datetime, latitude: float, longitude: float, altitude: float, duration_minutes: int = 60, output_dir: str = 'output'):
        """
        Initialize the SatelliteDataProcessor.

        Args:
            start_time (datetime): Start time of analysis
            end_time (datetime): End time of analysis
            latitude (float): Observer's latitude
            longitude (float): Observer's longitude
            altitude (float): Observer's altitude in meters
            duration_minutes (int): Duration of each data file in minutes
            output_dir (str): Directory where processed data will be saved
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize commonly used values
        self.ts = load.timescale()
        self.file_regex = re.compile(r'serving_satellite_data-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.csv')
        self.max_workers = 8
        self.merge_tolerance = pd.Timedelta(seconds=1)
        self.tle_cache = {}

        self.start_time = start_time
        self.end_time = end_time
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.duration_minutes = duration_minutes
        self.shared_results = None

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
    
    def generate_timestamps(self):
        """
        Generate timestamps for handover times (57, 12, 27, and 42 seconds) between start and end time.
        """
        current_time = self.start_time
        timestamps = []
        handover_seconds = [57, 12, 27, 42]  # Handover times

        while current_time <= self.end_time:
            # Only add timestamp if it's a handover second
            if current_time.second in handover_seconds:
                timestamps.append(current_time)
            current_time += timedelta(seconds=1)

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
            # Read file content
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Parse ping data
            ping_data = []
            for line in lines:
                # Skip empty lines
                if not line.strip():
                    continue
                
                # Try to parse timestamp and latency
                try:
                    # Extract timestamp and latency using regex
                    timestamp_match = re.search(r'\[(\d+\.\d+)\]', line)
                    latency_match = re.search(r'time=(\d+\.\d+)\s*ms', line)
                    
                    if timestamp_match and latency_match:
                        timestamp = float(timestamp_match.group(1))
                        latency = float(latency_match.group(1))
                        
                        # Convert timestamp to datetime
                        ping_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                        
                        # Only include if within time range
                        if start_time <= ping_time <= end_time:
                            ping_data.append({
                                'Timestamp': ping_time,
                                'Latency_ms': latency
                            })
                except Exception as e:
                    logger.warning(f"Error parsing ping line: {line.strip()} - {e}")
                    continue
            
            if not ping_data:
                logger.warning(f"No valid ping data found in {file_path}")
                return pd.DataFrame(columns=['Timestamp', 'Latency_ms'])
            
            # Convert to DataFrame
            df = pd.DataFrame(ping_data)
            logger.info(f"Loaded {len(df)} ping records from {file_path}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading ping file {file_path}: {e}")
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

    def get_data_files_for_timestamp(self, timestamp: datetime) -> Dict[str, str]:
        """
        Get all data files for a specific timestamp.

        Args:
            timestamp (datetime): The timestamp to get files for

        Returns:
            Dict[str, str]: Dictionary mapping file types to file paths
        """
        date_str = timestamp.strftime('%Y-%m-%d')
        time_str = timestamp.strftime('%H-%M-%S')
        
        files = {
            'grpc': os.path.join('data', 'grpc', date_str, f'GRPC_STATUS-{date_str}-{time_str}.csv'),
            'ping': os.path.join('data', 'latency', date_str, f'ping-10ms-{date_str}-{time_str}.txt'),
            'serving': os.path.join('data', f'serving_satellite_data-{date_str}-{time_str}.csv'),
            'tle': os.path.join('data', 'TLE', date_str, f'starlink-tle-{date_str}-{time_str}.txt')
        }
        
        return files

    def process_single_timestamp(self, timestamp: datetime) -> Optional[pd.DataFrame]:
        """
        Process all data files for a single timestamp.

        Args:
            timestamp (datetime): The timestamp to process

        Returns:
            Optional[pd.DataFrame]: Combined data for the timestamp, or None if no data
        """
        files = self.get_data_files_for_timestamp(timestamp)
        
        # Load and process each data type
        grpc_df = pd.DataFrame()
        ping_df = pd.DataFrame()
        serving_df = pd.DataFrame()
        
        # Load GRPC data
        if os.path.exists(files['grpc']):
            try:
                grpc_df = pd.read_csv(files['grpc'])
                grpc_df['timestamp'] = pd.to_datetime(grpc_df['timestamp'], unit='s', utc=True)
                logger.info(f"Loaded GRPC data from {files['grpc']}")
            except Exception as e:
                logger.error(f"Error loading GRPC file {files['grpc']}: {e}")
        
        # Load ping data
        if os.path.exists(files['ping']):
            try:
                ping_df = self.load_single_ping_file(files['ping'], timestamp, timestamp + timedelta(minutes=self.duration_minutes))
                if not ping_df.empty:
                    logger.info(f"Loaded {len(ping_df)} ping records")
                    # Print first few ping records for debugging
                    logger.debug(f"First few ping records:\n{ping_df.head()}")
            except Exception as e:
                logger.error(f"Error loading ping file {files['ping']}: {e}")
        
        # Load serving satellite data
        if os.path.exists(files['serving']):
            try:
                serving_df = pd.read_csv(files['serving'], parse_dates=['Timestamp'])
                serving_df['Timestamp'] = pd.to_datetime(serving_df['Timestamp'], utc=True)
                logger.info(f"Loaded serving satellite data from {files['serving']}")
                
                # Add TLE data if available
                if os.path.exists(files['tle']):
                    tle_data = self.load_tle_data(timestamp)
                    if not tle_data.empty:
                        tle_data = tle_data.rename(columns={
                            'satellite_name': 'Connected_Satellite',
                            'tle_line1': 'TLE_Line1',
                            'tle_line2': 'TLE_Line2',
                            'timestamp': 'TLE_Timestamp'
                        })
                        serving_df = serving_df.merge(tle_data, on='Connected_Satellite', how='left')
                        logger.info("Added TLE data to serving satellite data")
                        
                        # Calculate altitudes
                        def compute_altitude(row) -> Optional[float]:
                            try:
                                if pd.notna(row.TLE_Line1) and pd.notna(row.TLE_Line2):
                                    sat = EarthSatellite(row.TLE_Line1, row.TLE_Line2, row.Connected_Satellite, self.ts)
                                    t = self.ts.from_datetime(row.Timestamp.to_pydatetime())
                                    return wgs84.subpoint(sat.at(t)).elevation.km
                            except Exception as e:
                                logger.error(f"Altitude error for {row.Connected_Satellite}: {e}")
                            return None

                        # Compute altitudes in parallel
                        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                            serving_df['Altitude_km'] = list(executor.map(compute_altitude, serving_df.itertuples(index=False)))
                        logger.info("Calculated satellite altitudes")
            except Exception as e:
                logger.error(f"Error loading serving satellite file {files['serving']}: {e}")
        
        # Merge all data
        if not serving_df.empty:
            # First merge with ping data
            if not ping_df.empty:
                logger.info("Merging ping data...")
                # Sort both dataframes by timestamp
                serving_df = serving_df.sort_values('Timestamp')
                ping_df = ping_df.sort_values('Timestamp')
                
                # Merge using merge_asof to match closest timestamps
                serving_df = pd.merge_asof(
                    serving_df,
                    ping_df,
                    on='Timestamp',
                    direction='nearest',
                    tolerance=self.merge_tolerance
                )
                logger.info(f"After ping merge: {len(serving_df)} records")
            
            # Then merge with GRPC data
            if not grpc_df.empty:
                logger.info("Merging GRPC data...")
                grpc_df = grpc_df.rename(columns={'timestamp': 'Timestamp'})
                grpc_columns = [col for col in grpc_df.columns if col != 'Timestamp']
                serving_df = pd.merge_asof(
                    serving_df,
                    grpc_df[['Timestamp'] + grpc_columns].sort_values('Timestamp'),
                    on='Timestamp',
                    direction='nearest',
                    tolerance=self.merge_tolerance
                )
                logger.info(f"After GRPC merge: {len(serving_df)} records")
            
            logger.info(f"Final combined data has {len(serving_df)} records")
            return serving_df
        
        return None

    def process_data(self, start_time: datetime, end_time: datetime):
        """
        Process all data within the specified time window.

        Args:
            start_time (datetime): Start of the time window
            end_time (datetime): End of the time window
        """
        print(f"Processing data from {start_time} to {end_time}")
        
        # Create output files
        combined_file = os.path.join(self.output_dir, 'combined_serving_satellite.csv')
        periods_file = os.path.join(self.output_dir, 'connection_periods.csv')
        
        # Initialize or load existing data
        if os.path.exists(combined_file):
            print(f"Found existing combined data file: {combined_file}")
            combined_df = pd.read_csv(combined_file)
            combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'])
            print(f"Loaded {len(combined_df)} existing records")
        else:
            print("No existing combined data file found, starting fresh")
            combined_df = pd.DataFrame()
        
        # Calculate total number of timestamps to process
        total_minutes = int((end_time - start_time).total_seconds() / 60)
        total_timestamps = total_minutes // self.duration_minutes + 1
        
        # Process each timestamp with overall progress bar
        current_time = start_time
        with tqdm(total=total_timestamps, desc="Processing timestamps") as pbar:
            while current_time <= end_time:
                # Process single timestamp
                timestamp_df = self.process_single_timestamp(current_time)
                
                if timestamp_df is not None and not timestamp_df.empty:
                    # Append to combined data
                    combined_df = pd.concat([combined_df, timestamp_df], ignore_index=True)
                    
                    # Save intermediate results - append mode
                    if os.path.exists(combined_file):
                        timestamp_df.to_csv(combined_file, mode='a', header=False, index=False)
                    else:
                        timestamp_df.to_csv(combined_file, index=False)
                    
                    # Create connection periods for this chunk
                    periods_df = self.create_connection_periods_csv(
                        timestamp_df,
                        periods_file,
                        append=True
                    )
                    
                    # Update progress bar with additional info
                    pbar.set_postfix({
                        'records': len(timestamp_df),
                        'periods': len(periods_df),
                        'time': current_time.strftime('%H:%M:%S')
                    })
                
                # Move to next timestamp
                current_time += timedelta(minutes=self.duration_minutes)
                pbar.update(1)
        
        # Compute final visibility
        print("\nComputing satellite visibility...")
        visibility_results = self.compute_visibility()
        if visibility_results:
            print(f"Computed visibility for {len(visibility_results)} timestamps")
        
        print("Processing complete!")

    def create_connection_periods_csv(self, serving_df: pd.DataFrame, output_file: str, append: bool = False):
        """
        Create a CSV file with unique satellite connection periods.

        Args:
            serving_df (pd.DataFrame): DataFrame containing serving satellite data
            output_file (str): Path where the connection periods CSV will be saved
            append (bool): Whether to append to existing file or create new
        """
        periods = []
        current_sat = None
        start_time = None
        current_data = []

        # Define handover times
        handover_seconds = [57, 12, 27, 42]

        def get_nearest_handover_time(timestamp: datetime) -> datetime:
            """
            Get the nearest handover time for a given timestamp.
            If the timestamp is between handover times, round to the next handover time.
            """
            current_second = timestamp.second
            current_minute = timestamp.minute
            
            # Find the next handover second
            next_handover = None
            for handover in sorted(handover_seconds):
                if handover > current_second:
                    next_handover = handover
                    break
            
            # If no next handover in this minute, use first handover of next minute
            if next_handover is None:
                next_handover = handover_seconds[0]
                current_minute += 1
            
            # Create new timestamp with handover second
            new_timestamp = timestamp.replace(
                minute=current_minute,
                second=next_handover,
                microsecond=0
            )
            return new_timestamp

        def get_previous_handover_time(timestamp: datetime) -> datetime:
            """
            Get the previous handover time for a given timestamp.
            If the timestamp is between handover times, round to the previous handover time.
            """
            current_second = timestamp.second
            current_minute = timestamp.minute
            current_hour = timestamp.hour
            
            # Find the previous handover second
            prev_handover = None
            for handover in sorted(handover_seconds, reverse=True):
                if handover < current_second:
                    prev_handover = handover
                    break
            
            # If no previous handover in this minute, use last handover of previous minute
            if prev_handover is None:
                prev_handover = handover_seconds[-1]
                if current_minute == 0:
                    current_minute = 59
                    current_hour = (current_hour - 1) % 24  # Properly handle hour rollover
                else:
                    current_minute -= 1
            
            # Create new timestamp with handover second
            new_timestamp = timestamp.replace(
                hour=current_hour,
                minute=current_minute,
                second=prev_handover,
                microsecond=0
            )
            return new_timestamp

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
            # Adjust start and end times to handover times
            adjusted_start = get_previous_handover_time(start)
            adjusted_end = get_nearest_handover_time(end)
            
            # Safely get values with defaults
            altitudes = [d.get('Altitude_km') for d in data if pd.notna(d.get('Altitude_km'))]
            latencies = [d.get('Latency_ms') for d in data if pd.notna(d.get('Latency_ms'))]
            pop_latencies = [d.get('popPingLatencyMs') for d in data if pd.notna(d.get('popPingLatencyMs'))]

            first_entry = data[0]
            return {
                'Satellite': satellite,
                'Start_Time': adjusted_start,
                'End_Time': adjusted_end,
                'Duration_Seconds': (adjusted_end - adjusted_start).total_seconds(),
                'Mean_Altitude_km': np.mean(altitudes) if altitudes else None,
                'Mean_Latency_ms': np.mean(latencies) if latencies else None,
                'Mean_popPingLatencyMs': np.mean(pop_latencies) if pop_latencies else None,
                'TLE_Line1': first_entry.get('TLE_Line1'),
                'TLE_Line2': first_entry.get('TLE_Line2'),
                'TLE_Timestamp': first_entry.get('TLE_Timestamp')
            }

        # Process rows to create periods
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

        # Handle final period
        if current_sat is not None and current_data:
            end_time = current_data[-1]['Timestamp']
            periods.append(append_period(current_sat, start_time, end_time, current_data))

        periods_df = pd.DataFrame(periods)
        
        # Save periods
        if append and os.path.exists(output_file):
            periods_df.to_csv(output_file, mode='a', header=False, index=False)
        else:
            periods_df.to_csv(output_file, index=False)
        
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
        
    def is_valid_satellite(self, alt, az, tilt_deg, rotation_deg):
        """
        Check if a satellite is within the valid field of view.

        Args:
            alt (float): Satellite altitude in degrees
            az (float): Satellite azimuth in degrees
            tilt_deg (float): Dish tilt angle in degrees
            rotation_deg (float): Dish rotation angle in degrees

        Returns:
            bool: True if satellite is within valid FOV, False otherwise
        """
        if alt is None or az is None:
            return False

        # Convert satellite position to Cartesian coordinates
        # r is distance from center (90 - alt to match the polar plot)
        r = 90 - alt
        theta = math.radians(az)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        
        # Rotate point back to FOV frame
        angle_rad = math.radians(-rotation_deg)
        x_rot = x * math.cos(angle_rad) - y * math.sin(angle_rad)
        y_rot = x * math.sin(angle_rad) + y * math.cos(angle_rad)

        # Translate to ellipse center
        dx = x_rot - tilt_deg
        dy = y_rot

        # Check if inside ellipse equation
        base_radius = 50  # Base radius of the FOV
        x_radius = base_radius
        y_radius = math.sqrt(base_radius**2 - tilt_deg**2)
        
        # Ellipse equation: (x/a)^2 + (y/b)^2 <= 1
        inside = (dx**2 / x_radius**2) + (dy**2 / y_radius**2) <= 1
        return inside
 
    def compute_visibility(self, frame_type=2):
        """
        Compute visible satellites at handover times using dish parameters from GRPC data.
        Uses multiple processes to handle computations in parallel.

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

        timestamps = self.generate_timestamps()
        output_file = os.path.join(self.output_dir, 'satellite_visibility.json')
        
        # Initialize or load existing visibility results
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    visibility_results = json.load(f)
            except json.JSONDecodeError:
                visibility_results = {}
        else:
            visibility_results = {}

        # Create a manager for shared dictionary
        manager = Manager()
        self.shared_results = manager.dict(visibility_results)
        
        # Create process pool for computations
        num_processes = min(4, mp.cpu_count())
        with mp.Pool(num_processes) as pool:
            # Create partial function with fixed arguments
            process_func = partial(
                process_timestamp,
                serving_df=serving_df,
                processor=self,
                output_file=output_file
            )
            
            # Process timestamps with progress bar
            processed = 0
            with tqdm(total=len(timestamps), desc="Computing visibility at handover times") as pbar:
                # Use map instead of imap_unordered to ensure all timestamps are processed
                for _ in pool.map(process_func, timestamps):
                    processed += 1
                    pbar.update(1)
        
        # Load final results
        with open(output_file, 'r') as f:
            visibility_results = json.load(f)

        return visibility_results

def main():
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(description='Process satellite connection data')
    parser.add_argument('--start', required=True, help='Start time (YYYY-MM-DD-HH-MM-SS)')
    parser.add_argument('--end', required=True, help='End time (YYYY-MM-DD-HH-MM-SS)')
    parser.add_argument('--output_dir', default='output', help='Output directory for CSV files')
    parser.add_argument('--lat', type=float, required=True, help='Latitude of the observer')
    parser.add_argument('--lon', type=float, required=True, help='Longitude of the observer')
    parser.add_argument('--alt', type=float, required=True, help='Altitude of the observer (in meters)')
    parser.add_argument('--duration', type=int, default=60, help='Duration of each data file in minutes')

    args = parser.parse_args()
    
    # Parse time window
    start_time = SatelliteDataProcessor.parse_timestamp(args.start)
    end_time = SatelliteDataProcessor.parse_timestamp(args.end)
    
    # Create processor and process data
    processor = SatelliteDataProcessor(
        start_time=start_time,
        end_time=end_time,
        latitude=args.lat,
        longitude=args.lon,
        altitude=args.alt,
        duration_minutes=args.duration,
        output_dir=args.output_dir
    )
    
    processor.process_data(start_time, end_time)

if __name__ == "__main__":
    main()