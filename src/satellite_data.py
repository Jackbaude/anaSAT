import os
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
import re
from typing import List, Dict, Any, Optional, Tuple
from skyfield.api import EarthSatellite, load, wgs84
from concurrent.futures import ThreadPoolExecutor
from logger import logger

class SatelliteData:
    """Class to handle satellite data loading and processing."""
    
    def __init__(self, latitude: float, longitude: float, altitude: float):
        logger.info(f"Initializing SatelliteData with location: lat={latitude}, lon={longitude}, alt={altitude}m")
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.ts = load.timescale()
        self.file_regex = re.compile(r'serving_satellite_data-(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})\.csv')
        self.max_workers = 8
        self.merge_tolerance = pd.Timedelta(seconds=1)
        self.tle_cache = {}
        logger.debug(f"Using {self.max_workers} workers for parallel processing")

    def load_tle_data(self, timestamp: datetime) -> pd.DataFrame:
        """Load TLE data for a specific hour."""
        hour_timestamp = timestamp.replace(minute=0, second=0, microsecond=0)
        date_str = hour_timestamp.strftime('%Y-%m-%d')
        tle_file = os.path.join('data', 'TLE', date_str, 
                               f"starlink-tle-{date_str}-{hour_timestamp.strftime('%H-%M-%S')}.txt")
        logger.debug(f"Loading TLE data from {tle_file}")
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
                logger.debug(f"Loaded {len(tle_records)} TLE records from {tle_file}")
            except Exception as e:
                logger.error(f"Error loading TLE file {tle_file}: {e}")
        else:
            logger.warning(f"TLE file not found: {tle_file}")
        
        return pd.DataFrame(tle_records) if tle_records else pd.DataFrame(
            columns=['satellite_name', 'tle_line1', 'tle_line2', 'timestamp'])

    def get_matching_serving_satellite_files(self, start_time: datetime, end_time: datetime) -> List[Tuple[datetime, str]]:
        """Get all serving satellite files that match the time window."""
        logger.info(f"Searching for satellite data files between {start_time} and {end_time}")
        
        # Ensure start_time and end_time are timezone-aware
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)
        
        matched_files = []
        
        # Walk through the data directory
        for root, _, files in os.walk('data'):
            for file in files:
                if file.startswith('serving_satellite_data-'):
                    try:
                        # Extract timestamp from filename
                        timestamp_str = file.replace('serving_satellite_data-', '').replace('.csv', '')
                        file_time = datetime.strptime(timestamp_str, '%Y-%m-%d-%H-%M-%S').replace(tzinfo=timezone.utc)
                        
                        if start_time <= file_time <= end_time:
                            file_path = os.path.join(root, file)
                            matched_files.append((file_time, file_path))
                    except ValueError as e:
                        logger.warning(f"Could not parse timestamp from filename {file}: {e}")
                        continue
        
        # Sort by timestamp
        matched_files.sort(key=lambda x: x[0])
        logger.info(f"Found {len(matched_files)} matching satellite data files")
        return matched_files

    def process_satellite_file(self, file_time: datetime, file_path: Path) -> pd.DataFrame:
        """Process a single satellite data file."""
        logger.info(f"Processing satellite file: {file_path}")
        tle_data = self.load_tle_data(file_time)

        try:
            df = pd.read_csv(file_path, parse_dates=['Timestamp'])
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)
            logger.debug(f"Loaded {len(df)} records from {file_path}")

            if not tle_data.empty:
                logger.debug(f"Merging {len(tle_data)} TLE records with satellite data")
                tle_data = tle_data.rename(columns={
                    'satellite_name': 'Connected_Satellite',
                    'tle_line1': 'TLE_Line1',
                    'tle_line2': 'TLE_Line2',
                    'timestamp': 'TLE_Timestamp'
                })
                df = df.merge(tle_data, on='Connected_Satellite', how='left')
                logger.debug(f"After TLE merge: {len(df)} records")

            def compute_altitude(row) -> Optional[float]:
                try:
                    if pd.notna(row.TLE_Line1) and pd.notna(row.TLE_Line2):
                        sat = EarthSatellite(row.TLE_Line1, row.TLE_Line2, 
                                          row.Connected_Satellite, self.ts)
                        t = self.ts.from_datetime(row.Timestamp.to_pydatetime())
                        return wgs84.subpoint(sat.at(t)).elevation.km
                except Exception as e:
                    logger.error(f"Altitude error for {row.Connected_Satellite}: {e}")
                return None

            logger.info(f"Computing altitudes for {len(df)} records using {self.max_workers} workers")
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                df['Altitude_km'] = list(executor.map(compute_altitude, df.itertuples(index=False)))
            
            valid_altitudes = df['Altitude_km'].notna().sum()
            logger.info(f"Computed {valid_altitudes} valid altitudes out of {len(df)} records")
            return df

        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return pd.DataFrame()

    def merge_satellite_data(self, satellite_df: pd.DataFrame, grpc_df: pd.DataFrame) -> pd.DataFrame:
        """Merge satellite data with gRPC data."""
        logger.info(f"Merging satellite data ({len(satellite_df)} records) with gRPC data ({len(grpc_df)} records)")
        if satellite_df.empty:
            logger.warning("Empty satellite dataframe, returning empty result")
            return pd.DataFrame()

        if not grpc_df.empty:
            grpc_df = grpc_df.rename(columns={'timestamp': 'Timestamp'})
            grpc_columns = [col for col in grpc_df.columns if col != 'Timestamp']
            logger.debug(f"Merging with gRPC columns: {grpc_columns}")
            satellite_df = pd.merge_asof(
                satellite_df.sort_values('Timestamp'),
                grpc_df[['Timestamp'] + grpc_columns].sort_values('Timestamp'),
                on='Timestamp',
                direction='nearest',
                tolerance=self.merge_tolerance
            )
            logger.info(f"After merge: {len(satellite_df)} records")

        return satellite_df 