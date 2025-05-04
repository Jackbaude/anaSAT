#!/usr/bin/env python3
"""
Script to analyze satellite connections and latency data.
"""

import pandas as pd
import os
import sys
import re
from datetime import datetime, timezone, timedelta
import logging
from satellite_plotter import SatellitePlotter
from skyfield.api import load, wgs84, EarthSatellite
import argparse
import numpy as np
import glob
from typing import List, Dict, Any
import json
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class SatelliteAnalysis:
    def __init__(self, timestamp, observer_lat=None, observer_lon=None, observer_alt=None):
        """Initialize the SatelliteAnalysis class with a timestamp and observer location."""
        self.timestamp = timestamp
        self.output_dir = self._ensure_output_directory()
        self.df = None
        self.periods = None
        self.satellite_change_times = None
        self.regular_handover_times = None
        self.latency_series = None
        self.satellites = None
        self.observer_lat = observer_lat
        self.observer_lon = observer_lon
        self.observer_alt = observer_alt
        self.tle_data = None
        self.satellite_objects = None

    def _ensure_output_directory(self):
        """Create output directory if it doesn't exist."""
        output_dir = f"{self.timestamp}_figures"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        return output_dir

    def load_tle_data(self, tle_dir: str = 'data/TLE') -> Dict[str, Any]:
        """Load the most recent TLE data for each satellite from multiple sources."""
        print("Loading TLE data...")
        tle_data = {}
        ts = load.timescale()
        
        # Track the most recent TLE for each satellite
        satellite_tles = {}  # {satellite_name: (timestamp, tle_data)}
        
        # Find all TLE files in the directory and subdirectories
        for root, _, files in os.walk(tle_dir):
            for tle_file in files:
                if tle_file.endswith(('.txt', '.tle')):
                    try:
                        file_path = os.path.join(root, tle_file)
                        
                        # Try to extract timestamp from filename
                        try:
                            # Handle different filename formats
                            if 'starlink-tle-' in tle_file:
                                file_time = datetime.strptime(tle_file.replace('starlink-tle-', '').replace('.txt', ''),
                                                            '%Y-%m-%d-%H-%M-%S')
                            elif 'tle-' in tle_file:
                                file_time = datetime.strptime(tle_file.replace('tle-', '').replace('.txt', '').replace('.tle', ''),
                                                            '%Y-%m-%d-%H-%M-%S')
                            else:
                                # If no timestamp in filename, use file modification time
                                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        except ValueError:
                            # If timestamp parsing fails, use file modification time
                            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                        
                        with open(file_path, 'r') as f:
                lines = f.readlines()
            
                            # Handle different TLE formats
                            if len(lines) >= 3 and lines[0].strip().startswith('STARLINK'):
                                # Standard 3-line format
            for i in range(0, len(lines), 3):
                                    if i + 2 < len(lines):
                name = lines[i].strip()
                line1 = lines[i+1].strip()
                line2 = lines[i+2].strip()
                
                                        # Keep only the most recent TLE for each satellite
                                        if name not in satellite_tles or file_time > satellite_tles[name][0]:
                                            satellite_tles[name] = (file_time, (line1, line2))
                            else:
                                # Try to parse as 2-line format
                                for i in range(0, len(lines), 2):
                                    if i + 1 < len(lines):
                                        line1 = lines[i].strip()
                                        line2 = lines[i+1].strip()
                                        
                                        # Extract satellite name from first line
                                        name = f"STARLINK-{line1[2:7]}"
                                        
                                        # Keep only the most recent TLE for each satellite
                                        if name not in satellite_tles or file_time > satellite_tles[name][0]:
                                            satellite_tles[name] = (file_time, (line1, line2))
                    except Exception as e:
                        print(f"Error loading TLE file {tle_file}: {e}")
        
        # Create EarthSatellite objects from the most recent TLEs
        for name, (timestamp, (line1, line2)) in satellite_tles.items():
            try:
                tle_data[name] = {
                    'satellite': EarthSatellite(line1, line2, name, ts),
                    'timestamp': timestamp,
                    'source': 'TLE'
                }
            except Exception as e:
                print(f"Error creating satellite object for {name}: {e}")
        
        print(f"Loaded TLE data for {len(tle_data)} satellites")
        
        # Check TLE data freshness
        now = datetime.now()
        for name, data in tle_data.items():
            age = (now - data['timestamp']).days
            if age > 7:
                print(f"Warning: TLE data for {name} is {age} days old (accuracy may be degraded)")
            
            self.tle_data = tle_data
        return tle_data

    def parse_ping_file(self, ping_file):
        """Parse the ping file and return timestamps and latencies."""
        logger.info(f"Reading ping data from {ping_file}")
        timestamps = []
        latencies = []
        
        try:
            with open(ping_file, 'r') as f:
                for line in f:
                    if line.startswith('PING'):
                        continue
                    
                    match = re.match(r'\[(\d+\.\d+)\].*time=(\d+\.\d+) ms', line)
                    if match:
                        timestamp = float(match.group(1))
                        latency = float(match.group(2))
                        timestamps.append(timestamp)
                        latencies.append(latency)
        except FileNotFoundError:
            logger.error(f"Ping file {ping_file} not found")
            return [], []
        except Exception as e:
            logger.error(f"Error reading ping file: {e}")
            return [], []
        
        logger.info(f"Successfully parsed {len(timestamps)} ping measurements")
        return timestamps, latencies

    def load_satellite_data(self, serving_file):
        """Load and process satellite connection data."""
        logger.info(f"Loading satellite data from {serving_file}")
        try:
            self.df = pd.read_csv(serving_file)
            self.df['timestamp'] = pd.to_datetime(self.df['Timestamp'])
            self.df = self.df.sort_values('timestamp')
            return self.df
        except Exception as e:
            logger.error(f"Error loading satellite data: {e}")
            raise

    def find_connection_periods(self):
        """Identify continuous connection periods for each satellite."""
        logger.info("Analyzing satellite connection periods")
        self.periods = []
        current_sat = None
        start_time = None
        self.satellite_change_times = []
        
        for _, row in self.df.iterrows():
            if row['Connected_Satellite'] != current_sat:
                if current_sat is not None:
                    self.periods.append({
                        'satellite': current_sat,
                        'start_time': start_time,
                        'end_time': row['timestamp'],
                        'duration': (row['timestamp'] - start_time).total_seconds()
                    })
                    self.satellite_change_times.append(row['timestamp'])
                
                current_sat = row['Connected_Satellite']
                start_time = row['timestamp']
        
        if current_sat is not None:
            self.periods.append({
                'satellite': current_sat,
                'start_time': start_time,
                'end_time': self.df['timestamp'].iloc[-1],
                'duration': (self.df['timestamp'].iloc[-1] - start_time).total_seconds()
            })
        
        logger.info(f"Found {len(self.periods)} connection periods across {len(set(p['satellite'] for p in self.periods))} satellites")
        return self.periods, self.satellite_change_times

    def calculate_regular_handovers(self):
        """Calculate regular handover times at specific seconds within each minute (57, 12, 37, 42)."""
        logger.info("Calculating regular handover times")
        start_time = self.df['timestamp'].min()
        end_time = self.df['timestamp'].max()
        
        # Define the handover seconds within each minute
        handover_seconds = [57, 12, 27, 42]
        
        # Generate handover times for each minute in the data range
        self.regular_handover_times = []
        current_time = start_time.replace(second=0, microsecond=0)
        
        while current_time <= end_time:
            for second in handover_seconds:
                handover_time = current_time + pd.Timedelta(seconds=second)
                if start_time <= handover_time <= end_time:
                    self.regular_handover_times.append(handover_time)
            current_time += pd.Timedelta(minutes=1)
        
        logger.info(f"Found {len(self.regular_handover_times)} regular handover times")
        return self.regular_handover_times

    def process_data(self, serving_file, ping_file):
        """Process all data and create plots."""
        try:
            # Load and process satellite data
            self.load_satellite_data(serving_file)
            self.find_connection_periods()
            
            # Load TLE data
            self.load_tle_data()
            
            # Load and process latency data
            ping_timestamps, ping_latencies = self.parse_ping_file(ping_file)
            if ping_timestamps and ping_latencies:
                ping_times = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in ping_timestamps]
                self.latency_series = pd.Series(ping_latencies, index=ping_times)
            else:
                logger.warning("No valid latency data found")
            
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise

    def get_time_frame_data(self, start_time, end_time):
        """
        Get satellite connection data within a specific time frame.
        
        Args:
            start_time (str): Start time in format 'YYYY-MM-DD-HH-MM-SS'
            end_time (str): End time in format 'YYYY-MM-DD-HH-MM-SS'
            
        Returns:
            DataFrame: Filtered data within the specified time range
        """
        try:
            # Convert string timestamps to datetime objects with UTC timezone
            start_dt = pd.to_datetime(start_time, format='%Y-%m-%d-%H-%M-%S', utc=True)
            end_dt = pd.to_datetime(end_time, format='%Y-%m-%d-%H-%M-%S', utc=True)
            
            # Filter the data
            mask = (self.df['timestamp'] >= start_dt) & (self.df['timestamp'] <= end_dt)
            time_frame_data = self.df[mask].copy()
            
            logger.info(f"Found {len(time_frame_data)} data points between {start_time} and {end_time}")
            return time_frame_data
            
        except Exception as e:
            logger.error(f"Error filtering time frame data: {e}")
            raise

    def analyze_connections_in_time_frame(self, start_time, end_time):
        """
        Analyze satellite connections within a specific time frame.
        
        Args:
            start_time (str): Start time in format 'YYYY-MM-DD-HH-MM-SS'
            end_time (str): End time in format 'YYYY-MM-DD-HH-MM-SS'
            
        Returns:
            dict: Analysis results including:
                - total_connections: Total number of connections
                - unique_satellites: Number of unique satellites
                - avg_connection_duration: Average connection duration
                - connection_stats: Detailed statistics per satellite
        """
        try:
            time_frame_data = self.get_time_frame_data(start_time, end_time)
            
            if time_frame_data.empty:
                logger.warning("No data found in the specified time frame")
                return None
            
            # Calculate basic statistics
            total_connections = len(time_frame_data)
            unique_satellites = time_frame_data['Connected_Satellite'].nunique()
            
            # Calculate connection durations
            connection_periods = []
            current_sat = None
            start_timestamp = None
            
            for _, row in time_frame_data.iterrows():
                if row['Connected_Satellite'] != current_sat:
                    if current_sat is not None:
                        connection_periods.append({
                            'satellite': current_sat,
                            'duration': (row['timestamp'] - start_timestamp).total_seconds()
                        })
                    current_sat = row['Connected_Satellite']
                    start_timestamp = row['timestamp']
            
            if current_sat is not None:
                connection_periods.append({
                    'satellite': current_sat,
                    'duration': (time_frame_data['timestamp'].iloc[-1] - start_timestamp).total_seconds()
                })
            
            # Calculate average connection duration
            avg_connection_duration = sum(p['duration'] for p in connection_periods) / len(connection_periods)
            
            # Calculate statistics per satellite
            connection_stats = {}
            for period in connection_periods:
                sat = period['satellite']
                if sat not in connection_stats:
                    connection_stats[sat] = {
                        'total_connections': 0,
                        'total_duration': 0,
                        'avg_duration': 0
                    }
                connection_stats[sat]['total_connections'] += 1
                connection_stats[sat]['total_duration'] += period['duration']
            
            # Calculate average duration per satellite
            for sat in connection_stats:
                connection_stats[sat]['avg_duration'] = (
                    connection_stats[sat]['total_duration'] / 
                    connection_stats[sat]['total_connections']
                )
            
            results = {
                'total_connections': total_connections,
                'unique_satellites': unique_satellites,
                'avg_connection_duration': avg_connection_duration,
                'connection_stats': connection_stats
            }
            
            logger.info(f"Analysis complete for time frame {start_time} to {end_time}")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing connections in time frame: {e}")
            raise

    def get_connection_events(self, start_time, end_time):
        """
        Get all connection events (handovers) within a specific time frame.
        
        Args:
            start_time (str): Start time in format 'YYYY-MM-DD-HH-MM-SS'
            end_time (str): End time in format 'YYYY-MM-DD-HH-MM-SS'
            
        Returns:
            list: List of connection events with timestamps and satellite changes
        """
        try:
            time_frame_data = self.get_time_frame_data(start_time, end_time)
            
            if time_frame_data.empty:
                logger.warning("No data found in the specified time frame")
                return []
            
            connection_events = []
            prev_sat = None
            
            for _, row in time_frame_data.iterrows():
                if row['Connected_Satellite'] != prev_sat:
                    connection_events.append({
                        'timestamp': row['timestamp'],
                        'from_satellite': prev_sat,
                        'to_satellite': row['Connected_Satellite']
                    })
                    prev_sat = row['Connected_Satellite']
            
            logger.info(f"Found {len(connection_events)} connection events between {start_time} and {end_time}")
            return connection_events
            
        except Exception as e:
            logger.error(f"Error getting connection events: {e}")
            raise

def parse_timestamp(ts: str) -> datetime:
    """Parse timestamp string into datetime object."""
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S%z")

def parse_unix_timestamp(ts: float) -> datetime:
    """Convert UNIX timestamp to datetime."""
    return datetime.fromtimestamp(ts, tz=timezone.utc)

def load_ping_data(start_time: datetime, end_time: datetime) -> Dict[str, pd.DataFrame]:
    """Load and cache ping data files by hour."""
    ping_cache = {}
    
    # Find all relevant ping files
    date = start_time.date()
    while date <= end_time.date():
        ping_dir = os.path.join('data', 'latency', date.strftime('%Y-%m-%d'))
        if os.path.exists(ping_dir):
            print(f"Processing ping files in {ping_dir}")
            for ping_file in glob.glob(os.path.join(ping_dir, 'ping-100ms-*.txt')):
                try:
                    # Extract time from filename
                    filename = os.path.basename(ping_file)
                    file_time = datetime.strptime(filename.replace('ping-100ms-', '').replace('.txt', ''), 
                                                "%Y-%m-%d-%H-%M-%S")
                    
                    # Only process files within our time window
                    if start_time <= file_time <= end_time:
                        print(f"Loading {filename} into cache")
                        # Read the file into a DataFrame
                        with open(ping_file, 'r') as f:
                            data = []
                            for line in f:
                                match = re.match(r'\[(\d+\.\d+)\].*time=(\d+\.\d+)\s+ms', line)
                                if match:
                                    timestamp = float(match.group(1))
                                    latency = float(match.group(2))
                                    dt = parse_unix_timestamp(timestamp)
                                    data.append({
                                        'timestamp': dt,
                                        'latency_ms': latency
                                    })
                            
                            if data:
                                df = pd.DataFrame(data)
                                # Cache by hour
                                cache_key = file_time.strftime("%Y-%m-%d-%H")
                                ping_cache[cache_key] = df
                                print(f"Cached {len(df)} records for {cache_key}")
                except Exception as e:
                    print(f"Error processing ping file {ping_file}: {e}")
        date += timedelta(days=1)
    
    if not ping_cache:
        print("Warning: No ping data found in the specified time window")
    
    return ping_cache

def get_ping_data_for_period(ping_cache: Dict[str, pd.DataFrame], 
                           start_time: datetime, 
                           end_time: datetime) -> pd.DataFrame:
    """Get ping data for a specific time period from the cache."""
    # Get all relevant hours
    current_time = start_time
    relevant_data = []
    
    while current_time <= end_time:
        cache_key = current_time.strftime("%Y-%m-%d-%H")
        if cache_key in ping_cache:
            df = ping_cache[cache_key]
            # Filter for the specific time period
            mask = (df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)
            relevant_data.append(df[mask])
        current_time += timedelta(hours=1)
    
    if not relevant_data:
        return pd.DataFrame(columns=['timestamp', 'latency_ms'])
    
    return pd.concat(relevant_data, ignore_index=True).sort_values('timestamp')

def load_grpc_data(start_time: datetime, end_time: datetime) -> pd.DataFrame:
    """Load and process GRPC status data files."""
    grpc_data = []
    
    # Find all relevant GRPC files
    date = start_time.date()
    while date <= end_time.date():
        grpc_dir = os.path.join('data', 'grpc', date.strftime('%Y-%m-%d'))
        if os.path.exists(grpc_dir):
            for grpc_file in glob.glob(os.path.join(grpc_dir, 'GRPC_STATUS-*.csv')):
                try:
                    df = pd.read_csv(grpc_file, header=None)
                    df.columns = ['timestamp', 'popPingLatencyMs', 'downlinkThroughputBps', 'uplinkThroughputBps']
                    df['timestamp'] = df['timestamp'].apply(parse_unix_timestamp)
                    df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
                    grpc_data.append(df)
                            except Exception as e:
                    print(f"Error processing GRPC file {grpc_file}: {e}")
        date += timedelta(days=1)
    
    if not grpc_data:
        print("Warning: No GRPC data found in the specified time window")
        return pd.DataFrame(columns=['timestamp', 'popPingLatencyMs', 'downlinkThroughputBps', 'uplinkThroughputBps'])
    
    return pd.concat(grpc_data, ignore_index=True).sort_values('timestamp')

def find_connection_periods(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Identify continuous connection periods for each satellite."""
    periods = []
    current_sat = None
    current_start = None
                    current_data = []
                
    for _, row in df.iterrows():
        if current_sat != row['Connected_Satellite']:
            # End of current period
            if current_sat is not None:
                periods.append({
                    'satellite': current_sat,
                    'start_time': current_start,
                    'end_time': row['Timestamp'],
                    'data': current_data
                })
            
            # Start new period
            current_sat = row['Connected_Satellite']
            current_start = row['Timestamp']
            current_data = []
        
        current_data.append(row.to_dict())
    
    # Add the last period
            if current_sat is not None:
        periods.append({
            'satellite': current_sat,
            'start_time': current_start,
            'end_time': df.iloc[-1]['Timestamp'],
            'data': current_data
        })
    
    return periods

def load_serving_satellite_data(start_time: datetime, end_time: datetime) -> pd.DataFrame:
    """Load and concatenate all serving satellite data files within the time window."""
    all_data = []
    
    # Find all relevant files
    print("Finding relevant data files...")
    relevant_files = []
    for file_path in glob.glob('data/serving_satellite_data-*.csv'):
        try:
            # Extract date and time from filename
            filename = os.path.basename(file_path)
            # Remove the prefix and .csv extension
            date_str = filename.replace('serving_satellite_data-', '').replace('.csv', '')
            # Parse the date and time
            file_time = datetime.strptime(date_str, "%Y-%m-%d-%H-%M-%S")
            if start_time <= file_time <= end_time:
                relevant_files.append(file_path)
        except (ValueError, IndexError) as e:
            print(f"Error parsing filename {filename}: {e}")
            continue
    
    if not relevant_files:
        raise ValueError("No data files found in the specified time window")
    
    print(f"Found {len(relevant_files)} files to process")
    
    # Process files in chunks to manage memory
    chunk_size = 24  # Process 24 hours at a time
    for i in tqdm(range(0, len(relevant_files), chunk_size), desc="Processing files"):
        chunk_files = relevant_files[i:i + chunk_size]
        chunk_data = []
        for file_path in chunk_files:
            try:
                df = pd.read_csv(file_path)
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                chunk_data.append(df)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        if chunk_data:
            all_data.append(pd.concat(chunk_data, ignore_index=True))
    
    if not all_data:
        raise ValueError("No valid data could be loaded")
    
    return pd.concat(all_data, ignore_index=True).sort_values('Timestamp')

def process_connection_periods(df: pd.DataFrame, ping_cache: Dict[str, pd.DataFrame], grpc_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Process the data to create a list of dictionaries with one entry per connection period."""
    print("Identifying connection periods...")
    periods = find_connection_periods(df)
    
    print("Processing connection periods...")
    processed_data = []
    for period in tqdm(periods, desc="Processing periods"):
        period_data = pd.DataFrame(period['data'])
        
        # Get ping data for this period from cache
        period_ping = get_ping_data_for_period(ping_cache, period['start_time'], period['end_time'])
        
        # Get GRPC data for this period
        period_grpc = grpc_df[
            (grpc_df['timestamp'] >= period['start_time']) & 
            (grpc_df['timestamp'] <= period['end_time'])
        ] if not grpc_df.empty else pd.DataFrame()
        
        # Create a dictionary for this period
        period_dict = {
            'satellite': period['satellite'],
            'start_time': period['start_time'].isoformat(),
            'end_time': period['end_time'].isoformat(),
            'num_samples': len(period_data),
            'metrics': {
                'elevation': period_data['Elevation'].tolist(),
                'azimuth': period_data['Azimuth'].tolist(),
                'distance': period_data['Distance'].tolist(),
                'x': period_data['X'].tolist(),
                'y': period_data['Y'].tolist()
            },
            'ping_data': {
                'timestamps': period_ping['timestamp'].dt.isoformat().tolist() if not period_ping.empty else [],
                'latency_ms': period_ping['latency_ms'].tolist() if not period_ping.empty else []
            },
            'grpc_data': {
                'timestamps': period_grpc['timestamp'].dt.isoformat().tolist() if not period_grpc.empty else [],
                'pop_ping_latency_ms': period_grpc['popPingLatencyMs'].tolist() if not period_grpc.empty else [],
                'downlink_throughput_bps': period_grpc['downlinkThroughputBps'].tolist() if not period_grpc.empty else [],
                'uplink_throughput_bps': period_grpc['uplinkThroughputBps'].tolist() if not period_grpc.empty else []
            }
        }
        
        processed_data.append(period_dict)
    
    return processed_data

def combine_serving_satellite_files(start_time: datetime, end_time: datetime, output_file: str) -> pd.DataFrame:
    """Combine all serving satellite CSV files within the time window."""
    print("Combining serving satellite files...")
    all_data = []
    
    # Find all relevant files
    current_time = start_time
    while current_time <= end_time:
        file_path = os.path.join('data', f"serving_satellite_data-{current_time.strftime('%Y-%m-%d-%H-%M-%S')}.csv")
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                all_data.append(df)
                print(f"Loaded {file_path}")
                    except Exception as e:
                print(f"Error loading {file_path}: {e}")
        current_time += timedelta(hours=1)
    
    if not all_data:
        raise ValueError("No serving satellite data files found in the specified time window")
    
    combined_df = pd.concat(all_data, ignore_index=True).sort_values('Timestamp')
    combined_df.to_csv(output_file, index=False)
    print(f"Saved combined serving satellite data to {output_file}")
    return combined_df

def combine_ping_files(start_time: datetime, end_time: datetime, output_file: str) -> pd.DataFrame:
    """Combine all ping files within the time window and convert timestamps."""
    print("Combining ping files...")
    all_data = []
    
    # Ensure start_time and end_time are timezone-aware
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)
    
    # Find all relevant files
    current_time = start_time
    while current_time <= end_time:
        date_str = current_time.strftime('%Y-%m-%d')
        hour_str = current_time.strftime('%H')
        ping_file = os.path.join('data', 'latency', date_str, f"ping-100ms-{date_str}-{hour_str}-00-00.txt")
        
                if os.path.exists(ping_file):
                    try:
                with open(ping_file, 'r') as f:
                    data = []
                    for line in f:
                        match = re.match(r'\[(\d+\.\d+)\].*time=(\d+\.\d+)\s+ms', line)
                        if match:
                            timestamp = float(match.group(1))
                            latency = float(match.group(2))
                            dt = parse_unix_timestamp(timestamp)
                            if start_time <= dt <= end_time:
                                data.append({
                                    'Timestamp': dt,
                                    'Latency_ms': latency
                                })
                    if data:
                        df = pd.DataFrame(data)
                        all_data.append(df)
                        print(f"Loaded {ping_file}")
                    except Exception as e:
                print(f"Error loading {ping_file}: {e}")
        current_time += timedelta(hours=1)
    
    if not all_data:
        raise ValueError("No ping data files found in the specified time window")
    
    combined_df = pd.concat(all_data, ignore_index=True).sort_values('Timestamp')
    combined_df.to_csv(output_file, index=False)
    print(f"Saved combined ping data to {output_file}")
    return combined_df

def combine_grpc_files(start_time: datetime, end_time: datetime, output_file: str) -> pd.DataFrame:
    """Combine all GRPC status CSV files within the time window."""
    print("Combining GRPC status files...")
    all_data = []
    
    # Ensure start_time and end_time are timezone-aware
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)
    
    # Find all relevant files
    current_time = start_time
    while current_time <= end_time:
        date_str = current_time.strftime('%Y-%m-%d')
        hour_str = current_time.strftime('%H')
        grpc_file = os.path.join('data', 'grpc', date_str, f"GRPC_STATUS-{date_str}-{hour_str}-00-00.csv")
        
        if os.path.exists(grpc_file):
            try:
                # Read the CSV with headers
                df = pd.read_csv(grpc_file)
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
                df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
                
                all_data.append(df)
                print(f"Loaded {grpc_file} with {len(df.columns)} columns")
            except Exception as e:
                print(f"Error loading {grpc_file}: {e}")
                print(f"Error details: {str(e)}")
                print(f"First line of file: {open(grpc_file).readline().strip()}")
        current_time += timedelta(hours=1)
    
    if not all_data:
        raise ValueError("No GRPC data files found in the specified time window")
    
    combined_df = pd.concat(all_data, ignore_index=True).sort_values('timestamp')
    combined_df.to_csv(output_file, index=False)
    print(f"Saved combined GRPC data to {output_file}")
    return combined_df

def create_connection_periods_csv(serving_df: pd.DataFrame, output_file: str):
    """Create a CSV file with unique satellite connection periods using existing distance data."""
    print("Creating connection periods CSV...")
    periods = []
    current_sat = None
    start_time = None
    current_data = []
    
    for _, row in serving_df.iterrows():
        if row['Connected_Satellite'] != current_sat:
            if current_sat is not None:
                # Calculate distance metrics for the period
                if current_data:
                    distances = [d['Distance'] for d in current_data if pd.notna(d['Distance'])]
                    if distances:
                        periods.append({
                            'Satellite': current_sat,
                            'Start_Time': start_time,
                            'End_Time': row['Timestamp'],
                            'Duration_Seconds': (row['Timestamp'] - start_time).total_seconds(),
                            'Max_Distance': max(distances),
                            'Min_Distance': min(distances),
                            'First_Distance': distances[0],
                            'Last_Distance': distances[-1],
                            'Mean_Distance': np.mean(distances),
                            'Std_Distance': np.std(distances)
                        })
                    else:
                        periods.append({
                            'Satellite': current_sat,
                            'Start_Time': start_time,
                            'End_Time': row['Timestamp'],
                            'Duration_Seconds': (row['Timestamp'] - start_time).total_seconds(),
                            'Max_Distance': None,
                            'Min_Distance': None,
                            'First_Distance': None,
                            'Last_Distance': None,
                            'Mean_Distance': None,
                            'Std_Distance': None
                        })
            current_sat = row['Connected_Satellite']
            start_time = row['Timestamp']
            current_data = []
        
        current_data.append(row.to_dict())
    
    # Add the last period
    if current_sat is not None and current_data:
        distances = [d['Distance'] for d in current_data if pd.notna(d['Distance'])]
        if distances:
            periods.append({
                'Satellite': current_sat,
                'Start_Time': start_time,
                'End_Time': serving_df['Timestamp'].iloc[-1],
                'Duration_Seconds': (serving_df['Timestamp'].iloc[-1] - start_time).total_seconds(),
                'Max_Distance': max(distances),
                'Min_Distance': min(distances),
                'First_Distance': distances[0],
                'Last_Distance': distances[-1],
                'Mean_Distance': np.mean(distances),
                'Std_Distance': np.std(distances)
            })
        else:
            periods.append({
                'Satellite': current_sat,
                'Start_Time': start_time,
                'End_Time': serving_df['Timestamp'].iloc[-1],
                'Duration_Seconds': (serving_df['Timestamp'].iloc[-1] - start_time).total_seconds(),
                'Max_Distance': None,
                'Min_Distance': None,
                'First_Distance': None,
                'Last_Distance': None,
                'Mean_Distance': None,
                'Std_Distance': None
            })
    
    periods_df = pd.DataFrame(periods)
    periods_df.to_csv(output_file, index=False)
    print(f"Saved connection periods to {output_file}")
    return periods_df

def main():
    parser = argparse.ArgumentParser(description='Process satellite connection data')
    parser.add_argument('--start', required=True, help='Start time (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--end', required=True, help='End time (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--output_dir', default='output', help='Output directory for CSV files')
    
    args = parser.parse_args()
    
    # Parse time window and ensure timezone-aware
    start_time = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    end_time = datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Processing data from {start_time} to {end_time}")
    
    try:
        # Combine serving satellite files
        serving_df = combine_serving_satellite_files(
            start_time, 
            end_time, 
            os.path.join(args.output_dir, 'combined_serving_satellite.csv')
        )
        
        # Combine ping files
        ping_df = combine_ping_files(
            start_time, 
            end_time, 
            os.path.join(args.output_dir, 'combined_ping_data.csv')
        )
        
        # Combine GRPC files
        grpc_df = combine_grpc_files(
            start_time, 
            end_time, 
            os.path.join(args.output_dir, 'combined_grpc_data.csv')
        )
        
        # Create connection periods CSV
        periods_df = create_connection_periods_csv(
            serving_df,
            os.path.join(args.output_dir, 'connection_periods.csv')
        )
        
        print("Processing complete!")
        print(f"Found {len(periods_df)} unique connection periods")
            
        except Exception as e:
        print(f"Error: {e}")
            raise

if __name__ == "__main__":
    main()