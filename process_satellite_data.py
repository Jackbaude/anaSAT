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

class SatelliteDataProcessor:
    def __init__(self, output_dir: str = 'output'):
        """Initialize the SatelliteDataProcessor."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    @staticmethod
    def parse_timestamp(ts: str) -> datetime:
        """Parse timestamp string into datetime object."""
        return datetime.strptime(ts, "%Y-%m-%d-%H-%M-%S").replace(tzinfo=timezone.utc)

    def load_tle_data(self, timestamp: datetime) -> Dict[str, Dict[str, str]]:
        """Load TLE data for a specific timestamp."""
        date_str = timestamp.strftime('%Y-%m-%d')
        tle_file = os.path.join('data', 'TLE', date_str, f"starlink-tle-{date_str}-{timestamp.strftime('%H-%M-%S')}.txt")
        
        tle_data = {}
        if os.path.exists(tle_file):
            try:
                with open(tle_file, 'r') as f:
                    lines = f.readlines()
                    for i in range(0, len(lines), 3):
                        if i + 2 < len(lines):
                            name = lines[i].strip()
                            line1 = lines[i+1].strip()
                            line2 = lines[i+2].strip()
                            tle_data[name] = {
                                'line1': line1,
                                'line2': line2,
                                'timestamp': timestamp
                            }
                # print(f"Loaded TLE data from {tle_file}")
            except Exception as e:
                print(f"Error loading TLE file {tle_file}: {e}")
        
        return tle_data

    def load_serving_satellite_data(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Load and combine all serving satellite data files within the time window."""
        # print("Loading serving satellite data...")
        all_data = []
        
        # Calculate total minutes for progress bar
        total_minutes = int((end_time - start_time).total_seconds() / 60) + 1
        
        # Find all relevant files
        current_time = start_time
        with tqdm(total=total_minutes, desc="Loading serving satellite files") as pbar:
            while current_time <= end_time:
                date_str = current_time.strftime('%Y-%m-%d')
                file_path = os.path.join('data', f"serving_satellite_data-{date_str}-{current_time.strftime('%H-%M-%S')}.csv")
                
                if os.path.exists(file_path):
                    try:
                        # Load TLE data for this timestamp
                        tle_data = self.load_tle_data(current_time)
                        
                        # Load serving satellite data
                        df = pd.read_csv(file_path)
                        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                        
                        # Add TLE data to the DataFrame
                        df['TLE_Line1'] = df['Connected_Satellite'].map(lambda x: tle_data.get(x, {}).get('line1', None))
                        df['TLE_Line2'] = df['Connected_Satellite'].map(lambda x: tle_data.get(x, {}).get('line2', None))
                        df['TLE_Timestamp'] = df['Connected_Satellite'].map(lambda x: tle_data.get(x, {}).get('timestamp', None))
                        
                        all_data.append(df)
                        pbar.set_postfix({'file': os.path.basename(file_path)})
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                
                current_time += timedelta(minutes=1)
                pbar.update(1)
        
        if not all_data:
            print("Warning: No serving satellite data files found in the specified time window")
            return pd.DataFrame(columns=['Timestamp', 'Connected_Satellite', 'Distance', 'TLE_Line1', 'TLE_Line2', 'TLE_Timestamp'])
        
        print("Combining serving satellite data...")
        combined_df = pd.concat(all_data, ignore_index=True).sort_values('Timestamp')
        return combined_df

    def load_ping_data(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Load and combine all ping data files within the time window."""
        # print("Loading ping data...")
        all_data = []
        
        # Calculate total minutes for progress bar
        total_minutes = int((end_time - start_time).total_seconds() / 60) + 1
        
        current_time = start_time
        with tqdm(total=total_minutes, desc="Loading ping files") as pbar:
            while current_time <= end_time:
                date_str = current_time.strftime('%Y-%m-%d')
                ping_file = os.path.join('data', 'latency', date_str, f"ping-100ms-{date_str}-{current_time.strftime('%H-%M-%S')}.txt")
                
                if os.path.exists(ping_file):
                    try:
                        with open(ping_file, 'r') as f:
                            data = []
                            for line in f:
                                match = re.match(r'\[(\d+\.\d+)\].*time=(\d+\.\d+)\s+ms', line)
                                if match:
                                    timestamp = float(match.group(1))
                                    latency = float(match.group(2))
                                    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                                    if start_time <= dt <= end_time:
                                        data.append({
                                            'Timestamp': dt,
                                            'Latency_ms': latency
                                        })
                        if data:
                            df = pd.DataFrame(data)
                            all_data.append(df)
                            pbar.set_postfix({'file': os.path.basename(ping_file)})
                    except Exception as e:
                        print(f"Error loading {ping_file}: {e}")
                
                current_time += timedelta(minutes=1)
                pbar.update(1)
        
        if not all_data:
            print("Warning: No ping data files found in the specified time window")
            return pd.DataFrame(columns=['Timestamp', 'Latency_ms'])
        
        print("Combining ping data...")
        combined_df = pd.concat(all_data, ignore_index=True).sort_values('Timestamp')
        return combined_df

    def load_grpc_data(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Load and combine all GRPC status files within the time window."""
        # print("Loading GRPC data...")
        all_data = []
        
        # Calculate total minutes for progress bar
        total_minutes = int((end_time - start_time).total_seconds() / 60) + 1
        
        current_time = start_time
        with tqdm(total=total_minutes, desc="Loading GRPC files") as pbar:
            while current_time <= end_time:
                date_str = current_time.strftime('%Y-%m-%d')
                grpc_file = os.path.join('data', 'grpc', date_str, f"GRPC_STATUS-{date_str}-{current_time.strftime('%H-%M-%S')}.csv")
                
                if os.path.exists(grpc_file):
                    try:
                        df = pd.read_csv(grpc_file)
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
                        df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
                        all_data.append(df)
                        pbar.set_postfix({'file': os.path.basename(grpc_file)})
                    except Exception as e:
                        print(f"Error loading {grpc_file}: {e}")
                
                current_time += timedelta(minutes=1)
                pbar.update(1)
        
        if not all_data:
            print("Warning: No GRPC data files found in the specified time window")
            return pd.DataFrame(columns=['timestamp', 'popPingLatencyMs', 'downlinkThroughputBps', 'uplinkThroughputBps'])
        
        print("Combining GRPC data...")
        combined_df = pd.concat(all_data, ignore_index=True).sort_values('timestamp')
        return combined_df

    def create_connection_periods_csv(self, serving_df: pd.DataFrame, output_file: str):
        """Create a CSV file with unique satellite connection periods."""
        print("Creating connection periods CSV...")
        periods = []
        current_sat = None
        start_time = None
        current_data = []

        timestamps = serving_df['Timestamp']
        rows = serving_df.to_dict('records')
        
        def append_period(satellite, start, end, data):
            distances = [d['Distance'] for d in data if pd.notna(d['Distance'])]
            # Get TLE data from the first entry of the period
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
                'TLE_Line1': first_entry.get('TLE_Line1'),
                'TLE_Line2': first_entry.get('TLE_Line2'),
                'TLE_Timestamp': first_entry.get('TLE_Timestamp')
            }

        print("Processing connection periods...")
        # Set this based on your sampling interval, e.g., 1 second, 3 seconds, etc.
        MAX_GAP_SECONDS = 3  

        with tqdm(total=len(rows), desc="Analyzing connections") as pbar:
            for i, row in enumerate(rows):
                satellite = row['Connected_Satellite']
                timestamp = row['Timestamp']

                if current_sat is None:
                    # First entry
                    current_sat = satellite
                    start_time = timestamp
                    current_data = [row]
                else:
                    time_gap = (timestamp - rows[i - 1]['Timestamp']).total_seconds()
                    same_satellite = satellite == current_sat

                    if not same_satellite or time_gap > MAX_GAP_SECONDS:
                        # Determine inferred end time
                        prev_time = rows[i - 1]['Timestamp']
                        inferred_end_time = timestamp - pd.Timedelta(seconds=1)
                        # inferred_end_time = next_timestamp - pd.Timedelta(seconds=1)

                        periods.append(append_period(current_sat, start_time, inferred_end_time, current_data))

                        # Start new connection period
                        current_sat = satellite
                        start_time = timestamp
                        current_data = []

                    current_data.append(row)
                pbar.update(1)

        # Handle final period
        if current_sat is not None and current_data:
            last_timestamp = rows[-1]['Timestamp']
            periods.append(append_period(current_sat, start_time, last_timestamp, current_data))


        print("Saving connection periods...")
        periods_df = pd.DataFrame(periods)
        periods_df.to_csv(output_file, index=False)
        print(f"Saved connection periods to {output_file}")
        return periods_df

    def process_data(self, start_time: datetime, end_time: datetime):
        """Process all data within the specified time window."""
        print(f"Processing data from {start_time} to {end_time}")
        
        try:
            # Load and combine serving satellite files
            serving_df = self.load_serving_satellite_data(start_time, end_time)
            if not serving_df.empty:
                serving_df.to_csv(os.path.join(self.output_dir, 'combined_serving_satellite.csv'), index=False)
                print(f"Saved serving satellite data with {len(serving_df)} records")
            else:
                print("No serving satellite data to save")
            
            # Load and combine ping files
            ping_df = self.load_ping_data(start_time, end_time)
            if not ping_df.empty:
                ping_df.to_csv(os.path.join(self.output_dir, 'combined_ping_data.csv'), index=False)
                print(f"Saved ping data with {len(ping_df)} records")
            else:
                print("No ping data to save")
            
            # Load and combine GRPC files
            grpc_df = self.load_grpc_data(start_time, end_time)
            if not grpc_df.empty:
                grpc_df.to_csv(os.path.join(self.output_dir, 'combined_grpc_data.csv'), index=False)
                print(f"Saved GRPC data with {len(grpc_df)} records")
            else:
                print("No GRPC data to save")
            
            # Create connection periods CSV only if we have serving satellite data
            if not serving_df.empty:
                periods_df = self.create_connection_periods_csv(
                    serving_df,
                    os.path.join(self.output_dir, 'connection_periods.csv')
                )
                print(f"Found {len(periods_df)} unique connection periods")
            else:
                print("No connection periods to analyze (no serving satellite data)")
            
            print("Processing complete!")
                
        except Exception as e:
            print(f"Error: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Process satellite connection data')
    parser.add_argument('--start', required=True, help='Start time (YYYY-MM-DD-HH-MM-SS)')
    parser.add_argument('--end', required=True, help='End time (YYYY-MM-DD-HH-MM-SS)')
    parser.add_argument('--output_dir', default='output', help='Output directory for CSV files')
    
    args = parser.parse_args()
    
    # Parse time window
    start_time = SatelliteDataProcessor.parse_timestamp(args.start)
    end_time = SatelliteDataProcessor.parse_timestamp(args.end)
    
    # Create processor and process data
    processor = SatelliteDataProcessor(args.output_dir)
    processor.process_data(start_time, end_time)

if __name__ == "__main__":
    main()