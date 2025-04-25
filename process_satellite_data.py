#!/usr/bin/env python3
"""
Script to analyze satellite connections and latency data.
"""

import pandas as pd
import os
import sys
import re
from datetime import datetime, timezone
import logging
from satellite_plotter import SatellitePlotter
from skyfield.api import load, wgs84, EarthSatellite

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

    def load_tle_data(self):
        """Load and parse TLE data for the given timestamp."""
        logger.info("Loading TLE data")
        tle_file = os.path.join("data", "TLE", self.timestamp[:10], f"starlink-tle-{self.timestamp}.txt")
        
        if not os.path.exists(tle_file):
            logger.warning(f"TLE file not found: {tle_file}")
            return None, None
        
        try:
            with open(tle_file, 'r') as f:
                lines = f.readlines()
            
            tle_data = {}
            satellite_objects = {}
            ts = load.timescale()
            
            for i in range(0, len(lines), 3):
                if i + 2 >= len(lines):
                    break
                    
                name = lines[i].strip()
                line1 = lines[i+1].strip()
                line2 = lines[i+2].strip()
                
                tle_data[name] = (line1, line2)
                satellite_objects[name] = EarthSatellite(line1, line2, name, ts)
            
            self.tle_data = tle_data
            self.satellite_objects = satellite_objects
            logger.info(f"Loaded TLE data for {len(tle_data)} satellites")
            return tle_data, satellite_objects
            
        except Exception as e:
            logger.error(f"Error loading TLE data: {e}")
            return None, None

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