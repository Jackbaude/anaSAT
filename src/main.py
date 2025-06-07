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

import argparse
from datetime import datetime, timezone
from data_processor import DataProcessor
from logger import logger
import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_timestamp(ts: str) -> datetime:
    """Parse timestamp string into datetime object."""
    return datetime.strptime(ts, "%Y-%m-%d-%H-%M-%S").replace(tzinfo=timezone.utc)

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Process satellite connection data')
    parser.add_argument('--start', required=True, help='Start time (YYYY-MM-DD-HH-MM-SS)')
    parser.add_argument('--end', required=True, help='End time (YYYY-MM-DD-HH-MM-SS)')
    parser.add_argument('--output_dir', default='output', help='Output directory for CSV files')
    parser.add_argument('--lat', type=float, required=True, help='Latitude of the observer')
    parser.add_argument('--lon', type=float, required=True, help='Longitude of the observer')
    parser.add_argument('--alt', type=float, required=True, help='Altitude of the observer (in meters)')
    parser.add_argument('--duration', type=int, default=60, help='Duration of each data file in minutes')
    parser.add_argument('--process_ping', action='store_true', help='Process ping data (default: False)')

    args = parser.parse_args()
    
    # Parse time window
    start_time = parse_timestamp(args.start)
    end_time = parse_timestamp(args.end)
    
    # Create processor and process data
    processor = DataProcessor(
        start_time=start_time,
        end_time=end_time,
        latitude=args.lat,
        longitude=args.lon,
        altitude=args.alt,
        duration_minutes=args.duration,
        output_dir=args.output_dir
    )
    
    processor.process_data(start_time, end_time, process_ping=args.process_ping)

if __name__ == "__main__":
    main() 