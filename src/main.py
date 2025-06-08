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
import pandas as pd
from tqdm import tqdm

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_timestamp(ts: str) -> datetime:
    """Parse timestamp string into datetime object."""
    return datetime.strptime(ts, "%Y-%m-%d-%H-%M-%S").replace(tzinfo=timezone.utc)

def parse_args():
    parser = argparse.ArgumentParser(description='Process satellite data')
    parser.add_argument('--start', required=True, help='Start time (YYYY-MM-DD-HH-MM-SS)')
    parser.add_argument('--end', required=True, help='End time (YYYY-MM-DD-HH-MM-SS)')
    parser.add_argument('--lat', required=True, type=float, help='Latitude in degrees')
    parser.add_argument('--lon', required=True, type=float, help='Longitude in degrees')
    parser.add_argument('--alt', required=True, type=float, help='Altitude in meters')
    parser.add_argument('--output_dir', default='output', help='Output directory')
    parser.add_argument('--process_ping', action='store_true', help='Process ping data')
    parser.add_argument('--compute_visibility', action='store_true', 
                       help='Compute satellite visibility at reconfiguration periods')
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    try:
        start_time = parse_timestamp(args.start)
        end_time = parse_timestamp(args.end)
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        return

    processor = DataProcessor(
        start_time=start_time,
        end_time=end_time,
        latitude=args.lat,
        longitude=args.lon,
        altitude=args.alt,
        output_dir=args.output_dir
    )

    # Process the data
    processor.process_data(start_time, end_time, process_ping=args.process_ping)
    
    # Compute reconfiguration visibility if requested
    if args.compute_visibility:
        logger.info("Computing satellite visibility at reconfiguration periods...")
        timestamps = processor.get_reconfiguration_timestamps(start_time, end_time)
        print(f"Computing visibility at {timestamps[0]} to {timestamps[-1]}")
        processor.compute_reconfiguration_visibility(timestamps)
    
if __name__ == "__main__":
    main() 