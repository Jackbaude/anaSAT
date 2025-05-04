#!/usr/bin/env python3
"""
Script to process satellite connection periods between specified dates.
"""

import os
import sys
from process_satellite_data import SatelliteAnalysis
import logging
from datetime import datetime, timedelta
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def get_ping_file_path(timestamp):
    """
    Get the correct ping file path for a given timestamp.
    
    Args:
        timestamp (str): Timestamp in format 'YYYY-MM-DD-HH-MM-SS'
        
    Returns:
        str: Path to the ping file
    """
    date = timestamp[:10]  # Get YYYY-MM-DD
    return os.path.join("data", "latency", date, f"ping-100ms-{timestamp}.txt")

def get_serving_file_path(timestamp):
    """
    Get the correct serving file path for a given timestamp.
    
    Args:
        timestamp (str): Timestamp in format 'YYYY-MM-DD-HH-MM-SS'
        
    Returns:
        str: Path to the serving file
    """
    return f"data/serving_satellite_data-{timestamp}.csv"

def process_connection_periods(start_date, end_date):
    """
    Process satellite connection periods between start_date and end_date.
    
    Args:
        start_date (str): Start date in format 'YYYY-MM-DD-HH-MM-SS'
        end_date (str): End date in format 'YYYY-MM-DD-HH-MM-SS'
    """
    try:
        # Initialize analysis with the start date
        analysis = SatelliteAnalysis(timestamp=start_date)
        
        # Load the initial data
        serving_file = get_serving_file_path(start_date)
        ping_file = get_ping_file_path(start_date)
        
        if not os.path.exists(serving_file):
            logger.error(f"Serving file not found: {serving_file}")
            return
        if not os.path.exists(ping_file):
            logger.error(f"Ping file not found: {ping_file}")
            return
            
        # Process the data
        logger.info(f"Processing data from {start_date} to {end_date}")
        analysis.process_data(serving_file, ping_file)
        
        # Create the connection periods CSV
        output_file = analysis.create_connection_periods_csv(
            start_time=start_date,
            end_time=end_date,
            output_dir="connection_periods"
        )
        
        if output_file:
            logger.info(f"Successfully created connection periods CSV: {output_file}")
        else:
            logger.error("Failed to create connection periods CSV")
            
    except Exception as e:
        logger.error(f"Error processing connection periods: {e}")
        raise

def main():
    """Main function to process satellite connection periods."""
    # Define the date range
    start_date = "2025-04-19-06-00-00"
    end_date = "2025-04-25-00-10-55"
    
    # Process the connection periods
    process_connection_periods(start_date, end_date)

if __name__ == "__main__":
    main() 