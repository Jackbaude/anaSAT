# Satellite Data Processing Tool

A Python tool for processing and analyzing Starlink satellite connection data, including serving satellite information, latency measurements, and satellite visibility calculations.

## Overview

This tool processes various types of satellite-related data files and combines them into a comprehensive dataset for analysis. It handles:
- Serving satellite data (which satellite is currently connected)
- TLE (Two-Line Element) data for satellite orbital information
- Ping latency measurements
- Satellite visibility calculations

## Project Structure

```
anaSAT/
├── src/
│   ├── data_processor.py    # Main data processing logic
│   ├── satellite_data.py    # Satellite data handling
│   ├── satellite_utils.py   # Utility functions
│   ├── logger.py           # Logging configuration
│   └── main.py             # Entry point
├── data/
│   ├── serving_satellite_data-YYYY-MM-DD-HH-MM-SS.csv
│   ├── TLE/
│   │   └── YYYY-MM-DD/
│   │       └── starlink-tle-YYYY-MM-DD-HH-MM-SS.txt
│   └── latency/
│       └── YYYY-MM-DD/
│           └── ping-10ms-YYYY-MM-DD-HH-MM-SS.txt
└── output/
    ├── combined_serving_satellite.csv
    ├── connection_periods.csv
    └── satellite_visibility.json
```

## Environment Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- pandas
- numpy
- tqdm
- skyfield
- python-dateutil

## Data Processing

### Connection Periods

The tool processes satellite connection data to identify distinct connection periods:

1. Data is sorted chronologically
2. Connection periods are identified by:
   - Changes in satellite ID
   - Gaps in consecutive timestamps (must be exactly 1 second apart)
3. For each period, the following is calculated:
   - Start and end times
   - Duration in seconds
   - Mean altitude
   - TLE information

### Output Files

1. `combined_serving_satellite.csv`:
   - Combined serving satellite data with TLE information
   - Columns: Timestamp, Connected_Satellite, Distance, TLE_Line1, TLE_Line2, TLE_Timestamp

2. `connection_periods.csv`:
   - Analysis of satellite connection periods
   - Columns: Satellite, Start_Time, End_Time, Duration_Seconds, Mean_Altitude_km, TLE_Line1, TLE_Line2, TLE_Timestamp

3. `satellite_visibility.json`:
   - Analysis of visible satellites at timestamps
   - Contains satellite positions and visibility information

## Usage

```bash
python src/main.py --start YYYY-MM-DD-HH-MM-SS --end YYYY-MM-DD-HH-MM-SS --lat LATITUDE --lon LONGITUDE --alt ALTITUDE [--output_dir OUTPUT_DIR]
```

### Arguments

- `--start`: Start time in format YYYY-MM-DD-HH-MM-SS (required)
- `--end`: End time in format YYYY-MM-DD-HH-MM-SS (required)
- `--lat`: Observer's latitude in degrees (required)
- `--lon`: Observer's longitude in degrees (required)
- `--alt`: Observer's altitude in meters (required)
- `--output_dir`: Output directory for CSV files (default: 'output')

### Example

```bash
python src/main.py --start 2025-05-31-17-00-00 --end 2025-05-31-18-00-00 --lat 37.7749 --lon -122.4194 --alt 100
```

## Features

1. **Data Processing**:
   - Automatic TLE data matching with serving satellites
   - Precise connection period analysis with exact timestamp matching
   - Statistical calculations for each connection period
   - Satellite visibility calculations

2. **Error Handling**:
   - Graceful handling of missing files
   - Warning messages for missing data
   - Detailed error reporting

3. **Progress Tracking**:
   - Progress bars for file loading and processing
   - Real-time status updates
   - Detailed logging

## Notes

- All timestamps are in UTC
- Connection periods require exactly consecutive timestamps (1-second intervals)
- Missing data is handled gracefully with appropriate warnings
- The tool uses multiprocessing for efficient data processing
- Detailed logging is available for debugging and monitoring
