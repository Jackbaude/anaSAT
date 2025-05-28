# Satellite Data Processing Tool

A Python tool for processing and analyzing Starlink satellite connection data, including serving satellite information, latency measurements, GRPC status data, and satellite visibility calculations.

## Overview

This tool processes various types of satellite-related data files and combines them into a comprehensive dataset for analysis. It handles:
- Serving satellite data (which satellite is currently connected)
- TLE (Two-Line Element) data for satellite orbital information
- Ping latency measurements
- GRPC status data
- Satellite visibility calculations and Field of View (FOV) analysis

## Data Structure

The tool expects data files to be organized in the following structure:

```
data/
├── serving_satellite_data-YYYY-MM-DD-HH-MM-SS.csv
├── TLE/
│   └── YYYY-MM-DD/
│       └── starlink-tle-YYYY-MM-DD-HH-MM-SS.txt
├── latency/
│   └── YYYY-MM-DD/
│       └── ping-100ms-YYYY-MM-DD-HH-MM-SS.txt
└── grpc/
    └── YYYY-MM-DD/
        └── GRPC_STATUS-YYYY-MM-DD-HH-MM-SS.csv
```

### Input File Formats

1. **Serving Satellite Data** (`serving_satellite_data-*.csv`):
   - Contains information about which satellite is currently serving
   - Columns: Timestamp, Connected_Satellite, Distance, etc.

2. **TLE Data** (`starlink-tle-*.txt`):
   - Contains orbital elements for Starlink satellites
   - Format: Three lines per satellite (name, line1, line2)

3. **Ping Data** (`ping-100ms-*.txt`):
   - Contains latency measurements
   - Format: `[timestamp] ... time=latency ms`

4. **GRPC Status** (`GRPC_STATUS-*.csv`):
   - Contains GRPC connection status and metrics
   - Columns: timestamp, popPingLatencyMs, downlinkThroughputBps, uplinkThroughputBps, tiltAngleDeg, boresightAzimuthDeg

## Output Files

The tool generates the following output files in the specified output directory:

1. `combined_serving_satellite.csv`:
   - Combined serving satellite data with TLE information
   - Columns: Timestamp, Connected_Satellite, Distance, TLE_Line1, TLE_Line2, TLE_Timestamp

2. `combined_ping_data.csv`:
   - Combined ping latency measurements
   - Columns: Timestamp, Latency_ms

3. `combined_grpc_data.csv`:
   - Combined GRPC status data
   - Columns: timestamp, popPingLatencyMs, downlinkThroughputBps, uplinkThroughputBps

4. `connection_periods.csv`:
   - Analysis of satellite connection periods
   - Columns: Satellite, Start_Time, End_Time, Duration_Seconds, Max_Distance, Min_Distance, First_Distance, Last_Distance, Mean_Distance, Std_Distance, TLE_Line1, TLE_Line2, TLE_Timestamp

5. `satellite_visibility.json`:
   - Analysis of visible satellites at handover timestamps
   - Contains satellite positions and FOV information
   - Includes altitude, azimuth, and FOV parameters for each visible satellite

## Usage

```bash
python process_satellite_data.py --start YYYY-MM-DD-HH-MM-SS --end YYYY-MM-DD-HH-MM-SS --lat LATITUDE --lon LONGITUDE --alt ALTITUDE [--output_dir OUTPUT_DIR]
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
python process_satellite_data.py --start 2025-04-20-23-00-00 --end 2025-05-02-19-00-00 --lat 37.7749 --lon -122.4194 --alt 100 --output_dir analysis_results
```

## Features

1. **Progress Tracking**:
   - Progress bars for file loading and processing
   - Real-time status updates
   - Estimated time remaining

2. **Error Handling**:
   - Graceful handling of missing files
   - Warning messages for missing data
   - Detailed error reporting

3. **Data Processing**:
   - Automatic TLE data matching with serving satellites
   - Connection period analysis
   - Statistical calculations for each connection period
   - Satellite visibility calculations
   - Field of View (FOV) analysis

4. **Satellite Visibility**:
   - Calculates visible satellites at handover timestamps
   - Determines if satellites are within the dish's FOV
   - Accounts for dish tilt and rotation
   - Uses elliptical FOV model for accurate visibility determination

5. **Output Organization**:
   - Chronologically sorted data
   - Combined datasets for easy analysis
   - Comprehensive connection period analysis
   - JSON output for satellite visibility data

## Dependencies

- Python 3.6+
- pandas
- numpy
- tqdm
- skyfield (for satellite position calculations)

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install pandas numpy tqdm skyfield
```

## Notes

- The tool processes data in one-minute intervals
- TLE data is matched with serving satellite data at each timestamp
- Connection periods are determined by changes in the serving satellite
- Missing data is handled gracefully with appropriate warnings
- All timestamps are in UTC
- Satellite visibility is calculated using an elliptical FOV model
- The FOV model accounts for dish tilt and rotation
- Handover timestamps are generated at 12, 27, 42, and 57 seconds of each minute
