# Satellite Data Analysis Tool

A tool for analyzing satellite visibility, connection periods, and latency data.

## Features

### Ping Data Processing

1. Raw ping data is read from files in the format:
   ```
   [timestamp] ... time=latency ms
   ```

2. Data is processed into a JSON structure:
   ```json
   {
     "2025-05-31T17:00:00": [
       {
         "time_ms": 1717167600000,
         "latency_ms": 15.2
       },
       ...
     ],
     ...
   }
   ```

3. Features:
   - Processes data in one-hour chunks
   - Handles missing files gracefully
   - Maintains chronological order
   - Outputs to `ping_data_by_second.json`

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

### Satellite Visibility

The tool computes satellite visibility at specific reconfiguration timestamps (12, 27, 42, and 57 seconds of each minute):

1. Memory-Optimized Processing:
   - Processes timestamps in chunks of 1000 to manage memory usage
   - Uses 50% of available CPU cores to prevent system overload
   - Saves intermediate results after each chunk
   - Implements robust error handling for individual timestamps

2. Visibility Computation:
   - Uses 3D cone angle check for accurate FOV calculations
   - Considers antenna tilt and boresight parameters
   - Supports different antenna models with varying FOV angles
   - Minimum elevation angle of 20 degrees

3. Output:
   - Saves results to `reconfiguration_visibility.json`
   - Includes satellite positions, FOV parameters, and TLE data
   - Maintains progress even if process is interrupted

### Output Files

1. `combined_serving_satellite.csv`:
   - Combined serving satellite data with TLE information
   - Columns: Timestamp, Connected_Satellite, Distance, TLE_Line1, TLE_Line2, TLE_Timestamp

2. `connection_periods.csv`:
   - Analysis of satellite connection periods
   - Columns: Satellite, Start_Time, End_Time, Duration_Seconds, Mean_Altitude_km, TLE_Line1, TLE_Line2, TLE_Timestamp

3. `ping_data_by_second.json`:
   - Processed ping latency data
   - Organized by second with timestamp and latency measurements
   - Used for correlation with satellite connection periods

4. `reconfiguration_visibility.json`:
   - Analysis of visible satellites at reconfiguration timestamps
   - Contains satellite positions, FOV parameters, and visibility information
   - Updated incrementally as computation progresses

## Usage

```bash
python src/main.py --start YYYY-MM-DD-HH-MM-SS --end YYYY-MM-DD-HH-MM-SS --lat LATITUDE --lon LONGITUDE --alt ALTITUDE [--output_dir OUTPUT_DIR] [--process_ping] [--compute_visibility]
```

### Arguments

- `--start`: Start time in format YYYY-MM-DD-HH-MM-SS (required)
- `--end`: End time in format YYYY-MM-DD-HH-MM-SS (required)
- `--lat`: Observer's latitude in degrees (required)
- `--lon`: Observer's longitude in degrees (required)
- `--alt`: Observer's altitude in meters (required)
- `--output_dir`: Output directory for CSV files (default: 'output')
- `--process_ping`: Flag to process ping data (optional)
- `--compute_visibility`: Flag to compute visibility at reconfiguration timestamps (optional)

### Example

```bash
# Process data with ping and visibility computation
python src/main.py --start 2025-05-31-17-00-00 --end 2025-05-31-18-00-00 --lat 37.7749 --lon -122.4194 --alt 100 --process_ping --compute_visibility
```

## System Requirements

- Python 3.8 or higher
- Sufficient disk space for output files
- Recommended: At least 8GB RAM for processing large time ranges
- For large datasets, consider processing in smaller time windows

## Memory Management

The tool implements several strategies to manage memory usage:

1. Chunked Processing:
   - Processes timestamps in batches of 1000
   - Saves intermediate results after each chunk
   - Allows resuming from last successful chunk

2. Resource Allocation:
   - Uses 50% of available CPU cores
   - Implements LRU caching for TLE data
   - Cleans up resources after each chunk

3. Error Handling:
   - Graceful handling of individual timestamp failures
   - Continues processing remaining chunks if one fails
   - Detailed logging of errors and progress

## Troubleshooting

If you encounter memory issues:

1. Reduce the time window being processed
2. Process data in smaller chunks by splitting the time range
3. Monitor system resources during processing
4. Check the logs for specific error messages

For other issues, check the log files in the output directory for detailed error information.
