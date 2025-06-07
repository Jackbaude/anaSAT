import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import multiprocessing as mp
from logger import logger
from satellite_data import SatelliteData

class DataProcessor:
    """Main class for processing satellite data and computing visibility."""
    
    def __init__(self, start_time: datetime, end_time: datetime, latitude: float, 
                 longitude: float, altitude: float, duration_minutes: int = 60, 
                 output_dir: str = 'output'):
        logger.info(f"Initializing DataProcessor with time range {start_time} to {end_time}")
        logger.info(f"Location: lat={latitude}, lon={longitude}, alt={altitude}m")
        self.start_time = start_time
        self.end_time = end_time
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.duration_minutes = duration_minutes
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
        
        self.satellite_data = SatelliteData(latitude, longitude, altitude)
        self.shared_results = None

    def process_ping_data_to_json(self, start_time: datetime, end_time: datetime) -> None:
        """Process ping data into a per-second JSON format."""
        output_file = os.path.join(self.output_dir, 'ping_data_by_second.json')
        ping_data_by_second = {}
        
        current_time = start_time.replace(microsecond=0)
        while current_time <= end_time:
            ping_data_by_second[current_time.isoformat()] = []
            current_time += timedelta(seconds=1)
        
        current_time = start_time
        total_hours = int((end_time - start_time).total_seconds() / 3600) + 1
        
        with tqdm(total=total_hours, desc="Processing ping data by second") as pbar:
            while current_time <= end_time:
                date_str = current_time.strftime('%Y-%m-%d')
                ping_file = os.path.join('data', 'latency', date_str, 
                                       f"ping-10ms-{date_str}-{current_time.strftime('%H-%M-%S')}.txt")
                
                if os.path.exists(ping_file):
                    try:
                        with open(ping_file, 'r') as f:
                            lines = f.readlines()
                            
                        for line in lines:
                            if not line.strip():
                                continue
                            
                            timestamp_match = re.search(r'\[(\d+\.\d+)\]', line)
                            latency_match = re.search(r'time=(\d+\.\d+)\s*ms', line)
                            
                            if timestamp_match and latency_match:
                                timestamp = float(timestamp_match.group(1))
                                latency = float(latency_match.group(1))
                                
                                ping_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                                
                                if start_time <= ping_time <= end_time:
                                    second_key = ping_time.replace(microsecond=0).isoformat()
                                    
                                    if second_key in ping_data_by_second:
                                        ping_data_by_second[second_key].append({
                                            'time_ms': timestamp * 1000,
                                            'latency_ms': latency
                                        })
                    
                    except Exception as e:
                        logger.error(f"Error processing ping file {ping_file}: {e}")
                else:
                    logger.warning(f"Ping file not found: {ping_file}")
                
                current_time += timedelta(hours=1)
                pbar.update(1)
        
        try:
            with open(output_file, 'w') as f:
                json.dump(ping_data_by_second, f, indent=2)
            logger.info(f"Successfully wrote ping data to {output_file}")
        except Exception as e:
            logger.error(f"Error writing ping data to {output_file}: {e}")

    def process_data(self, start_time: datetime, end_time: datetime, process_ping: bool = False):
        """Process all data within the specified time window."""
        logger.info(f"Processing data from {start_time} to {end_time}")
        
        combined_file = os.path.join(self.output_dir, 'combined_serving_satellite.csv')
        periods_file = os.path.join(self.output_dir, 'connection_periods.csv')
        
        if process_ping:
            logger.info("Processing ping data into per-second format...")
            self.process_ping_data_to_json(start_time, end_time)
        
        logger.info("Combining all serving satellite data...")
        all_serving_data = []
        
        matched_files = self.satellite_data.get_matching_serving_satellite_files(start_time, end_time)
        
        for file_time, file_path in tqdm(matched_files, desc="Loading satellite files"):
            try:
                df = self.satellite_data.process_satellite_file(file_time, file_path)
                if not df.empty:
                    df = df[(df['Timestamp'] >= start_time) & (df['Timestamp'] <= end_time)]
                    all_serving_data.append(df)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue
        
        if not all_serving_data:
            logger.error("No serving satellite data found")
            return
        
        combined_df = pd.concat(all_serving_data, ignore_index=True)
        combined_df = combined_df.sort_values('Timestamp')
        
        combined_df.to_csv(combined_file, index=False)
        logger.info(f"Saved combined serving satellite data with {len(combined_df)} records")
        
        logger.info("Processing connection periods...")
        periods_df = self.create_connection_periods_csv(combined_df, periods_file)
        logger.info(f"Created {len(periods_df)} connection periods")

    def create_connection_periods_csv(self, serving_df: pd.DataFrame, output_file: str, append: bool = False):
        """Create a CSV file with unique satellite connection periods."""
        logger.info(f"Creating connection periods CSV with {len(serving_df)} records")
        periods = []
        current_sat = None
        start_time = None
        current_data = []

        # Sort by timestamp to ensure chronological order
        serving_df = serving_df.sort_values('Timestamp')
        rows = serving_df.to_dict('records')
        logger.debug(f"Processing {len(rows)} rows of satellite data")
        
        def append_period(satellite, start, end, data):
            altitudes = [d.get('Altitude_km') for d in data if pd.notna(d.get('Altitude_km'))]
            mean_altitude = np.mean(altitudes) if altitudes else None
            duration = (end - start).total_seconds()
            logger.debug(f"Period for {satellite}: {start} to {end}, duration: {duration:.1f}s, mean altitude: {mean_altitude:.2f}km")

            first_entry = data[0]
            return {
                'Satellite': satellite,
                'Start_Time': start,
                'End_Time': end,
                'Duration_Seconds': duration,
                'Mean_Altitude_km': mean_altitude,
                'TLE_Line1': first_entry.get('TLE_Line1'),
                'TLE_Line2': first_entry.get('TLE_Line2'),
                'TLE_Timestamp': first_entry.get('TLE_Timestamp')
            }

        for i, row in enumerate(rows):
            satellite = row['Connected_Satellite']
            timestamp = row['Timestamp']

            # If this is the first row or we have a satellite change
            if current_sat is None or satellite != current_sat:
                # If we have a previous period, save it
                if current_sat is not None and current_data:
                    end_time = current_data[-1]['Timestamp']
                    logger.debug(f"Ending period for satellite {current_sat} at {end_time}")
                    periods.append(append_period(current_sat, start_time, end_time, current_data))

                # Start new period
                current_sat = satellite
                start_time = timestamp
                current_data = [row]
                logger.debug(f"Starting new period for satellite {satellite} at {timestamp}")
            else:
                # Check if this is exactly 1 second after the previous timestamp
                prev_timestamp = current_data[-1]['Timestamp']
                time_diff = (timestamp - prev_timestamp).total_seconds()
                
                if time_diff == 1:  # Must be exactly 1 second
                    # Continue current period
                    current_data.append(row)
                else:
                    # End current period and start new one
                    end_time = current_data[-1]['Timestamp']
                    logger.debug(f"Ending period for satellite {current_sat} at {end_time} due to non-consecutive timestamp")
                    periods.append(append_period(current_sat, start_time, end_time, current_data))
                    
                    # Start new period
                    current_sat = satellite
                    start_time = timestamp
                    current_data = [row]
                    logger.debug(f"Starting new period for satellite {satellite} at {timestamp}")

        # Don't forget to save the last period
        if current_sat is not None and current_data:
            end_time = current_data[-1]['Timestamp']
            logger.debug(f"Ending final period for satellite {current_sat} at {end_time}")
            periods.append(append_period(current_sat, start_time, end_time, current_data))

        periods_df = pd.DataFrame(periods)
        logger.info(f"Created {len(periods_df)} connection periods")
        
        if append and os.path.exists(output_file):
            logger.info(f"Appending {len(periods_df)} periods to existing file {output_file}")
            periods_df.to_csv(output_file, mode='a', header=False, index=False)
        else:
            logger.info(f"Writing {len(periods_df)} periods to new file {output_file}")
            periods_df.to_csv(output_file, index=False)
        
        return periods_df

    def compute_visibility(self, frame_type=2):
        """Compute visible satellites at each timestamp in the serving data."""
        logger.info(f"Computing satellite visibility with frame type {frame_type}")
        combined_file = os.path.join(self.output_dir, 'combined_serving_satellite.csv')
        if not os.path.exists(combined_file):
            logger.info("Combined serving satellite file not found, processing data...")
            self.process_data(self.start_time, self.end_time)
        
        try:
            serving_df = pd.read_csv(combined_file)
            serving_df['Timestamp'] = pd.to_datetime(serving_df['Timestamp'])
            logger.info(f"Loaded combined serving satellite data with {len(serving_df)} records")
        except Exception as e:
            logger.error(f"Error loading combined serving satellite data: {e}")
            return {}

        # Get unique timestamps from serving data
        timestamps = serving_df['Timestamp'].unique()
        batch_size = 10
        timestamp_batches = [timestamps[i:i + batch_size] for i in range(0, len(timestamps), batch_size)]
        logger.info(f"Split {len(timestamps)} timestamps into {len(timestamp_batches)} batches")
        
        tle_cache = {}
        current_hour = self.start_time.replace(minute=0, second=0, microsecond=0)
        while current_hour <= self.end_time:
            tle_data = self.satellite_data.load_tle_data(current_hour)
            if not tle_data.empty:
                tle_cache[current_hour] = tle_data
                logger.debug(f"Loaded TLE data for {current_hour} with {len(tle_data)} records")
            current_hour += timedelta(hours=1)
        
        logger.info(f"Cached TLE data for {len(tle_cache)} hours")
        
        timestamp_batches = [[ts.isoformat() for ts in batch] for batch in timestamp_batches]
        
        observer_data = {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'altitude': self.altitude
        }
        
        tle_cache_serializable = {
            hour.isoformat(): df.to_dict('records') 
            for hour, df in tle_cache.items()
        }
        
        def process_batch(batch_data):
            batch_timestamps, serving_data, observer_info, tle_data = batch_data
            batch_results = {}
            
            serving_df = pd.DataFrame(serving_data)
            serving_df['Timestamp'] = pd.to_datetime(serving_df['Timestamp'])
            
            for ts_str in batch_timestamps:
                ts = datetime.fromisoformat(ts_str)
                
                mask = serving_df['Timestamp'] <= ts
                if not mask.any():
                    continue
                    
                grpc_data = serving_df[mask].iloc[-1]
                
                tilt_deg = grpc_data.get('tiltAngleDeg', 0)
                rotation_deg = grpc_data.get('boresightAzimuthDeg', 0)
                fov_azimuth = grpc_data.get('desiredBoresightAzimuthDeg', 0)
                fov_elevation = grpc_data.get('desiredBoresightElevationDeg', 0)
                
                hour_key = ts.replace(minute=0, second=0, microsecond=0).isoformat()
                if hour_key not in tle_data:
                    continue
                
                tle_records = tle_data[hour_key]
                visible_sats_data = []
                
                for sat_data in tle_records:
                    try:
                        alt, az = get_satellite_position(
                            sat_data, ts, observer_info['latitude'],
                            observer_info['longitude'], observer_info['altitude'],
                            self.satellite_data.ts
                        )
                        
                        if is_valid_satellite(alt, az, tilt_deg, rotation_deg):
                            visible_sats_data.append({
                                'satellite': sat_data['satellite_name'],
                                'sat_elevation_deg': alt,
                                'sat_azimuth_deg': az,
                                'UT_boresight_elevation': tilt_deg,
                                'UT_boresight_azimuth': rotation_deg,
                                'desired_boresight_azimuth': fov_azimuth,
                                'desired_boresight_elevation': fov_elevation,
                                'tle_line1': sat_data['tle_line1'],
                                'tle_line2': sat_data['tle_line2']
                            })
                    except Exception as e:
                        logger.error(f"Error processing satellite {sat_data.get('satellite_name', 'Unknown')}: {e}")
                        continue
                
                if visible_sats_data:
                    batch_results[ts_str] = visible_sats_data
            
            return batch_results
        
        num_processes = min(12, mp.cpu_count())
        output_file = os.path.join(self.output_dir, 'satellite_visibility.json')
        
        batch_data = [
            (batch, serving_df.to_dict('records'), observer_data, tle_cache_serializable)
            for batch in timestamp_batches
        ]
        
        with mp.Pool(num_processes) as pool:
            with tqdm(total=len(timestamp_batches), desc="Computing visibility at timestamps") as pbar:
                results = []
                for batch_result in pool.imap_unordered(process_batch, batch_data):
                    results.append(batch_result)
                    pbar.update(1)
        
        visibility_results = {}
        for batch_result in results:
            visibility_results.update(batch_result)
        
        with open(output_file, 'w') as f:
            json.dump(visibility_results, f)
        
        return visibility_results 