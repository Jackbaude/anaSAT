import os
import re
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from tqdm import tqdm
from logger import logger
from satellite_data import SatelliteData
from skyfield.api import EarthSatellite, wgs84
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import multiprocessing

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

    def process_ping_data_to_csv(self, start_time: datetime, end_time: datetime) -> None:
        """Process ping data into a clean CSV format with precise timestamps and latency measurements."""
        output_file = os.path.join(self.output_dir, 'ping_data.csv')
        records = []
        
        # Ensure start_time and end_time are timezone-aware
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)
        
        current_time = start_time
        total_hours = int((end_time - start_time).total_seconds() / 3600) + 1
        
        with tqdm(total=total_hours, desc="Processing ping data") as pbar:
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
                                
                                # Create timezone-aware datetime from timestamp
                                ping_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                                
                                if start_time <= ping_time <= end_time:
                                    records.append({
                                        'timestamp': ping_time,
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
            # Create DataFrame and sort by timestamp
            df = pd.DataFrame(records)
            if not df.empty:
                df = df.sort_values('timestamp')
                df.to_csv(output_file, index=False)
                logger.info(f"Successfully wrote {len(df)} ping records to {output_file}")
            else:
                logger.warning("No ping records found in the specified time range")
        except Exception as e:
            logger.error(f"Error writing ping data to {output_file}: {e}")

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


    def process_data(self, start_time: datetime, end_time: datetime, process_ping: bool = False):
        """Process all data within the specified time window."""
        logger.info(f"Processing data from {start_time} to {end_time}")
        
        # Ensure start_time and end_time are timezone-aware
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=timezone.utc)
        
        combined_file = os.path.join(self.output_dir, 'combined_serving_satellite.csv')
        periods_file = os.path.join(self.output_dir, 'connection_periods.csv')
        
        if process_ping:
            logger.info("Processing ping data into CSV format...")
            self.process_ping_data_to_csv(start_time, end_time)
        
        logger.info("Loading processed obstruction data...")
        all_serving_data = []
        
        # Get and process satellite data files
        matched_files = self.satellite_data.get_matching_serving_satellite_files(start_time, end_time)
        
        for file_time, file_path in tqdm(matched_files, desc="Loading satellite files"):
            try:
                df = self.satellite_data.process_satellite_file(file_time, file_path)
                if not df.empty:
                    # Convert start_time and end_time to pandas timestamps for comparison
                    start_ts = pd.Timestamp(start_time)
                    end_ts = pd.Timestamp(end_time)
                    df = df[(df['Timestamp'] >= start_ts) & (df['Timestamp'] <= end_ts)]
                    all_serving_data.append(df)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue
        
        if not all_serving_data:
            logger.error("No serving satellite data found")
            return
        
        combined_df = pd.concat(all_serving_data, ignore_index=True)
        combined_df = combined_df.sort_values('Timestamp')
        
        # Get and process obstruction data files
        logger.info("Loading processed obstruction data...")
        all_obstruction_data = []
        
        # Get all processed obstruction data files in the time range
        current_time = start_time
        while current_time <= end_time:
            date_str = current_time.strftime('%Y-%m-%d')
            hour_str = current_time.strftime('%H')
            minute_str = current_time.strftime('%M')
            
            obstruction_file = os.path.join('data', f'processed_obstruction-data-{date_str}-{hour_str}-{minute_str}-00.csv')
            
            if os.path.exists(obstruction_file):
                try:
                    df = pd.read_csv(obstruction_file)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    all_obstruction_data.append(df)
                except Exception as e:
                    logger.error(f"Error processing obstruction file {obstruction_file}: {e}")
            
            current_time += timedelta(minutes=1)
        
        # Merge obstruction data if available
        if all_obstruction_data:
            obstruction_df = pd.concat(all_obstruction_data, ignore_index=True)
            obstruction_df = obstruction_df.sort_values('timestamp')
            logger.info(f"Merging {len(obstruction_df)} obstruction records with satellite data")
            
            # Merge on timestamp
            combined_df = pd.merge_asof(
                combined_df,
                obstruction_df,
                left_on='Timestamp',
                right_on='timestamp',
                direction='nearest'
            )
        
        combined_df.to_csv(combined_file, index=False)
        logger.info(f"Saved combined serving satellite data with {len(combined_df)} records")
        
        logger.info("Processing connection periods...")
        periods_df = self.create_connection_periods_csv(combined_df, periods_file)
        logger.info(f"Created {len(periods_df)} connection periods")



    from datetime import datetime, timedelta

    def get_reconfiguration_timestamps(self, start_time: datetime, end_time: datetime) -> list:
        """Return timestamps at 12, 27, 42, and 57 seconds between start and end times."""
        target_seconds = [12, 27, 42, 57]
        timestamps = []

        # Round start_time to the next second
        current_minute = start_time.replace(second=0, microsecond=0)

        while current_minute <= end_time:
            for sec in target_seconds:
                t = current_minute.replace(second=sec)
                if start_time <= t <= end_time:
                    timestamps.append(t)
            current_minute += timedelta(minutes=1)

        return timestamps


    def rotate_points(self, x: np.ndarray, y: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
        """Rotates points by the given angle."""
        x_rot = x * np.cos(angle) - y * np.sin(angle)
        y_rot = x * np.sin(angle) + y * np.cos(angle)
        return x_rot, y_rot

    def get_fov_degree_from_model(self, model: str) -> float:
        """
        Returns the field of view (FoV) in degrees based on the antenna model.
        """
        if str.startswith(model, "mini_") or str.startswith(model, "rev3_") or str.startswith(model, "rev4_"):
            return 110.0
        elif str.startswith(model, "hp1_"):
            return 140.0
        else:
            return 110.0


    def altaz_to_unit_vector(self, alt_deg, az_deg):
        """Convert (alt, az) in degrees to a 3D unit vector."""
        alt_rad = math.radians(alt_deg)
        az_rad = math.radians(az_deg)
        x = math.cos(alt_rad) * math.cos(az_rad)
        y = math.cos(alt_rad) * math.sin(az_rad)
        z = math.sin(alt_rad)
        return (x, y, z)

    def angular_separation(self, vec1, vec2):
        """Compute angular separation (in degrees) between two unit vectors."""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        dot = max(min(dot, 1.0), -1.0)  # Clamp for numerical stability
        return math.degrees(math.acos(dot))

    # IGNORE RIGHT NOW, SATLLITES NOT ALWAYS IN THE FOV?
    def is_inside_fov(self, sat_alt, sat_az, tilt_deg, rotation_deg):
        """
        Check if satellite is within angular cone of dish boresight.

        Args:
            sat_alt (float): Satellite altitude (degrees)
            sat_az (float): Satellite azimuth (degrees)
            tilt_deg (float): Dish tilt from zenith (degrees)
            rotation_deg (float): Dish boresight azimuth direction (degrees)
            base_radius (float): Half-angle of the conical FOV (degrees)

        Returns:
            bool: True if satellite is inside FOV, False otherwise
        """
        if sat_alt is None or sat_az is None:
            return False

        sat_vec = self.altaz_to_unit_vector(sat_alt, sat_az)
        boresight_alt = 90 - tilt_deg  # zenith is 90
        boresight_vec = self.altaz_to_unit_vector(boresight_alt, rotation_deg)
        
        separation = self.angular_separation(sat_vec, boresight_vec)
        # return separation <= base_radius
        if sat_alt > 20:
            return True

    # def is_inside_fov(self, alt, az, tilt_deg, rotation_deg, base_radius=55):
    #     """
    #     Check if a satellite is within the valid elliptical FOV projected on the dome.

    #     Args:
    #         alt (float): Satellite altitude in degrees (0 = horizon, 90 = zenith)
    #         az (float): Satellite azimuth in degrees (0 = north, increasing clockwise)
    #         tilt_deg (float): Dish tilt angle in degrees (0 = pointing at zenith)
    #         rotation_deg (float): Dish boresight rotation angle in degrees (rotation of ellipse)
    #         base_radius (float): Base radius of the circular FOV before tilt distortion

    #     Returns:
    #         bool: True if satellite is inside the elliptical FOV, False otherwise
    #     """
    #     if alt is None or az is None:
    #         return False

    #     # Convert alt/az to polar plot coordinates (r = 90 - alt to map zenith to center)
    #     r = 90 - alt
    #     theta = math.radians(az)

    #     # Convert polar to Cartesian (x, y) - satellite position on dome projection
    #     x = r * math.cos(theta)
    #     y = r * math.sin(theta)

    #     # Calculate dish boresight direction after tilt and rotation
    #     # Tilt rotates the boresight away from zenith in the direction of rotation_deg
    #     boresight_alt = 90 - tilt_deg  # altitude of dish boresight
    #     boresight_az = rotation_deg    # azimuth of dish boresight direction
        
    #     # Convert boresight to Cartesian coordinates
    #     boresight_r = 90 - boresight_alt
    #     boresight_theta = math.radians(boresight_az)
    #     boresight_x = boresight_r * math.cos(boresight_theta)
    #     boresight_y = boresight_r * math.sin(boresight_theta)

    #     # Translate coordinates so dish boresight is at origin
    #     dx = x - boresight_x
    #     dy = y - boresight_y

    #     # For a tilted dish, the FOV becomes elliptical when projected onto the dome
    #     # The ellipse is stretched in the direction perpendicular to the tilt
        
    #     # Rotate coordinates to align with tilt direction (boresight direction)
    #     angle_rad = math.radians(-rotation_deg)
    #     dx_rot = dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
    #     dy_rot = dx * math.sin(angle_rad) + dy * math.cos(angle_rad)

    #     # Calculate ellipse parameters
    #     # Along tilt direction (x-axis after rotation): compressed by cos(tilt)
    #     # Perpendicular to tilt (y-axis after rotation): unchanged
    #     tilt_rad = math.radians(tilt_deg)
    #     x_radius = base_radius * math.cos(tilt_rad)  # compressed
    #     y_radius = base_radius  # unchanged

    #     # Ellipse equation: (dx_rot / x_radius)^2 + (dy_rot / y_radius)^2 <= 1
    #     inside = (dx_rot**2 / x_radius**2) + (dy_rot**2 / y_radius**2) <= 1
    #     return inside
        

    @lru_cache(maxsize=1000)
    def _get_cached_tle_data(self, timestamp_str: str) -> pd.DataFrame:
        """Cached version of TLE data loading."""
        timestamp = datetime.fromisoformat(timestamp_str)
        return self.satellite_data.load_tle_data(timestamp)

    def _compute_visibility_worker(self, timestamp_str: str) -> tuple:
        """Worker function for parallel visibility computation."""
        timestamp = datetime.fromisoformat(timestamp_str)
        visible_satellites = self.compute_visibility_at_timestamp(timestamp)
        return timestamp_str, visible_satellites

    def compute_reconfiguration_visibility(self, timestamps: list) -> dict:
        """Compute satellite visibility at all reconfiguration periods using parallel processing."""
        logger.info(f"Computing visibility at reconfiguration periods from {timestamps[0]} to {timestamps[-1]}")
        
        # Print out all timestamps we'll be using
        logger.info("Timestamps for visibility computation:")
        for ts in timestamps:
            logger.info(f"  {ts.isoformat()} (second: {ts.second})")
        
        # Convert timestamps to strings for caching
        timestamp_strings = [ts.isoformat() for ts in timestamps]
        
        # Determine number of processes (use 50% of available CPUs to be more conservative)
        num_processes = max(1, int(multiprocessing.cpu_count() * 0.5))
        logger.info(f"Using {num_processes} processes for parallel computation")
        
        # Process in smaller chunks to manage memory
        chunk_size = 1000  # Process 1000 timestamps at a time
        visibility_data = {}
        
        for i in range(0, len(timestamp_strings), chunk_size):
            chunk = timestamp_strings[i:i + chunk_size]
            logger.info(f"Processing chunk {i//chunk_size + 1} of {(len(timestamp_strings) + chunk_size - 1)//chunk_size}")
            
            try:
                with ProcessPoolExecutor(max_workers=num_processes) as executor:
                    # Submit all tasks in this chunk
                    future_to_timestamp = {
                        executor.submit(self._compute_visibility_worker, ts_str): ts_str 
                        for ts_str in chunk
                    }
                    
                    # Process results as they complete
                    for future in tqdm(as_completed(future_to_timestamp), total=len(chunk), 
                                     desc=f"Computing visibility (chunk {i//chunk_size + 1})"):
                        try:
                            ts_str, visible_satellites = future.result()
                            if visible_satellites:
                                visibility_data[ts_str] = visible_satellites
                                logger.info(f"Timestamp {ts_str}: Found {len(visible_satellites)} visible satellites")
                            else:
                                logger.info(f"Timestamp {ts_str}: No visible satellites")
                        except Exception as e:
                            logger.error(f"Error processing timestamp {future_to_timestamp[future]}: {e}")
                            continue
                
                # Save intermediate results after each chunk
                output_file = os.path.join(self.output_dir, 'reconfiguration_visibility.json')
                import json
                with open(output_file, 'w') as f:
                    json.dump(visibility_data, f, indent=2)
                logger.info(f"Saved intermediate results to {output_file}")
                
            except Exception as e:
                logger.error(f"Error processing chunk {i//chunk_size + 1}: {e}")
                continue
        
        logger.info(f"Completed visibility computation with {len(visibility_data)} timestamps")
        return visibility_data

    def compute_visibility_at_timestamp(self, timestamp: datetime) -> list:
        """Compute satellite visibility at a specific timestamp."""
        logger.debug(f"Computing visibility at {timestamp}")
        visible_satellites = []
        
        # Load TLE data for this timestamp (using cached version)
        tle_data = self._get_cached_tle_data(timestamp.isoformat())
        
        if tle_data.empty:
            logger.warning(f"No TLE data available for {timestamp}")
            return visible_satellites
        
        # Get the observer's location
        observer = self.satellite_data.ts.from_datetime(timestamp)
        location = wgs84.latlon(self.latitude, self.longitude, self.altitude)  # alt in meters
        
        # Get antenna parameters from serving satellite data
        # try:
        #     # Read the combined serving satellite file
        #     combined_file = os.path.join(self.output_dir, 'combined_serving_satellite.csv')
        #     if os.path.exists(combined_file):
        #         df = pd.read_csv(combined_file)
        #         df['Timestamp'] = pd.to_datetime(df['Timestamp'])
                
        #         # Find the closest timestamp
        #         df['time_diff'] = abs(df['Timestamp'] - timestamp)
        #         closest_row = df.loc[df['time_diff'].idxmin()]
                
        #         # Get antenna parameters
        #         # tilt_angle = closest_row['tiltAngleDeg']
        #         # boresight_azimuth = closest_row['boresightAzimuthDeg']
                
        #         # Get hardware version to determine FOV
        #         # hardware_version = closest_row['hardwareVersion']
        #         # fov_radius = self.get_fov_degree_from_model(hardware_version) / 2.0
        #     else:
        #         logger.warning(f"Combined serving satellite file not found: {combined_file}")
        #         return visible_satellites
        # except Exception as e:
        #     logger.error(f"Error reading antenna parameters: {e}")
        #     return visible_satellites
        
        # Pre-allocate arrays for satellite positions
        n_satellites = len(tle_data)
        elevations = np.zeros(n_satellites)
        azimuths = np.zeros(n_satellites)
        satellite_names = []
        
        # Compute all satellite positions
        for i, (_, row) in enumerate(tle_data.iterrows()):
            try:
                # Create satellite object from TLE
                sat = EarthSatellite(row['tle_line1'], row['tle_line2'], row['satellite_name'], self.satellite_data.ts)
                
                # Compute satellite position relative to observer
                difference = sat - location
                topocentric = difference.at(observer)
                
                # Get altitude and azimuth
                alt, az, _ = topocentric.altaz()
                
                # Store angles and satellite info
                elevations[i] = alt.degrees
                azimuths[i] = az.degrees
                satellite_names.append(row['satellite_name'])
            except Exception as e:
                logger.error(f"Error computing position for {row['satellite_name']}: {e}")
                continue
        
        # Check visibility for each satellite
        for i, (elevation, azimuth, sat_name) in enumerate(zip(elevations, azimuths, satellite_names)):
            if elevation > 20:
                row = tle_data.iloc[i]
                visible_satellites.append({
                    "satellite": sat_name,
                    "sat_elevation_deg": elevation,
                    "sat_azimuth_deg": azimuth,
                    "tle_line1": row['tle_line1'],
                    "tle_line2": row['tle_line2']
                })
        
        return visible_satellites