import pandas as pd
from process_satellite_data import SatelliteDataProcessor
from datetime import datetime, timedelta
import argparse
from skyfield.api import load, wgs84, EarthSatellite, Topos
import logging

import math as math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class SatsInView:
    def __init__(self, start_time: datetime, end_time: datetime, latitude: float, longitude: float, altitude: float = 0, data_dir: str = "analysis_results"):
        self.satellite_data_processor = SatelliteDataProcessor()
        self.start_time = start_time
        self.end_time = end_time
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.ts = load.timescale()
        self.data_dir = data_dir
        self.observer_location = wgs84.latlon(latitude, longitude, altitude)
        
        # Timestamp,Y,X,Elevation,Azimuth,Connected_Satellite,Distance,TLE_Line1,TLE_Line2,TLE_Timestamp,Altitude_km,Latency_ms,sinr,popPingLatencyMs,downlinkThroughputBps,uplinkThroughputBps,tiltAngleDeg,boresightAzimuthDeg,boresightElevationDeg,attitudeEstimationState,attitudeUncertaintyDeg,desiredBoresightAzimuthDeg,desiredBoresightElevationDeg
        self.combined_serving = pd.read_csv(f"{self.data_dir}/combined_serving_satellite.csv")

        print((self.combined_serving.head()))
        logger.info(f"Initialized DataLoader with observer location: lat={latitude}, lon={longitude}, alt={altitude}")
        

    def generate_handover_timestamps(self):
        """
        Generate timestamps at 12, 27, 42, and 57 seconds of each minute between start and end time.
        """
        current_time = self.start_time
        handover_seconds = [12, 27, 42, 57]
        timestamps = []

        while current_time <= self.end_time:
            for sec in handover_seconds:
                ts = current_time.replace(second=sec, microsecond=0)
                if ts <= self.end_time:
                    timestamps.append(ts)
            current_time += timedelta(minutes=1)

        logger.info(f"Generated {len(timestamps)} handover timestamps")
        return timestamps
    
    # Function to compute satellite position (alt, az) using TLE and Skyfield
    def get_satellite_position(tle_data, observer, timestamp):
        ts = load.timescale()
        t = ts.utc(*timestamp)  # UTC time as tuple (year, month, day, hour, minute, second)
        
        satellite = load.satellite(tle_data[1], tle_data[2])
        topocentric = satellite.at(t).from_altaz(alt_degrees=None, az_degrees=None)
        
        alt, az, _ = topocentric.altaz()
        return alt.degrees, az.degrees
    
    def is_valid_satellite(alt, az, tilt_deg, rotation_deg, fov_azimuth, fov_elevation):
        # Rotate azimuth based on dish rotation
        adjusted_azimuth = (az + rotation_deg) % 360
        
        # Check if satellite is within FOV azimuth range
        azimuth_range = 90  # Width of the FOV in degrees
        if adjusted_azimuth < (fov_azimuth - azimuth_range) or adjusted_azimuth > (fov_azimuth + azimuth_range):
            return False
        
        # Check if satellite is within FOV elevation range
        fov_max_elevation = 90 - tilt_deg  # Max elevation based on tilt
        fov_min_elevation = 0  # Minimum elevation (adjust as necessary)
        if alt < fov_min_elevation or alt > fov_max_elevation:
            return False
        
        return True
    
    def check_satellites_in_fov(self, timestamp, tle_data, tilt_deg, rotation_deg, fov_azimuth, fov_elevation):
        observer = Topos(latitude_degrees=self.lat, longitude_degrees=self.lon, elevation_m=self.altitude)
        
        valid_satellites = []
        
        for satellite_name, line1, line2 in satellites:
            tle_data = (satellite_name, line1, line2)
            # Get the satellite's position (altitude, azimuth)
            alt, az = self.get_satellite_position(tle_data, observer, timestamp)
            
            # Check if the satellite is within the FOV
            if self.is_valid_satellite(alt, az, tilt_deg, rotation_deg, fov_azimuth, fov_elevation):
                valid_satellites.append({
                    'satellite_name': satellite_name,
                    'timestamp': timestamp,
                    'altitude': alt,
                    'azimuth': az
                })
        
        return valid_satellites
        
    def satellite_visible(self, tle_data, timestamp):
        """
        Check if a satellite is visible (>20° elevation) from the observer at the given timestamp.
        """
        try:
            if not tle_data or 'line1' not in tle_data or 'line2' not in tle_data:
                logger.warning(f"Invalid TLE data for timestamp {timestamp}")
                return False

            # Create satellite object from TLE data
            sat = EarthSatellite(tle_data['line1'], tle_data['line2'], tle_data.get('name', 'Unknown'), self.ts)
            
            # Convert timestamp to Skyfield time
            t = self.ts.utc(timestamp)
            
            # Calculate satellite position relative to observer
            difference = sat - self.observer_location
            topocentric = difference.at(t)
            
            # Get altitude and azimuth
            alt, az, _ = topocentric.altaz()
            
            # Log visibility details
            logger.debug(f"Satellite {tle_data.get('name', 'Unknown')} at {timestamp}: "
                        f"altitude={alt.degrees:.2f}°, azimuth={az.degrees:.2f}°")
            
            return alt.degrees > 20
        except Exception as e:
            logger.error(f"Visibility check failed at {timestamp}: {str(e)}")
            return False

    def compute_visibility(self, frame_type=2):
        """
        For each timestamp, compute visible satellites.
        Load TLE data at hourly intervals and check visibility at handover times.
        """
        timestamps = self.generate_handover_timestamps()
        visibility_results = {}
        
        # First, load TLE data for each hour
        current_hour = self.start_time.replace(minute=0, second=0, microsecond=0)
        tle_data_by_hour = {}
        
        while current_hour <= self.end_time:
            tle_data = self.satellite_data_processor.load_tle_data(current_hour)
            if tle_data:
                logger.info(f"Loaded TLE data for {current_hour}: {len(tle_data)} satellites")
                tle_data_by_hour[current_hour] = tle_data
            else:
                logger.warning(f"No TLE data found for {current_hour}")
            current_hour += timedelta(hours=1)
        
        if not tle_data_by_hour:
            logger.error("No TLE data loaded for any hour in the time range")
            return visibility_results
        
        # Then check visibility at each handover timestamp
        for ts in timestamps:
            hour_key = ts.replace(minute=0, second=0, microsecond=0)
            if hour_key not in tle_data_by_hour:
                logger.warning(f"No TLE data available for hour {hour_key}")
                continue
                
            hour_data = tle_data_by_hour[hour_key]
            visible_sats = []
            
            for sat_name, tle_data in hour_data.items():
                if self.satellite_visible(tle_data, ts):
                    visible_sats.append(sat_name)            
                       
            visibility_results[ts] = visible_sats
            logger.info(f"{ts}: {len(visible_sats)} satellites visible")

        return visibility_results
    

def main():
    parser = argparse.ArgumentParser(description="Compute visible satellites at handover timestamps")
    parser.add_argument('--start_time', required=True, help='Start time (YYYY-MM-DD-HH-MM-SS)')
    parser.add_argument('--end_time', required=True, help='End time (YYYY-MM-DD-HH-MM-SS)')
    parser.add_argument('--lat', type=float, required=True, help='Observer latitude')
    parser.add_argument('--lon', type=float, required=True, help='Observer longitude')
    parser.add_argument('--alt', type=float, default=0.0, help='Observer altitude in meters')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    start = SatelliteDataProcessor.parse_timestamp(args.start_time)
    end = SatelliteDataProcessor.parse_timestamp(args.end_time)

    logger.info(f"Processing visibility from {start} to {end}")
    loader = SatsInView(start, end, args.lat, args.lon, args.alt)
    vis = loader.compute_visibility()

    # Save to CSV
    df = pd.DataFrame([
        {'timestamp': ts, 'visible_satellites': ', '.join(sats)}
        for ts, sats in vis.items()
    ])
    
    
    output_file = "satellite_visibility.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Saved results to {output_file}")
    loader.append_distance_and_alt(df)
    logger.info("Appended distance and altitude to DataFrame and saved to CSV")
    output_file = "satellite_visibility_with_distance_altitude.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Saved results with distance and altitude to {output_file}")
if __name__ == "__main__":
    main()
