import math
from datetime import datetime, timedelta
from skyfield.api import EarthSatellite, load, wgs84
from logger import logger

def get_satellite_position(tle_data, timestamp, latitude, longitude, altitude, ts):
    """
    Compute satellite position (alt, az) using TLE and Skyfield.

    Args:
        tle_data (dict): Dictionary containing TLE data
        timestamp (datetime): Timestamp for position calculation
        latitude (float): Observer's latitude
        longitude (float): Observer's longitude
        altitude (float): Observer's altitude in meters
        ts: Skyfield timescale object

    Returns:
        tuple: (altitude_degrees, azimuth_degrees)
    """
    try:
        sat = EarthSatellite(tle_data['tle_line1'], tle_data['tle_line2'], 
                           tle_data.get('satellite_name', 'Unknown'), ts)
        t = ts.from_datetime(timestamp)
        observer = wgs84.latlon(latitude, longitude, altitude)
        
        difference = sat - observer
        topocentric = difference.at(t)
        alt, az, _ = topocentric.altaz()
        
        logger.debug(f"Satellite {sat.name} position at {timestamp}: alt={alt.degrees:.2f}°, az={az.degrees:.2f}°")
        return alt.degrees, az.degrees
    except Exception as e:
        logger.error(f"Error calculating satellite position for {tle_data.get('satellite_name', 'Unknown')} at {timestamp}: {e}")
        return None, None 