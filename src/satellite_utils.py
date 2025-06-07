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

def is_valid_satellite(alt, az, tilt_deg, rotation_deg):
    """
    Check if a satellite is within the valid field of view.

    Args:
        alt (float): Satellite altitude in degrees
        az (float): Satellite azimuth in degrees
        tilt_deg (float): Dish tilt angle in degrees
        rotation_deg (float): Dish rotation angle in degrees

    Returns:
        bool: True if satellite is within valid FOV, False otherwise
    """
    if alt is None or az is None:
        logger.debug("Invalid satellite position (alt or az is None)")
        return False

    # Convert satellite position to Cartesian coordinates
    # r is distance from center (90 - alt to match the polar plot)
    r = 90 - alt
    theta = math.radians(az)
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    
    # Rotate point back to FOV frame
    angle_rad = math.radians(-rotation_deg)
    x_rot = x * math.cos(angle_rad) - y * math.sin(angle_rad)
    y_rot = x * math.sin(angle_rad) + y * math.cos(angle_rad)

    # Translate to ellipse center
    dx = x_rot - tilt_deg
    dy = y_rot

    # Check if inside ellipse equation
    base_radius = 50  # Base radius of the FOV
    x_radius = base_radius
    y_radius = math.sqrt(base_radius**2 - tilt_deg**2)
    
    # Ellipse equation: (x/a)^2 + (y/b)^2 <= 1
    inside = (dx**2 / x_radius**2) + (dy**2 / y_radius**2) <= 1
    
    if inside:
        logger.debug(f"Satellite at alt={alt:.2f}°, az={az:.2f}° is within FOV (tilt={tilt_deg}°, rot={rotation_deg}°)")
    else:
        logger.debug(f"Satellite at alt={alt:.2f}°, az={az:.2f}° is outside FOV (tilt={tilt_deg}°, rot={rotation_deg}°)")
    
    return inside 