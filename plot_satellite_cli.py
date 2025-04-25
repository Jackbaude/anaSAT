#!/usr/bin/env python3
"""
Command-line interface for satellite data visualization.
This script provides a flexible interface to generate various plots
from satellite data, TLE information, and latency measurements.
"""

import argparse
import logging
import sys
from datetime import datetime
from process_satellite_data import SatelliteAnalysis
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate satellite data visualizations.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Generate all plots
  %(prog)s 2025-04-23-11-00-00 --lat 11.1 --lon -11.1 --alt 1111.1 --all

  # Generate only trajectory plot
  %(prog)s 2025-04-23-11-00-00 --lat 11.1 --lon -11.1 --alt 1111.1 --trajectory

  # Generate connection and latency plots
  %(prog)s 2025-04-23-11-00-00 --lat 11.1 --lon -11.1 --alt 1111.1 -connections --latency
        '''
    )

    # Required arguments
    parser.add_argument('timestamp', 
                       help='Timestamp in format YYYY-MM-DD-HH-MM-SS')

    # Optional arguments for observer location
    location_group = parser.add_argument_group('observer location arguments')
    location_group.add_argument('--lat', type=float,
                              help='Observer latitude in degrees')
    location_group.add_argument('--lon', type=float,
                              help='Observer longitude in degrees')
    location_group.add_argument('--alt', type=float,
                              help='Observer altitude in meters')

    # Plot selection arguments
    plot_group = parser.add_argument_group('plot selection arguments')
    plot_group.add_argument('--all', action='store_true',
                           help='Generate all available plots')
    plot_group.add_argument('--connections', action='store_true',
                           help='Generate satellite connection timeline plot')
    plot_group.add_argument('--latency', action='store_true',
                           help='Generate latency over time plot')
    plot_group.add_argument('--window', action='store_true',
                           help='Generate 2-minute window plot')
    plot_group.add_argument('--trajectory', action='store_true',
                           help='Generate satellite trajectory map')

    # Output options
    output_group = parser.add_argument_group('output options')
    output_group.add_argument('--output-dir',
                            help='Custom output directory (default: timestamp_figures)')
    output_group.add_argument('--dpi', type=int, default=300,
                            help='DPI for saved figures (default: 300)')

    args = parser.parse_args()

    # Validate timestamp format
    try:
        datetime.strptime(args.timestamp, '%Y-%m-%d-%H-%M-%S')
    except ValueError:
        parser.error('Invalid timestamp format. Use YYYY-MM-DD-HH-MM-SS')

    # Check if at least one plot type is selected
    if not (args.all or args.connections or args.latency or 
            args.window or args.trajectory):
        parser.error('No plots selected. Use --all or specify individual plots')

    # Check if location is provided when needed
    if args.trajectory and not (args.lat and args.lon):
        parser.error('Latitude and longitude required for trajectory plot')

    return args

def main():
    """Main function to handle the CLI interface."""
    args = parse_args()
    
    try:
        # Initialize analysis
        analysis = SatelliteAnalysis(
            args.timestamp,
            observer_lat=args.lat,
            observer_lon=args.lon,
            observer_alt=args.alt
        )
        
        # Set up file paths
        serving_file = f"data/serving_satellite_data-{args.timestamp}.csv"
        ping_file = f"data/latency/{args.timestamp[:10]}/ping-100ms-{args.timestamp}.txt"
        
        # Process the data
        analysis.process_data(serving_file, ping_file)
        
        # Create plotter instance with proper output directory
        from satellite_plotter import SatellitePlotter
        output_dir = args.output_dir if args.output_dir else os.path.join(os.getcwd(), f"{args.timestamp}_figures")
        plotter = SatellitePlotter(analysis, output_dir)
        
        # Create figure for each selected plot
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        
        # Only generate the plots that were explicitly requested
        plots_to_generate = set()
        if args.all:
            plots_to_generate = {'connections', 'latency', 'window', 'trajectory'}
        else:
            if args.connections:
                plots_to_generate.add('connections')
            if args.latency:
                plots_to_generate.add('latency')
            if args.window:
                plots_to_generate.add('window')
            if args.trajectory:
                plots_to_generate.add('trajectory')
        
        # Generate only the requested plots
        if 'connections' in plots_to_generate:
            logger.info("Generating satellite connections plot...")
            fig1 = plt.figure(figsize=(25, 8))
            ax1 = fig1.add_subplot(111)
            plotter.plot_satellite_connections(ax1)
            fig1.savefig(f"{plotter.output_dir}/satellite_connections_{args.timestamp}.png",
                        dpi=args.dpi, bbox_inches='tight')
            plt.close(fig1)
        
        if 'latency' in plots_to_generate:
            if analysis.latency_series is not None:
                logger.info("Generating latency plot...")
                fig2 = plt.figure(figsize=(25, 8))
                ax2 = fig2.add_subplot(111)
                plotter.plot_latency(ax2)
                fig2.savefig(f"{plotter.output_dir}/latency_{args.timestamp}.png",
                            dpi=args.dpi, bbox_inches='tight')
                plt.close(fig2)
            else:
                logger.warning("No latency data available")
        
        if 'window' in plots_to_generate:
            if analysis.latency_series is not None:
                logger.info("Generating 2-minute window plot...")
                fig3 = plt.figure(figsize=(25, 8))
                ax3 = fig3.add_subplot(111)
                middle_time = analysis.df['timestamp'].min() + \
                            (analysis.df['timestamp'].max() - analysis.df['timestamp'].min()) / 2
                plotter.plot_two_minute_window(ax3, middle_time)
                fig3.savefig(f"{plotter.output_dir}/two_minute_window_{args.timestamp}.png",
                            dpi=args.dpi, bbox_inches='tight')
                plt.close(fig3)
            else:
                logger.warning("No latency data available")
        
        if 'trajectory' in plots_to_generate:
            logger.info("Generating satellite trajectory plot...")
            fig4 = plt.figure(figsize=(15, 15))
            ax4 = fig4.add_subplot(111, projection=ccrs.PlateCarree())
            plotter.plot_satellite_trajectories(ax4)
            fig4.savefig(f"{plotter.output_dir}/satellite_trajectories_{args.timestamp}.png",
                        dpi=args.dpi, bbox_inches='tight')
            plt.close(fig4)
        
        logger.info("Plot generation complete")
        
    except Exception as e:
        logger.error(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 