#!/usr/bin/env python3
"""
Class for plotting satellite and latency data.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import logging
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import timedelta
from skyfield.api import load, wgs84, EarthSatellite

logger = logging.getLogger(__name__)

class SatellitePlotter:
    def __init__(self, analysis, output_dir):
        """Initialize the SatellitePlotter with a SatelliteAnalysis instance."""
        self.analysis = analysis
        self.output_dir = output_dir
        
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")

    def create_satellite_colors(self, satellites):
        """Create a color map for satellites."""
        color_cycle = plt.cm.tab20(np.linspace(0, 1, len(satellites)))
        return {sat: color for sat, color in zip(satellites, color_cycle)}

    def plot_satellite_connections(self, ax):
        """Plot satellite connection periods."""
        logger.info("Plotting satellite connections")
        
        # Get unique satellites in order of first appearance
        satellites = []
        for period in self.analysis.periods:
            if period['satellite'] not in satellites:
                satellites.append(period['satellite'])
        
        # Track which satellites have connected before
        connected_before = set()
        
        # Plot each connection period
        for period in self.analysis.periods:
            y_pos = satellites.index(period['satellite'])
            color = 'red' if period['satellite'] in connected_before else 'blue'
            connected_before.add(period['satellite'])
            
            ax.hlines(
                y=y_pos,
                xmin=period['start_time'],
                xmax=period['end_time'],
                linewidth=10,
                color=color,
                label=f"{period['satellite']} ({period['duration']:.1f}s)"
            )
        
        # Get the start and end times of the data
        start_time = self.analysis.df['timestamp'].min()
        end_time = self.analysis.df['timestamp'].max()
        
        # Get the start and end minutes
        start_minute = start_time.replace(second=0, microsecond=0)
        end_minute = end_time.replace(second=0, microsecond=0) + pd.Timedelta(minutes=1)
        
        # Define the handover seconds within each minute
        handover_seconds = [57, 12, 27, 42]
        
        # Mark regular handovers for all minutes in the data range
        current_minute = start_minute
        while current_minute <= end_minute:
            for second in handover_seconds:
                handover_time = current_minute + pd.Timedelta(seconds=second)
                if start_time <= handover_time <= end_time:
                    ax.axvline(x=handover_time, color='red', linestyle='--', alpha=0.7)
            current_minute += pd.Timedelta(minutes=1)
        
        # Customize the plot
        ax.set_title('Satellite Connection Periods', pad=20)
        ax.set_xlabel('Time')
        ax.set_ylabel('Satellite')
        ax.set_yticks(range(len(satellites)))
        ax.set_yticklabels(satellites)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(plt.matplotlib.dates.SecondLocator(interval=30))
        plt.setp(ax.get_xticklabels(), rotation=45, fontsize=10)
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        legend_elements = [
            plt.Line2D([0], [0], color='blue', lw=10, label='First Connection'),
            plt.Line2D([0], [0], color='red', lw=10, label='Subsequent Connection'),
            plt.Line2D([0], [0], color='red', linestyle='--', label='Regular Handover')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        return satellites

    def plot_latency(self, ax):
        """Plot latency data with handover markers."""
        logger.info("Plotting latency data")
        
        # Plot latency
        ax.plot(self.analysis.latency_series.index, self.analysis.latency_series.values, 'g-', linewidth=2)
        
        # Mark satellite changes
        for change_time in self.analysis.satellite_change_times:
            ax.axvline(x=change_time, color='blue', linestyle=':', alpha=0.7)
        
        # Get the start and end times of the data
        start_time = self.analysis.latency_series.index[0]
        end_time = self.analysis.latency_series.index[-1]
        
        # Get the start and end minutes
        start_minute = start_time.replace(second=0, microsecond=0)
        end_minute = end_time.replace(second=0, microsecond=0) + pd.Timedelta(minutes=1)
        
        # Define the handover seconds within each minute
        handover_seconds = [57, 12, 27, 42]
        
        # Mark regular handovers for all minutes in the data range
        current_minute = start_minute
        while current_minute <= end_minute:
            for second in handover_seconds:
                handover_time = current_minute + pd.Timedelta(seconds=second)
                if start_time <= handover_time <= end_time:
                    ax.axvline(x=handover_time, color='red', linestyle='--', alpha=0.7)
            current_minute += pd.Timedelta(minutes=1)
        
        # Customize the plot
        ax.set_title('Latency', pad=20)
        ax.set_xlabel('Time')
        ax.set_ylabel('Latency (ms)')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(plt.matplotlib.dates.SecondLocator(interval=30))
        plt.setp(ax.get_xticklabels(), rotation=45, fontsize=10)
        
        # Add grid and legend
        ax.grid(True, alpha=0.3)
        legend_elements = [
            plt.Line2D([0], [0], color='green', label='Latency'),
            plt.Line2D([0], [0], color='blue', linestyle=':', label='Satellite Change'),
            plt.Line2D([0], [0], color='red', linestyle='--', label='Regular Handover')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

    def plot_satellite_trajectories(self, ax):
        """Plot satellite trajectories with latency information."""
        if not self.analysis.observer_lat or not self.analysis.observer_lon:
            logger.warning("Observer location not set. Cannot plot trajectories.")
            return
        
        if not self.analysis.tle_data or not self.analysis.satellite_objects:
            logger.warning("TLE data not loaded. Cannot plot trajectories.")
            return
        
        logger.info("Plotting satellite trajectories")
        
        # Set up the map with a reasonable extent
        offset = 10  # degrees
        ax.set_extent([
            self.analysis.observer_lon - offset,
            self.analysis.observer_lon + offset,
            self.analysis.observer_lat - offset,
            self.analysis.observer_lat + offset
        ], crs=ccrs.PlateCarree())
        
        # Add map features with white background
        ax.set_facecolor('white')
        ax.add_feature(cfeature.LAND, facecolor='white', edgecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.9, edgecolor='lightgray')
        ax.add_feature(cfeature.BORDERS, linewidth=0.9, edgecolor='lightgray')
        ax.add_feature(cfeature.STATES, linewidth=0.9, edgecolor='lightgray')
        
        # Plot observer location
        ax.plot(self.analysis.observer_lon, self.analysis.observer_lat,
                marker='*', color='black', markersize=15, markeredgewidth=2,
                transform=ccrs.PlateCarree(), zorder=100,
                label='Observer')
        
        # Create color map for latency
        cmap = plt.cm.RdYlGn_r  # Red (high) to Green (low)
        
        # Find global min/max latency for consistent coloring
        valid_latencies = []
        if self.analysis.latency_series is not None:
            valid_latencies = [x for x in self.analysis.latency_series.values if not np.isnan(x)]
        
        if valid_latencies:
            vmin = min(valid_latencies)
            vmax = max(valid_latencies)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
        else:
            logger.warning("No valid latency values found")
            return
        
        # Plot trajectories for each connection period
        for period in self.analysis.periods:
            sat_name = period['satellite']
            if sat_name not in self.analysis.satellite_objects:
                continue
            
            sat = self.analysis.satellite_objects[sat_name]
            start_time = period['start_time']
            end_time = period['end_time']
            
            # Calculate positions and get latencies
            positions = {
                'time': [],
                'lat': [],
                'lon': [],
                'latency': []
            }
            
            current_time = start_time
            while current_time <= end_time:
                # Get satellite position
                t = load.timescale().from_datetime(current_time)
                geocentric = sat.at(t)
                subpoint = wgs84.subpoint(geocentric)
                
                positions['time'].append(current_time)
                positions['lat'].append(subpoint.latitude.degrees)
                positions['lon'].append(subpoint.longitude.degrees)
                
                # Find closest latency measurement within 5 seconds
                closest_time = min(self.analysis.latency_series.index,
                                 key=lambda x: abs((x - current_time).total_seconds()))
                
                if abs((closest_time - current_time).total_seconds()) <= 5:
                    positions['latency'].append(self.analysis.latency_series[closest_time])
                else:
                    positions['latency'].append(np.nan)
                
                current_time += timedelta(seconds=1)
            
            if not positions['lon']:
                continue
            
            # Plot trajectory points with latency-based coloring
            valid_mask = ~np.isnan(positions['latency'])
            if np.any(valid_mask):
                points = ax.scatter(
                    np.array(positions['lon'])[valid_mask],
                    np.array(positions['lat'])[valid_mask],
                    c=np.array(positions['latency'])[valid_mask],
                    cmap=cmap,
                    norm=norm,
                    s=50,  # Point size
                    alpha=0.8,
                    transform=ccrs.PlateCarree(),
                    zorder=10
                )
            
            # Add satellite name at start of trajectory
            ax.text(positions['lon'][0], positions['lat'][0],
                   sat_name,
                   transform=ccrs.PlateCarree(),
                   fontsize=8, ha='right', va='bottom')
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.05)
        cbar.set_label('Latency (ms)', fontsize=12)
        
        # Add title
        ax.set_title('Satellite Trajectories with Latency', pad=20)

    def plot_two_minute_window(self, ax, middle_time):
        """Plot detailed 2-minute window view."""
        logger.info("Plotting 2-minute window view")
        
        # Calculate window boundaries
        window_start = middle_time - pd.Timedelta(minutes=1)
        window_end = middle_time + pd.Timedelta(minutes=1)
        
        # Get satellites in window
        window_satellites = set()
        for period in self.analysis.periods:
            if period['end_time'] >= window_start and period['start_time'] <= window_end:
                window_satellites.add(period['satellite'])
        
        # Create colors for satellites
        satellite_colors = self.create_satellite_colors(window_satellites)
        
        # Plot satellite connections
        for period in self.analysis.periods:
            if period['end_time'] >= window_start and period['start_time'] <= window_end:
                plot_start = max(period['start_time'], window_start)
                plot_end = min(period['end_time'], window_end)
                
                # Calculate average latency
                window_latency = self.analysis.latency_series[plot_start:plot_end]
                avg_latency = window_latency.mean()
                
                color = satellite_colors[period['satellite']]
                
                # Plot connection line
                ax.hlines(
                    y=avg_latency,
                    xmin=plot_start,
                    xmax=plot_end,
                    linewidth=10,
                    color=color,
                    alpha=1.0,
                    label=period['satellite']
                )
                
                # Add satellite name
                duration = (plot_end - plot_start).total_seconds()
                if duration > 5:
                    ax.text(
                        plot_start + (plot_end - plot_start)/2,
                        avg_latency,
                        period['satellite'],
                        ha='center',
                        va='center',
                        color='white',
                        fontweight='bold'
                    )
        
        # Plot latency
        window_latency = self.analysis.latency_series[window_start:window_end]
        ax.plot(window_latency.index, window_latency.values, 'g-', linewidth=2, label='Latency')
        
        # Get the start and end minutes for the window
        start_minute = window_start.replace(second=0, microsecond=0)
        end_minute = window_end.replace(second=0, microsecond=0) + pd.Timedelta(minutes=1)
        
        # Define the handover seconds within each minute
        handover_seconds = [57, 12, 27, 42]
        
        # Mark regular handovers for all minutes in the window
        current_minute = start_minute
        while current_minute <= end_minute:
            for second in handover_seconds:
                handover_time = current_minute + pd.Timedelta(seconds=second)
                if window_start <= handover_time <= window_end:
                    ax.axvline(x=handover_time, color='red', linestyle='--', alpha=0.7, label='Regular Handover')
            current_minute += pd.Timedelta(minutes=1)
        
        # Mark satellite changes
        for change_time in self.analysis.satellite_change_times:
            if window_start <= change_time <= window_end:
                ax.axvline(x=change_time, color='blue', linestyle=':', alpha=0.7, label='Satellite Change')
        
        # Customize the plot
        ax.set_title('Middle 2-Minute Window', pad=20)
        ax.set_xlabel('Time')
        ax.set_ylabel('Latency (ms)')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(plt.matplotlib.dates.SecondLocator(interval=2))
        ax.xaxis.set_minor_locator(plt.matplotlib.dates.SecondLocator(interval=1))
        plt.setp(ax.get_xticklabels(), rotation=45, fontsize=10)
        
        # Add grid
        ax.grid(True, alpha=0.3, which='major')
        ax.grid(True, alpha=0.1, which='minor', linestyle=':')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='green', label='Latency'),
            plt.Line2D([0], [0], color='blue', linestyle=':', label='Satellite Change'),
            plt.Line2D([0], [0], color='red', linestyle='--', label='Regular Handover')
        ]
        for sat, color in satellite_colors.items():
            legend_elements.append(
                plt.Line2D([0], [0], color=color, lw=10, label=f'Satellite {sat}')
            )
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))

    def create_all_plots(self):
        """Create and save all plots."""
        try:
            # Create separate figures for each plot
            fig1 = plt.figure(figsize=(25, 8))  # Satellite connections
            fig2 = plt.figure(figsize=(25, 8))  # Latency
            fig3 = plt.figure(figsize=(25, 8))  # 2-minute window
            fig4 = plt.figure(figsize=(15, 15))  # Satellite trajectories
            
            ax1 = fig1.add_subplot(111)
            ax2 = fig2.add_subplot(111)
            ax3 = fig3.add_subplot(111)
            ax4 = fig4.add_subplot(111, projection=ccrs.LambertConformal())
            
            # Plot satellite connections
            self.plot_satellite_connections(ax1)
            
            # Save satellite connections plot
            output_file1 = os.path.join(self.output_dir, f'satellite_connections_{self.analysis.timestamp}.png')
            fig1.savefig(output_file1, dpi=300, bbox_inches='tight')
            plt.close(fig1)
            logger.info(f"Saved satellite connections plot as '{output_file1}'")
            
            if self.analysis.latency_series is not None:
                # Plot latency
                self.plot_latency(ax2)
                
                # Save latency plot
                output_file2 = os.path.join(self.output_dir, f'latency_{self.analysis.timestamp}.png')
                fig2.savefig(output_file2, dpi=300, bbox_inches='tight')
                plt.close(fig2)
                logger.info(f"Saved latency plot as '{output_file2}'")
                
                # Plot and save 2-minute window
                middle_time = self.analysis.df['timestamp'].min() + (self.analysis.df['timestamp'].max() - self.analysis.df['timestamp'].min()) / 2
                self.plot_two_minute_window(ax3, middle_time)
                
                output_file3 = os.path.join(self.output_dir, f'two_minute_window_{self.analysis.timestamp}.png')
                fig3.savefig(output_file3, dpi=300, bbox_inches='tight')
                plt.close(fig3)
                logger.info(f"Saved 2-minute window plot as '{output_file3}'")
            else:
                logger.warning("No valid latency data found")
                plt.close(fig2)
                plt.close(fig3)
            
            # Plot and save satellite trajectories
            self.plot_satellite_trajectories(ax4)
            output_file4 = os.path.join(self.output_dir, f'satellite_trajectories_{self.analysis.timestamp}.png')
            fig4.savefig(output_file4, dpi=300, bbox_inches='tight')
            plt.close(fig4)
            logger.info(f"Saved satellite trajectories plot as '{output_file4}'")
            
        except Exception as e:
            logger.error(f"Error creating plots: {e}")
            raise 