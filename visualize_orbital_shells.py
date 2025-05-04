#!/usr/bin/env python3
"""
Script to visualize orbital shell classifications with interactive plots.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import argparse
import os

def load_data(classification_file: str, statistics_file: str) -> tuple:
    """Load the classification results and statistics."""
    print("Loading data...")
    df = pd.read_csv(classification_file)
    stats = pd.read_csv(statistics_file)
    
    # Convert timestamps to datetime
    df['Start_Time'] = pd.to_datetime(df['Start_Time'])
    df['End_Time'] = pd.to_datetime(df['End_Time'])
    
    return df, stats

def create_3d_distance_plot(df: pd.DataFrame, output_dir: str):
    """Create an interactive 3D scatter plot of the orbital shells."""
    print("Creating 3D distance plot...")
    
    fig = px.scatter_3d(
        df,
        x='Mean_Distance',
        y='Distance_Range',
        z='Std_Distance',
        color='Orbital_Shell',
        hover_data=['Satellite', 'Start_Time', 'End_Time'],
        title='Orbital Shell Classification in 3D Space',
        labels={
            'Mean_Distance': 'Mean Distance (km)',
            'Distance_Range': 'Distance Range (km)',
            'Std_Distance': 'Distance Std Dev (km)',
            'Orbital_Shell': 'Orbital Shell'
        }
    )
    
    fig.write_html(os.path.join(output_dir, 'orbital_shells_3d.html'))

def create_time_series_plot(df: pd.DataFrame, output_dir: str):
    """Create an interactive time series plot of satellite connections."""
    print("Creating time series plot...")
    
    # Create a figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces for each orbital shell
    for shell in sorted(df['Orbital_Shell'].unique()):
        shell_data = df[df['Orbital_Shell'] == shell]
        
        fig.add_trace(
            go.Scatter(
                x=shell_data['Start_Time'],
                y=shell_data['Mean_Distance'],
                mode='markers',
                name=f'Shell {shell}',
                marker=dict(
                    size=8,
                    symbol='circle',
                    line=dict(width=2)
                ),
                text=shell_data['Satellite'],
                hovertemplate="<br>".join([
                    "Satellite: %{text}",
                    "Time: %{x}",
                    "Mean Distance: %{y:.2f} km",
                    "Shell: %{name}"
                ])
            ),
            secondary_y=False
        )
        
        # Add error bars for distance range
        fig.add_trace(
            go.Scatter(
                x=shell_data['Start_Time'],
                y=shell_data['Mean_Distance'] + shell_data['Distance_Range']/2,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=shell_data['Start_Time'],
                y=shell_data['Mean_Distance'] - shell_data['Distance_Range']/2,
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                showlegend=False,
                hoverinfo='skip'
            ),
            secondary_y=False
        )
    
    # Update layout
    fig.update_layout(
        title='Satellite Connections Over Time by Orbital Shell',
        xaxis_title='Time',
        yaxis_title='Distance (km)',
        hovermode='closest'
    )
    
    fig.write_html(os.path.join(output_dir, 'orbital_shells_time_series.html'))

def create_distance_distribution_plots(df: pd.DataFrame, output_dir: str):
    """Create distribution plots for each orbital shell."""
    print("Creating distance distribution plots...")
    
    # Create violin plots
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=df, x='Orbital_Shell', y='Mean_Distance')
    plt.title('Distance Distribution by Orbital Shell')
    plt.xlabel('Orbital Shell')
    plt.ylabel('Mean Distance (km)')
    plt.savefig(os.path.join(output_dir, 'shell_distance_violin.png'))
    plt.close()
    
    # Create box plots with swarm plot overlay
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Orbital_Shell', y='Mean_Distance')
    sns.swarmplot(data=df, x='Orbital_Shell', y='Mean_Distance', color='black', alpha=0.5)
    plt.title('Distance Distribution by Orbital Shell')
    plt.xlabel('Orbital Shell')
    plt.ylabel('Mean Distance (km)')
    plt.savefig(os.path.join(output_dir, 'shell_distance_box_swarm.png'))
    plt.close()

def create_shell_statistics_plot(stats: pd.DataFrame, output_dir: str):
    """Create a bar plot showing statistics for each orbital shell."""
    print("Creating shell statistics plot...")
    
    # Create a figure with subplots
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        'Mean Distance by Shell',
        'Distance Range by Shell',
        'Number of Satellites by Shell',
        'Distance Variability by Shell'
    ))
    
    # Add traces for each statistic
    fig.add_trace(
        go.Bar(
            x=stats.index,
            y=stats['Mean_Distance'],
            name='Mean Distance',
            error_y=dict(
                type='data',
                array=stats['Std_Distance'],
                visible=True
            )
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=stats.index,
            y=stats['Distance_Range'],
            name='Distance Range'
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(
            x=stats.index,
            y=stats['Satellite'],
            name='Number of Satellites'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=stats.index,
            y=stats['Distance_Variance'],
            name='Distance Variance'
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title='Orbital Shell Statistics',
        showlegend=False,
        height=800
    )
    
    fig.write_html(os.path.join(output_dir, 'shell_statistics.html'))

def create_satellite_trajectory_plot(df: pd.DataFrame, output_dir: str):
    """Create a plot showing satellite trajectories through different shells."""
    print("Creating satellite trajectory plot...")
    
    # Group by satellite and sort by time
    satellite_groups = df.groupby('Satellite')
    
    fig = go.Figure()
    
    for satellite, group in satellite_groups:
        # Sort by time
        group = group.sort_values('Start_Time')
        
        # Add trace for this satellite
        fig.add_trace(
            go.Scatter(
                x=group['Start_Time'],
                y=group['Mean_Distance'],
                mode='lines+markers',
                name=satellite,
                line=dict(
                    color=px.colors.qualitative.Set1[group['Orbital_Shell'].iloc[0] % len(px.colors.qualitative.Set1)]
                ),
                marker=dict(
                    size=8,
                    symbol='circle'
                ),
                text=[f'Shell: {shell}' for shell in group['Orbital_Shell']],
                hovertemplate="<br>".join([
                    "Satellite: %{name}",
                    "Time: %{x}",
                    "Mean Distance: %{y:.2f} km",
                    "Shell: %{text}"
                ])
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Satellite Trajectories Through Orbital Shells',
        xaxis_title='Time',
        yaxis_title='Mean Distance (km)',
        hovermode='closest'
    )
    
    fig.write_html(os.path.join(output_dir, 'satellite_trajectories.html'))

def main():
    parser = argparse.ArgumentParser(description='Visualize orbital shell classifications')
    parser.add_argument('--classification', required=True, help='Path to orbital shell classification CSV file')
    parser.add_argument('--statistics', required=True, help='Path to orbital shell statistics CSV file')
    parser.add_argument('--output_dir', default='visualizations', help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    df, stats = load_data(args.classification, args.statistics)
    
    # Create visualizations
    create_3d_distance_plot(df, args.output_dir)
    create_time_series_plot(df, args.output_dir)
    create_distance_distribution_plots(df, args.output_dir)
    create_shell_statistics_plot(stats, args.output_dir)
    create_satellite_trajectory_plot(df, args.output_dir)
    
    print(f"Visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main() 