import pandas as pd
import numpy as np
import pandas as pd
from skyfield.api import load, EarthSatellite, wgs84
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans

def process_data_with_altitude(directory: str):
    base_path = directory

    # Load required CSVs
    df_ping = pd.read_csv(f"{base_path}/combined_ping_data.csv")
    print(f"Loaded combined_ping_data.csv into `df_ping`: shape = {df_ping.shape}")

    df_conn_periods = pd.read_csv(f"{base_path}/connection_periods.csv")
    print(f"Loaded connection_periods.csv into `df_conn_periods`: shape = {df_conn_periods.shape}")

    # Ensure ping data is sorted by timestamp
    df_ping = df_ping.sort_values('Timestamp').reset_index(drop=True)

    # We'll store the result here
    avg_latencies = []

    # Pointer to avoid re-scanning the ping DataFrame from the start
    ping_index = 0
    n_pings = len(df_ping)

    for _, row in tqdm(df_conn_periods.iterrows(), total=len(df_conn_periods)):
        start_time = row['Start_Time']
        end_time = row['End_Time']

        # Move the ping_index forward until we're inside or past the start_time
        while ping_index < n_pings and df_ping.at[ping_index, 'Timestamp'] < start_time:
            ping_index += 1

        # Use a secondary index to move until we pass end_time
        j = ping_index
        while j < n_pings and df_ping.at[j, 'Timestamp'] <= end_time:
            j += 1

        # Compute mean latency in the window
        if j > ping_index:
            mean_latency = df_ping.loc[ping_index:j, 'Latency_ms'].mean()
        else:
            mean_latency = None

        avg_latencies.append(mean_latency)

    # Add result to DataFrame
    df_conn_periods['avg_ping_latency_ms'] = avg_latencies

    # Ensure datetime column is parsed
    df_conn_periods['Start_Time'] = pd.to_datetime(df_conn_periods['Start_Time'])

    # Skyfield time and ephemeris setup
    ts = load.timescale()

    # Create a function to compute altitude using TLE and Start_Time
    def compute_altitude(row):
        try:
            satellite = EarthSatellite(row['TLE_Line1'], row['TLE_Line2'], row['Satellite'], ts)
            t = ts.from_datetime(row['Start_Time'].to_pydatetime())
            geocentric = satellite.at(t)
            subpoint = wgs84.subpoint(geocentric)
            return subpoint.elevation.km
        except Exception as e:
            return None

    # Optional: tqdm for progress bar
    tqdm.pandas()

    # Compute and store altitude
    df_conn_periods['Altitude_km'] = df_conn_periods.progress_apply(compute_altitude, axis=1)

    # Convert numeric columns to appropriate types
    df_conn_periods['avg_ping_latency_ms'] = pd.to_numeric(df_conn_periods['avg_ping_latency_ms'], errors='coerce')
    df_conn_periods['Altitude_km'] = pd.to_numeric(df_conn_periods['Altitude_km'], errors='coerce')

    # Save the processed DataFrame with altitude data
    output_file = f"{base_path}/connection_periods_with_altitude.csv"
    df_conn_periods.to_csv(output_file, index=False)
    print(f"Saved connection periods with altitude data to {output_file}")
    
    return df_conn_periods

def plot_altitude_vs_latency(df):
    """Plot satellite altitude vs average ping latency."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='avg_ping_latency_ms', y='Altitude_km')
    plt.title('Satellite Altitude vs Average Ping Latency')
    plt.xlabel('Average Ping Latency (ms)')
    plt.ylabel('Altitude (km)')
    plt.show()

def classify_and_plot_altitude_vs_latency_kmeans(df_conn_periods, n_clusters=3):
    df_filtered = df_conn_periods[['Altitude_km', 'avg_ping_latency_ms']].dropna().copy()

    altitudes = df_filtered['Altitude_km'].values.reshape(-1, 1)

    # KMeans for clearer altitude-based clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    labels = kmeans.fit_predict(altitudes)

    # Sort by cluster centers and relabel
    sorted_centers = np.argsort(kmeans.cluster_centers_.flatten())
    label_map = {old: new for new, old in enumerate(sorted_centers)}
    df_filtered['altitude_band'] = [label_map[l] for l in labels]

    band_names = {0: 'Low Altitude', 1: 'Mid Altitude', 2: 'High Altitude'}
    df_filtered['altitude_band_label'] = df_filtered['altitude_band'].map(band_names)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_filtered,
        x='avg_ping_latency_ms',
        y='Altitude_km',
        hue='altitude_band_label',
        palette='Set2'
    )
    plt.title('Altitude vs Average Ping Latency (KMeans Altitude Bands)')
    plt.xlabel('Average Ping Latency (ms)')
    plt.ylabel('Altitude (km)')
    plt.grid(True)
    plt.legend(title='Altitude Band')
    plt.tight_layout()
    plt.show()

    df_conn_periods.loc[df_filtered.index, 'altitude_band'] = df_filtered['altitude_band']
    df_conn_periods.loc[df_filtered.index, 'altitude_band_label'] = df_filtered['altitude_band_label']

    return df_conn_periods

def plot_altitude_distribution(df):
    """Plot the distribution of satellite altitudes."""
    plt.figure(figsize=(12, 6))
    
    # Create a combined histogram and KDE plot
    sns.histplot(data=df, x='Altitude_km', bins=30, kde=True)
    
    # Add vertical lines for mean and median
    mean_alt = df['Altitude_km'].mean()
    median_alt = df['Altitude_km'].median()
    
    plt.axvline(mean_alt, color='red', linestyle='--', label=f'Mean: {mean_alt:.1f} km')
    plt.axvline(median_alt, color='green', linestyle='--', label=f'Median: {median_alt:.1f} km')
    
    plt.title('Distribution of Satellite Altitudes')
    plt.xlabel('Altitude (km)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Print some statistics
    print("\nAltitude Statistics:")
    print(f"Mean altitude: {mean_alt:.2f} km")
    print(f"Median altitude: {median_alt:.2f} km")
    print(f"Standard deviation: {df['Altitude_km'].std():.2f} km")
    print(f"Min altitude: {df['Altitude_km'].min():.2f} km")
    print(f"Max altitude: {df['Altitude_km'].max():.2f} km")

def main():
    base_path = 'output'
    
    # Check if processed file exists
    if os.path.exists(f"{base_path}/connection_periods_with_altitude.csv"):
        print("Loading existing processed data...")
        df_conn_periods = pd.read_csv(f"{base_path}/connection_periods_with_altitude.csv")
        # Convert numeric columns to appropriate types
        df_conn_periods['avg_ping_latency_ms'] = pd.to_numeric(df_conn_periods['avg_ping_latency_ms'], errors='coerce')
        df_conn_periods['Altitude_km'] = pd.to_numeric(df_conn_periods['Altitude_km'], errors='coerce')
    else:
        print("Processing data...")
        df_conn_periods = process_data_with_altitude(base_path)
    
    # Plot the data
    plot_altitude_distribution(df_conn_periods)
    classify_and_plot_altitude_vs_latency_kmeans(df_conn_periods)

if __name__ == "__main__":
    main()
