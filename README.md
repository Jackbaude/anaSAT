# Satellite Analysis CLI

This tool analyzes and visualizes satellite connection data, including trajectories and latency information.

## Prerequisites

- Python 3.x
- Required Python packages (install using `pip install -r requirements.txt`):
  - pandas
  - matplotlib
  - cartopy
  - skyfield

## Running the CLI

The main script `plot_satellite_cli.py` can be run with the following command:

```bash
python plot_satellite_cli.py <timestamp> [options]
```

### Required Arguments

- `<timestamp>`: The timestamp of the data to analyze (format: YYYY-MM-DD-HH-MM-SS)
  Example: `2025-04-23-11-00-00`

### Optional Arguments

- `--lat LATITUDE`: Observer's latitude (default: None)
- `--lon LONGITUDE`: Observer's longitude (default: None)
- `--alt ALTITUDE`: Observer's altitude in meters (default: None)
- `--connections`: Plot satellite connections
- `--latency`: Plot latency data
- `--window`: Plot 2-minute window view
- `--output-dir OUTPUT_DIR`: Directory to save plots (default: timestamp_figures)

### Examples

1. Basic usage with all plots:
```bash
python plot_satellite_cli.py 2025-04-23-11-00-00 --connections --latency --window
```

2. With observer location:
```bash
python plot_satellite_cli.py 2025-04-23-11-00-00 --lat 11.1 --lon -11.1 --alt 111.11--connections --latency --window
```

3. With custom output directory:
```bash
python plot_satellite_cli.py 2025-04-23-11-00-00 --connections --latency --window --output-dir my_plots
```

## Output

The script generates the following plots in the specified output directory:
- Satellite connections over time
- Latency measurements
- 2-minute detailed window view
- Satellite trajectories (if observer location is provided)

Each plot is saved as a separate PNG file with the timestamp in the filename.
