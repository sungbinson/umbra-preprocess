# UMBRA SAR Data Preprocessing

Tools for preprocessing UMBRA SAR satellite data for ship detection in water areas.

## Features

- Download and process UMBRA SAR data
- Filter images by water coverage using OpenStreetMap data
- Process whole images or patches
- Save as GeoTIFF or PNG formats

## Requirements

- Python 3.6+
- Packages: numpy, pillow, boto3, geopandas, rasterio, shapely, pyproj

## Setup

1. Clone repository:
   ```
   git clone https://github.com/sungbinson/umbra-preprocessing.git
   cd umbra-preprocessing
   ```

2. Install packages:
   ```
   pip install numpy pillow boto3 geopandas rasterio shapely pyproj
   ```

3. Get water polygon data:
   - Download from [OpenStreetMap](https://osmdata.openstreetmap.de/data/water-polygons.html) (Shapefile, WGS84)
   - Extract to `water-polygon` directory

## Usage

### Basic
```bash
python preprocess/download_umbra.py --water-polygon water-polygon/water_polygons.shp --output-dir /path/to/output
```

### Main Options
- `--water-polygon`: Path to water polygon shapefile
- `--output-dir`: Output directory
- `--min-water-ratio`: Minimum water ratio (0.0-1.0, default: 0.7)
- `--max-water-ratio`: Maximum water ratio (0.0-1.0, default: 0.9)
- `--max-files`: Maximum files to process (-1 for all)
- `--save-as-png`: Save as PNG instead of GeoTIFF
- `--use-patches`: Process in patches
- `--patch-size`: Patch size in pixels (default: 512)

For more options, run with `--help`.

## Structure

- `preprocess/`: Main processing scripts
- `water-polygon/`: Directory for water polygon data

## Data Sources

- UMBRA SAR Data: From [UMBRA open data catalog](http://umbra-open-data-catalog.s3-website.us-west-2.amazonaws.com/) (AWS S3)
- Water Polygons: From [OpenStreetMap](https://osmdata.openstreetmap.de/data/water-polygons.html) (ODbL license)

## Image Comparison GUI

A GUI tool to visually compare paired mask and raster images, display metadata coordinates, and manage images easily.

### Usage
```bash
python preprocess/umbra_gui.py
```

### Features
- Navigate images using left/right arrow keys.
- Delete unwanted image pairs.
- Display and copy coordinates from metadata files.

### Requirements
- Python packages: tkinter, pillow, json