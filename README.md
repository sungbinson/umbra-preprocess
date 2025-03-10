# UMBRA SAR Data Preprocessing

This repository contains tools for preprocessing UMBRA SAR (Synthetic Aperture Radar) satellite data, with a focus on ship detection in water areas.

## Overview

The project provides utilities to:
- Download and process UMBRA SAR data from the open data catalog
- Filter images based on water coverage using OpenStreetMap water polygons
- Process images either as whole files or in patches
- Save outputs as GeoTIFF or PNG formats

## Requirements

- Python 3.6+
- Required packages:
  - numpy
  - PIL
  - boto3
  - geopandas
  - rasterio
  - shapely
  - pyproj

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/sungbinson/umbra-preprocessing.git
   cd umbra-preprocessing
   ```

2. Install required packages:
   ```
   pip install numpy pillow boto3 geopandas rasterio shapely pyproj
   ```

3. Download water polygon data (if not already present in the repository):
   - Water polygon data can be downloaded from [OpenStreetMap Data](https://osmdata.openstreetmap.de/data/water-polygons.html)
   - Download the appropriate format (Shapefile) and projection (WGS84)
   - Extract the downloaded zip file to the `water-polygon` directory in this repository
   - Note: The water polygon files themselves are not included in this Git repository due to their large size

## Usage

### Basic Usage

```bash
python preprocess/download_umbra.py --water-polygon water-polygon/water_polygons.shp --output-dir /path/to/output
```

### Advanced Options

```bash
python preprocess/download_umbra.py \
  --water-polygon water-polygon/water_polygons.shp \
  --output-dir /path/to/output \
  --water-ratio 0.7 \
  --max-files 100 \
  --save-as-png \
  --use-patches \
  --patch-size 512
```

### Command Line Arguments

- `--water-polygon`: Path to water polygon shapefile (default: 'water-polygon/water_polygons.shp')
- `--output-dir`: Directory to save output files (default: '/home/ssb/dataset/UMBRACoast')
- `--water-ratio`: Minimum water ratio threshold (0.0-1.0) to save files (default: 0.7)
- `--max-files`: Maximum number of files to process (-1 for all files) (default: -1)
- `--filter-suffix`: Filter for specific file suffix (default: '_GEC.tif')
- `--save-as-png`: Save output as PNG instead of GeoTIFF
- `--s3-prefix`: S3 prefix to list (default: 'sar-data/tasks/ship_detection_testdata')
- `--downsample`: Downsample factor to reduce image size (1=original, 2=half size, 4=quarter size) (default: 1)
- `--use-patches`: Process images in patches instead of whole images
- `--patch-size`: Size of patches in pixels (square patches) (default: 512)
- `--patch-overlap`: Overlap between patches in pixels (default: 0)

## Project Structure

- `preprocess/`: Contains the main processing scripts
  - `download_umbra.py`: Main script for downloading and processing UMBRA data
  - `umbra_utils.py`: Utility functions for processing raster data
- `water-polygon/`: Directory for OpenStreetMap water polygon data
  - This directory is included in the repository, but the actual data files need to be downloaded separately
  - See the Installation section for instructions on downloading the water polygon data

## Data Sources

- **UMBRA SAR Data**: Accessed from the UMBRA open data catalog on AWS S3
- **Water Polygons**: OpenStreetMap water polygon data from [osmdata.openstreetmap.de](https://osmdata.openstreetmap.de/data/water-polygons.html)
  - These polygons represent oceans and seas, derived from OpenStreetMap ways tagged with natural=coastline
  - The data is available in WGS84 and Mercator projections
  - Licensed under the ODbL (OpenStreetMap contributors)

## License

This project is available under the MIT License. 