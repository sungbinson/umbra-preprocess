import os
import argparse
import logging
import rasterio
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from umbra_utils import process_single_raster, process_raster_in_patches


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CONFIG')

# Configure rasterio to use GDAL environment variables for S3 access
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'  # For public buckets
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'  # Improves performance with S3
os.environ['CPL_VSIL_CURL_ALLOWED_EXTENSIONS'] = '.tif,.TIF,.tiff,.TIFF'



def get_s3_rasterio_path(bucket, key):
    """
    Get a rasterio-compatible path for an S3 object
    
    Args:
        bucket: S3 bucket name
        key: S3 object key
        
    Returns:
        rasterio-compatible path
    """
    # Use /vsis3/ prefix for GDAL virtual file system
    return f"/vsis3/{bucket}/{key}"


def list_umbra_data(prefix='sar-data/tasks/ship_detection_testdata', filter_suffix='_GEC.tif'):
    """
    List available Umbra SAR data
    
    Args:
        prefix: S3 prefix to list
        filter_suffix: Filter for specific file suffix (e.g., '_GEC.tif')
        
    Returns:
        List of S3 objects
    """
    try:
        # Create an anonymous/unsigned S3 client for public buckets
        s3_client = boto3.client(
            's3',
            config=Config(signature_version=UNSIGNED)
        )
        bucket = 'umbra-open-data-catalog'
        
        logger.info(f"  Listing objects in s3://{bucket}/{prefix}")
        
        # List objects in the bucket with the given prefix
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        
        # Collect all objects
        all_objects = []
        for page in pages:
            if 'Contents' in page:
                all_objects.extend(page['Contents'])
        
        # Filter for specific suffix if provided
        if filter_suffix:
            filtered_objects = [obj for obj in all_objects if obj['Key'].endswith(filter_suffix)]
            logger.info(f"  Found {len(filtered_objects)} objects with suffix '{filter_suffix}' out of {len(all_objects)} total objects")
            return filtered_objects
        
        return all_objects
    
    except Exception as e:
        logger.error(f"Error listing S3 objects: {e}")
        import traceback
        traceback.print_exc()
        return []


def process_umbra_data(
    water_polygon_path,
    output_dir='Output',
    max_files=10,
    min_water_ratio=0.05,
    max_water_ratio=0.95,
    filter_suffix='_GEC.tif',
    save_as_png=False,
    downsample_factor=1,
    s3_prefix='sar-data/tasks/ship_detection_testdata',
    use_patches=False,
    patch_size=512,
    patch_overlap=0
):
    """
    Process Umbra SAR data from the open data catalog without downloading full files
    
    Args:
        water_polygon_path: Path to water polygon shapefile
        output_dir: Directory to save output files
        max_files: Maximum number of files to process
        water_ratio_threshold: Minimum water ratio threshold to save files
        filter_suffix: Filter for specific file suffix (e.g., '_GEC.tif')
        save_as_png: Whether to save as PNG or GeoTIFF
        downsample_factor: Factor to reduce image size
        s3_prefix: S3 prefix to list
        use_patches: Whether to process images in patches
        patch_size: Size of patches in pixels
        patch_overlap: Overlap between patches in pixels
    """
    # List available data with specific suffix filter
    objects = list_umbra_data(prefix=s3_prefix, filter_suffix=filter_suffix)
    
    if not objects:
        logger.error(f"No objects found with suffix '{filter_suffix}' in the S3 bucket. Check your filter or AWS configuration.")
        return
    
    # Process up to max_files
    count = 0
    total_patches = 0
    saved_patches = 0

    objects = objects[:max_files] if max_files > 0 else objects
    
    for obj in objects:
        # Construct S3 URL for rasterio
        bucket = 'umbra-open-data-catalog'
        key = obj['Key']
        s3_url = get_s3_rasterio_path(bucket, key)
        
        # Create output paths for full image (if not using patches)
        filename = os.path.basename(key)
        output_mask_path = os.path.join(output_dir, 'masks', filename) if not use_patches else None
        output_raster_path = os.path.join(output_dir, 'rasters', filename) if not use_patches else None
        
        try:
            logger.info(f"  Processing {s3_url.split('/')[-1]}")
            
            # Open the raster directly from S3
            with rasterio.open(s3_url) as src:
                if use_patches:
                    # Process in patches
                    file_total_patches, file_saved_patches = process_raster_in_patches(
                        raster_src=src,
                        water_polygon_path=water_polygon_path,
                        output_dir=output_dir,
                        patch_size=patch_size,
                        patch_overlap=patch_overlap,
                        min_water_ratio=min_water_ratio,
                        max_water_ratio=max_water_ratio,
                        save_as_png=save_as_png,
                        use_global_normalization=False,
                    )
                    total_patches += file_total_patches
                    saved_patches += file_saved_patches
                else:
                    # Process the whole file
                    process_single_raster(
                        raster_src=src,
                        water_polygon_path=water_polygon_path,
                        output_mask_path=output_mask_path if min_water_ratio > 0 else None,
                        output_raster_path=output_raster_path if min_water_ratio > 0 else None,
                        save_as_png=save_as_png,
                        clip_to_bounds=True,
                        min_water_ratio=min_water_ratio,
                        max_water_ratio=max_water_ratio,
                        downsample_factor=downsample_factor
                    )
            
            count += 1
            if use_patches:
                logger.info(f"  Processed {count}/{len(objects)} files, saved {saved_patches} out of {total_patches} patches")
            else:
                logger.info(f"  Processed {count}/{len(objects)} files")
        
        except Exception as e:
            logger.error(f"  Error processing {s3_url}: {e}")
            import traceback
            traceback.print_exc()
    
    if use_patches:
        logger.info(f"Finished processing {count} files, saved {saved_patches} out of {total_patches} patches ({saved_patches/max(1, total_patches)*100:.2f}%)\n")
    else:
        logger.info(f"Finished processing {count} files")


def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Process Umbra SAR data to create water masks')
    
    # Input/output options
    parser.add_argument('--water-polygon', type=str, 
                        default='water-polygon/water_polygons.shp',
                        help='Path to water polygon shapefile')
    parser.add_argument('--output-dir', type=str, default='/home/ssb/DATASET/UmbraCoast',
                        help='Directory to save output files')
    
    # Processing options
    parser.add_argument('--min-water-ratio', type=float, default=0.7,
                        help='Minimum water ratio threshold (0.0-1.0) to save files')
    parser.add_argument('--max-water-ratio', type=float, default=0.95,
                        help='Maximum water ratio threshold (0.0-1.0) to save files')
    parser.add_argument('--max-files', type=int, default=-1,
                        help='Maximum number of files to process (-1 for all files)')
    parser.add_argument('--filter-suffix', type=str, default='_GEC.tif',
                        help='Filter for specific file suffix')
    
    # Output format options
    parser.add_argument('--save-as-png', action='store_true',
                        help='Save output as PNG instead of GeoTIFF')
    
    # S3 options
    parser.add_argument('--s3-prefix', type=str, default='sar-data/tasks/ship_detection_testdata',
                        help='S3 prefix to list')
    
    # Optimization options
    parser.add_argument('--downsample', type=int, default=1, choices=[1, 2, 4, 8],
                        help='Downsample factor to reduce image size (1=original, 2=half size, 4=quarter size)')
    
    # Patch processing options
    parser.add_argument('--use-patches', action='store_true',
                        help='Process images in patches instead of whole images')
    parser.add_argument('--patch-size', type=int, default=512,
                        help='Size of patches in pixels (square patches)')
    parser.add_argument('--patch-overlap', type=int, default=0,
                        help='Overlap between patches in pixels')
    
    return parser.parse_args()


# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Log the configuration
    logger.info(f"Processing Umbra SAR data with the following configuration:")
    logger.info(f"  Water polygon path: {args.water_polygon}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Min water ratio: {args.min_water_ratio}")
    logger.info(f"  Max water ratio: {args.max_water_ratio}")
    logger.info(f"  Save as PNG: {args.save_as_png}")
    logger.info(f"  Downsample factor: {args.downsample}")
    logger.info(f"  S3 prefix: {args.s3_prefix}")
    logger.info(f"  Filter suffix: {args.filter_suffix}")
    logger.info(f"  Max files: {args.max_files}")
    
    if args.use_patches:
        logger.info(f"  Patch processing: enabled (size={args.patch_size}, overlap={args.patch_overlap})")
    else:
        logger.info(f"  Patch processing: disabled")
    
    # Process Umbra data
    process_umbra_data(
        water_polygon_path=args.water_polygon,
        output_dir=args.output_dir,
        max_files=args.max_files,
        min_water_ratio=args.min_water_ratio,
        max_water_ratio=args.max_water_ratio,
        filter_suffix=args.filter_suffix,
        save_as_png=args.save_as_png,
        downsample_factor=args.downsample,
        s3_prefix=args.s3_prefix,
        use_patches=args.use_patches,
        patch_size=args.patch_size,
        patch_overlap=args.patch_overlap
    )
 