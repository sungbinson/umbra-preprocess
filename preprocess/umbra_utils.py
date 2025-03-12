import time
import glob
import numpy as np
import os
from PIL import Image
import datetime

import geopandas as gpd
import rasterio
from rasterio import features
from shapely.geometry import box
from shapely.ops import transform
from pyproj import Transformer


def reproject_bounds(bounds, from_crs, to_crs):
    """Exchange a bbox from one coordinate system to another"""
    project = Transformer.from_crs(from_crs, to_crs, always_xy=True).transform
    return transform(project, box(*bounds))


def save_mask(water_mask, output_path, save_as_png, crs=None, transform=None, height=None, width=None):
    """Save water mask as PNG or GeoTIFF"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mask_255 = water_mask * 255
    
    if save_as_png:
        img = Image.fromarray(mask_255.astype(np.uint8))
        img.save(output_path.replace('.tif', '.png'))
    else:
        with rasterio.open(
            output_path, 'w', driver='GTiff', height=height, width=width,
            count=1, dtype=np.uint8, crs=crs, transform=transform
        ) as dst:
            dst.write(mask_255.astype(np.uint8), 1)


def preprocess_raster_data(data):
    """Preprocess raster data for better visualization"""
    if data.shape[0] == 3:  # RGB image
        rgb = np.zeros((data.shape[1], data.shape[2], 3), dtype=np.uint8)
        for i in range(3):
            band = data[i]
            min_val, max_val = np.percentile(band, [2, 98])
            rgb[:, :, i] = np.clip(255 * (band - min_val) / (max_val - min_val), 0, 255).astype(np.uint8)
        return rgb
    
    elif data.shape[0] == 1:  # Single band image
        band = data[0]
        min_val, max_val = np.percentile(band, [2, 98])
        return data[0]
        #return np.clip(255 * (band - min_val) / (max_val - min_val), 0, 255).astype(np.uint8)
    
    else:
        print(f"Unsupported number of bands: {data.shape[0]}")
        return data

def preprocess_with_global_stats(data, global_min_vals, global_max_vals):
    """Preprocess raster data using global min/max statistics for consistent normalization"""
    if data.shape[0] == 3:  # RGB image
        rgb = np.zeros((data.shape[1], data.shape[2], 3), dtype=np.uint8)
        for i in range(3):
            band = data[i]
            min_val, max_val = global_min_vals[i], global_max_vals[i]
            rgb[:, :, i] = np.clip(255 * (band - min_val) / (max_val - min_val), 0, 255).astype(np.uint8)
        return rgb
    elif data.shape[0] == 1:  # Single band image
        band = data[0]
        min_val, max_val = global_min_vals[0], global_max_vals[0]
        return np.clip(255 * (band - min_val) / (max_val - min_val), 0, 255).astype(np.uint8)
    else:
        print(f"Unsupported number of bands: {data.shape[0]}")
        return data

def save_raster_image(raster_data, output_path, save_as_png=True, downsample_factor=1, global_stats=None):
    """Save raster data as PNG or GeoTIFF with preprocessing
    
    Args:
        raster_data: Tuple of (data, profile) or path to raster file
        output_path: Path to save the output file
        save_as_png: Whether to save as PNG (True) or GeoTIFF (False)
        downsample_factor: Factor to downsample the image by
        global_stats: Optional tuple of (global_min_vals, global_max_vals) for consistent normalization
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get the data and profile
    if isinstance(raster_data, tuple):
        data, profile = raster_data
    else:
        with rasterio.open(raster_data) as src:
            data = src.read()
            profile = src.profile.copy()
    
    # Apply downsampling if requested
    if downsample_factor > 1:
        bands, height, width = data.shape
        new_height = height // downsample_factor
        new_width = width // downsample_factor
        downsampled_data = np.zeros((bands, new_height, new_width), dtype=data.dtype)
        
        for i in range(bands):
            for y in range(new_height):
                for x in range(new_width):
                    y_start = y * downsample_factor
                    y_end = min((y + 1) * downsample_factor, height)
                    x_start = x * downsample_factor
                    x_end = min((x + 1) * downsample_factor, width)
                    block = data[i, y_start:y_end, x_start:x_end]
                    downsampled_data[i, y, x] = np.mean(block)
        
        data = downsampled_data
        
        # Update transform to reflect new resolution
        transform = profile['transform']
        new_transform = rasterio.Affine(
            transform.a * downsample_factor, transform.b, transform.c,
            transform.d, transform.e * downsample_factor, transform.f
        )
        profile.update({
            'height': new_height,
            'width': new_width,
            'transform': new_transform
        })
    
    # Preprocess the data for better visualization
    if global_stats is not None:
        global_min_vals, global_max_vals = global_stats
        processed_data = preprocess_with_global_stats(data, global_min_vals, global_max_vals)
    else:
        processed_data = preprocess_raster_data(data)
    
    if save_as_png:
        # Save as PNG
        img = Image.fromarray(processed_data if data.shape[0] == 1 else processed_data)
        img.save(output_path.replace('.tif', '.png'), optimize=True, quality=85)
    else:
        # For GeoTIFF
        profile.update(dtype=rasterio.uint8)
        
        if data.shape[0] == 3:  # RGB image
            # Convert back to band-first format for rasterio
            processed_bands = np.zeros_like(data)
            for i in range(3):
                processed_bands[i] = processed_data[:, :, i]
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(processed_bands)
        elif data.shape[0] == 1:  # Single band image
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(processed_data[np.newaxis, :, :])
        else:
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data)


def create_water_mask(raster_src, water_polygon_path, clip_to_bounds=True):
    """Create a water mask from a raster source and water polygon"""
    # Handle input source
    close_src = False
    if isinstance(raster_src, str):
        src = rasterio.open(raster_src)
        close_src = True
    else:
        src = raster_src
    
    try:
        # Get raster metadata
        raster_transform = src.transform
        raster_crs = src.crs
        height, width = src.height, src.width
        raster_bounds = src.bounds
        
        # Create an empty mask
        water_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Load water polygon data with spatial filtering
        try:
            sample_gdf = gpd.read_file(water_polygon_path, rows=1)
            water_crs = sample_gdf.crs
            
            # Convert raster boundaries to water polygon coordinate system
            raster_bounds_water_crs = reproject_bounds(raster_bounds, raster_crs, water_crs)
            
            # Filter water polygons with transformed boundaries
            water_gdf = gpd.read_file(water_polygon_path, bbox=raster_bounds_water_crs.bounds)
            water_gdf = water_gdf.to_crs(raster_crs)
        except Exception:
            # If spatial filter fails, load all data
            water_gdf = gpd.read_file(water_polygon_path)
        
        # Check if water_gdf is empty
        if water_gdf.empty:
            return water_mask
        
        # Reproject if needed
        if water_gdf.crs != raster_crs:
            water_gdf = water_gdf.to_crs(raster_crs)
    
        # Clip to raster bounds if requested
        if clip_to_bounds:
            try:
                water_gdf = gpd.clip(water_gdf, raster_bounds)
                if water_gdf.empty:
                    return water_mask
            except Exception as e:
                print(f"Warning: Error during clipping, proceeding with unclipped polygons: {e}")
        
        # Rasterize in batches to save memory
        batch_size = 100
        for i in range(0, len(water_gdf), batch_size):
            batch = water_gdf.iloc[i:i+batch_size]
            shapes = [(geom, 1) for geom in batch.geometry]
            
            # Update the mask with this batch
            batch_mask = features.rasterize(
                shapes=shapes, out_shape=(height, width), transform=raster_transform,
                fill=0, dtype=np.uint8, all_touched=False
            )
            
            # Combine with existing mask (using logical OR)
            water_mask = np.logical_or(water_mask, batch_mask).astype(np.uint8)
        
        return water_mask
    
    finally:
        if close_src:
            src.close()


def process_single_raster(raster_src, water_polygon_path, output_mask_path=None, 
                         output_raster_path=None, save_as_png=True, clip_to_bounds=True,
                         min_water_ratio=0.05, max_water_ratio=0.95, downsample_factor=1):
    """Process a single raster to create a water mask"""
    try:
        # Open raster source
        close_src = False
        if isinstance(raster_src, str):
            src = rasterio.open(raster_src)
            close_src = True
        else:
            src = raster_src
        
        try:
            # Create water mask
            water_mask = create_water_mask(
                raster_src=src,
                water_polygon_path=water_polygon_path,
                clip_to_bounds=clip_to_bounds
            )
            
            # Calculate water ratio
            water_pixels = np.sum(water_mask)
            total_pixels = water_mask.size
            water_ratio = water_pixels / total_pixels
            
            # Get raster name for logging
            raster_name = src.name if hasattr(src, 'name') else "raster"
            raster_basename = raster_name.split('/')[-1]
            
            # Log water mask information
            print(f"\t📌 Water mask for {raster_basename}: {water_pixels} water pixels ({water_ratio*100:.2f}% of the raster).")
            
            # Save only if water ratio exceeds the threshold
            if water_ratio >= min_water_ratio and water_ratio <= max_water_ratio:
                print(f"\t✅ Water ratio {water_ratio*100:.2f}%, saving files.")
                
                # Save the mask if path is provided
                if output_mask_path:
                    save_mask(
                        water_mask, output_mask_path, save_as_png,
                        src.crs, src.transform, src.height, src.width
                    )
                    mask_path = output_mask_path if not save_as_png else output_mask_path.replace('.tif', '.png')
                    print(f"\t✅ Mask saved to {mask_path}")
                
                # Save the original raster if path is provided
                if output_raster_path:
                    data = src.read()
                    save_raster_image(
                        (data, src.profile), output_raster_path, 
                        save_as_png, downsample_factor
                    )
                    
                    raster_path = output_raster_path if not save_as_png else output_raster_path.replace('.tif', '.png')
                    optimization_info = f" (downsampled by {downsample_factor}x)" if downsample_factor > 1 else ""
                    print(f"\t✅ Original raster saved to {raster_path}{optimization_info}")

            
            return water_mask, water_ratio
        
        finally:
            if close_src:
                src.close()
        
    except Exception as e:
        print(f"Error processing raster: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0


def process_raster_in_patches(raster_src, water_polygon_path, output_dir, 
                             patch_size=512, patch_overlap=0, 
                             min_water_ratio=0.05, max_water_ratio=0.95, save_as_png=False,
                             use_global_normalization=False):
    """Process a raster image in patches, saving only patches with sufficient water coverage
    
    Args:
        raster_src: Path to raster file or open rasterio dataset
        water_polygon_path: Path to water polygon shapefile
        output_dir: Directory to save output files
        patch_size: Size of patches to extract
        patch_overlap: Overlap between adjacent patches
        min_water_ratio: Minimum ratio of water pixels to keep patch
        max_water_ratio: Maximum ratio of water pixels to keep patch
        save_as_png: Whether to save as PNG (True) or GeoTIFF (False)
        use_global_normalization: Whether to use global image statistics for normalization
    """
    # Open raster source
    close_src = False
    if isinstance(raster_src, str):
        src = rasterio.open(raster_src)
        close_src = True
    else:
        src = raster_src
    
    try:
        # Get raster metadata
        height, width = src.height, src.width
        raster_transform = src.transform
        raster_crs = src.crs
        raster_name = os.path.splitext(os.path.basename(src.name))[0] if hasattr(src, 'name') else "raster"
        
        # Create output directories
        masks_dir = os.path.join(output_dir, 'masks')
        rasters_dir = os.path.join(output_dir, 'rasters')
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(rasters_dir, exist_ok=True)
        
        # Create full water mask for the entire image
        print(f"Creating water mask for {raster_name}...")
        full_water_mask = create_water_mask(
            raster_src=src,
            water_polygon_path=water_polygon_path,
            clip_to_bounds=True
        )
        
        # Calculate total water ratio for the entire image
        total_water_pixels = np.sum(full_water_mask)
        total_water_ratio = total_water_pixels / full_water_mask.size
        
        print(f"Total water ratio for {raster_name}: {total_water_ratio*100:.2f}% ({total_water_pixels} water pixels)")
        
        # If the entire image has no water, skip processing patches
        if total_water_pixels == 0:
            print(f"No water found in {raster_name}, skipping patch processing")
            return 0, 0
        
        # Calculate patch grid
        effective_patch_size = patch_size - patch_overlap
        n_rows = (height - patch_overlap) // effective_patch_size + (1 if (height - patch_overlap) % effective_patch_size > 0 else 0)
        n_cols = (width - patch_overlap) // effective_patch_size + (1 if (width - patch_overlap) % effective_patch_size > 0 else 0)
        
        print(f"Processing {raster_name} in {n_rows}x{n_cols} = {n_rows*n_cols} patches (size: {patch_size}x{patch_size}, overlap: {patch_overlap})")
        
        # Calculate global statistics for normalization if requested
        global_stats = None
        if use_global_normalization:
            print("Calculating global statistics for normalization...")
            # Read the entire raster data
            full_data = src.read()
            bands = full_data.shape[0]
            
            global_min_vals = []
            global_max_vals = []
            
            for i in range(bands):
                min_val, max_val = np.percentile(full_data[i], [2, 98])
                global_min_vals.append(min_val)
                global_max_vals.append(max_val)
            
            global_stats = (global_min_vals, global_max_vals)
            print(f"Global min values: {global_min_vals}")
            print(f"Global max values: {global_max_vals}")
        
        # Process each patch
        total_patches = 0
        saved_patches = 0
        
        for row in range(n_rows):
            for col in range(n_cols):
                # Calculate patch coordinates
                y_start = row * effective_patch_size
                x_start = col * effective_patch_size
                
                # Ensure patch doesn't exceed image boundaries
                y_end = min(y_start + patch_size, height)
                x_end = min(x_start + patch_size, width)
                
                # Adjust start coordinates if patch would be smaller than patch_size
                y_start = max(0, y_end - patch_size)
                x_start = max(0, x_end - patch_size)
                
                # Extract patch water mask
                patch_water_mask = full_water_mask[y_start:y_end, x_start:x_end]
                
                # Calculate water ratio for this patch
                patch_water_pixels = np.sum(patch_water_mask)
                patch_water_ratio = patch_water_pixels / patch_water_mask.size
                
                # Create patch identifier
                patch_id = f"{raster_name}_y{y_start}_x{x_start}"
                
                # Only save patches with sufficient water
                if patch_water_ratio >= min_water_ratio and patch_water_ratio <= max_water_ratio:
                    # Extract patch data from original raster
                    patch_window = rasterio.windows.Window(x_start, y_start, x_end - x_start, y_end - y_start)
                    patch_data = src.read(window=patch_window)
                    
                    # Calculate patch transform
                    patch_transform = rasterio.transform.from_origin(
                        raster_transform.c + x_start * raster_transform.a,
                        raster_transform.f + y_start * raster_transform.e,
                        raster_transform.a,
                        -raster_transform.e
                    )
                    
                    # Create patch profile
                    patch_profile = src.profile.copy()
                    patch_profile.update({
                        'height': y_end - y_start,
                        'width': x_end - x_start,
                        'transform': patch_transform
                    })
                    
                    # Save patch mask and raster
                    patch_mask_path = os.path.join(masks_dir, f"{patch_id}.tif")
                    save_mask(
                        patch_water_mask, patch_mask_path, save_as_png,
                        raster_crs, patch_transform, y_end - y_start, x_end - x_start
                    )
                    
                    patch_raster_path = os.path.join(rasters_dir, f"{patch_id}.tif")
                    save_raster_image(
                        (patch_data, patch_profile), patch_raster_path,
                        save_as_png, downsample_factor=1, global_stats=global_stats
                    )
                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{current_time}] ✅ Saved patch {patch_id} (water ratio: {patch_water_ratio*100:.2f}%)")
                    saved_patches += 1

                
                total_patches += 1
        
        print(f"Finished processing {raster_name}: saved {saved_patches} out of {total_patches} patches ({saved_patches/total_patches*100:.2f}%)")
        return total_patches, saved_patches
    
    finally:
        if close_src:
            src.close()