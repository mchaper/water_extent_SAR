#### Author: Miguel Chapel Rivas
# Functions for de SAR exercise

import rasterio
from rasterio.mask import mask
import rasterio.crs as rio_crs
import os
import numpy as np
import warnings
from scipy.ndimage import median_filter

# Suppress RuntimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

def clip_image_to_aoi(image_path,aoi_to_clip):
    """
    This function receives a image, clips to aoi and convert to dB
    """
    with rasterio.open(image_path, 'r') as src:

        # Specify the desired CRS (e.g., EPSG:4326 for WGS84)
        dst_crs = rio_crs.CRS.from_epsg(4326)

        # Create a new dataset with the desired CRS and GCPs
        dst_dataset = rasterio.vrt.WarpedVRT(src, crs=dst_crs, gcps=src.gcps)

        # Clip the image to the clipping extent
        clipped_image, clipped_transform = mask(dst_dataset, [aoi_to_clip], crop=True)
        
        # Create a new rasterio dataset for the clipped image
        clipped_profile = src.profile
        clipped_profile.update({
            'crs':dst_crs,
            'height': clipped_image.shape[1],
            'width': clipped_image.shape[2],
            'transform': clipped_transform,
            'nodata': None  # Update with the appropriate nodata value if needed
        })
        #Convert clipped image to dB
        clipped_image_dB = 10 * np.log10(clipped_image)

        # Get filename of the original file 
        file_name = os.path.basename(image_path)

        # Save the clipped image to a new file
        clipped_image_path = os.path.join('temp','_'.join(['clipped',file_name]))
        with rasterio.open(clipped_image_path, 'w', **clipped_profile) as dst:
            dst.write(clipped_image_dB)

    print("Clipped " + file_name )
    return clipped_image_path


def speckle_filter(image_path):
    """
    This recieves an image an applies a median filter
    """
    with rasterio.open(image_path) as src:
        image = src.read(1)
        profile = src.profile.copy()

    # Apply the median filter
    filtered_image = median_filter(image, size=5)

    # Get filename of the original file 
    file_name = os.path.basename(image_path)

    # Save the filtered image to a new file
    filtered_image_path =  os.path.join('temp','_'.join(['filter',file_name]))
    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(filtered_image_path, 'w', **profile) as dst:
        dst.write(filtered_image.astype(rasterio.float32), 1)

    return filtered_image_path

def get_water_mask(image_path, water_threshold):
    """
    This recieves an image an using a user given threshold calculates the part that corrsponds to water
    """
    with rasterio.open(image_path) as src:
        image = src.read(1)
        # Convert the image mask to a bitmask where water pixels are set to 1 and non-water pixels are set to 0
        water_mask = np.where(image <= water_threshold, 1, 0)
        bitmask = water_mask.astype('uint8')
        # Get the metadata from the source image
        profile = src.profile
 
    water_pixels = np.count_nonzero(bitmask)  # Count the number of water pixels
    area_km2 = (water_pixels * 10 * 10) / 1e6 # Assume square pixels of 10 by 10 meters

    # Get filename of the original file 
    file_name = os.path.basename(image_path)

    # Save the bitmask 
    bitmask_path = os.path.join('output','_'.join(['water_bitmask',file_name]))
    with rasterio.open(bitmask_path, 'w', **profile) as dst:
        dst.write(bitmask, 1)  # Use band index 1

    return bitmask_path,area_km2
