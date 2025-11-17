import os
import numpy as np
import torch
import rasterio
import cv2

# -------------------------------
# CONFIG
# -------------------------------
input_folder = "/Users/frreyamehta/Documents/CV_Lab/mini_project/ls2s2/ls2s2_3_images"  # Folder with 3 Landsat .tif or converted .tif images
output_folder = "/Users/frreyamehta/Documents/CV_Lab/mini_project/ls2s2/ls2s2_preprocessed"
landsat_files = [
    "20220125_TM.tif", #"landsat_bands_20220125.tif"
    "20220705_TM.tif", #"landsat_bands_20220705.tif"
    "20221228_TM.tif"  #"landsat_bands_20221228.tif"
]

target_height = 2560  # same as original Landsat H
target_width = 3072   # same as original Landsat W
normalize_factor = 10000.0  # as per dataset

os.makedirs(output_folder, exist_ok=True)

# -------------------------------
# FUNCTIONS
# -------------------------------
def read_and_preprocess_tif(file_path, target_height, target_width):
    """
    Reads a multi-band Landsat TIFF and preprocesses it:
    - Replace no-data or negative values with 0
    - Normalize to 0-1
    - Resize bands if needed (here we keep original size)
    """
    with rasterio.open(file_path) as src:
        img = src.read(list(range(1, 7)))  # only bands 1-6  # shape: (bands, H, W)
    
    # Replace invalid / negative pixels
    img[img < 0] = 0

    # Normalize
    img = img / normalize_factor

    # Resize each band (optional, here keeping original Landsat size)
    bands, h, w = img.shape
    if (h != target_height) or (w != target_width):
        img_resized = np.zeros((bands, target_height, target_width), dtype=np.float32)
        for b in range(bands):
            img_resized[b] = cv2.resize(img[b], (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        return img_resized
    else:
        return img.astype(np.float32)

# -------------------------------
# MAIN
# -------------------------------
preprocessed_images = []

for landsat_file in landsat_files:
    file_path = os.path.join(input_folder, landsat_file)
    print(f"Processing {file_path}...")
    img_proc = read_and_preprocess_tif(file_path, target_height, target_width)
    preprocessed_images.append(img_proc)

# Stack along time dimension: shape = (3, bands, H, W)
stacked_images = np.stack(preprocessed_images, axis=0)

# Convert to torch tensor
stacked_tensor = torch.from_numpy(stacked_images)

# Save tensor for SwinSTFM input
output_path = os.path.join(output_folder, "landsat_3dates.pt")
torch.save(stacked_tensor, output_path)

print(f"Saved preprocessed Landsat tensor to {output_path}")
print(f"Tensor shape: {stacked_tensor.shape}")
