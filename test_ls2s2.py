import os
import torch
import numpy as np
import rasterio
from rasterio.transform import Affine
from models.swinstfm import SwinSTFM


# ---------------------------------------------
# Load the preprocessed .pt tensor
# ---------------------------------------------
def load_preprocessed_tensor(pt_path):
    data = torch.load(pt_path)  # shape = [3, 6, H, W]
    assert data.ndim == 4 and data.shape[0] == 3
    return data.float()


# ---------------------------------------------
# Tile a large image into 256×256 patches
# ---------------------------------------------
def tile_image(img, tile=256):
    _, H, W = img.shape
    patches = []
    coords = []
    for i in range(0, H, tile):
        for j in range(0, W, tile):
            patch = img[:, i:i+tile, j:j+tile]
            if patch.shape[1] == tile and patch.shape[2] == tile:
                patches.append(patch)
                coords.append((i, j))
    return patches, coords


# ---------------------------------------------
# Reconstruct full image from tiles
# ---------------------------------------------
def reconstruct_image(tiles, coords, H, W, C):
    out = np.zeros((C, H, W), dtype=np.float32)
    for tile, (i, j) in zip(tiles, coords):
        out[:, i:i+tile.shape[1], j:j+tile.shape[2]] = tile
    return out


# ---------------------------------------------
# Save output as GeoTIFF
# ---------------------------------------------
def save_geotiff(output_path, array, reference_tif_path):
    array = array.astype(np.float32)

    with rasterio.open(reference_tif_path) as src:
        profile = src.profile
        profile.update({
            "height": array.shape[1],
            "width": array.shape[2],
            "count": array.shape[0]
        })

    with rasterio.open(output_path, "w", **profile) as dst:
        for b in range(array.shape[0]):
            dst.write(array[b], b + 1)


# ---------------------------------------------
# Main Inference
# ---------------------------------------------
def run_inference(
    pt_path,
    reference_tif,
    output_path,
    tile=256
):

    device = torch.device("cpu")

    # Load preprocessed 3-date tensor
    data = load_preprocessed_tensor(pt_path)  # [3, 6, 2560, 3072]
    print("Loaded tensor shape:", data.shape)

    # Split per SwinSTFM expectations
    ref_lr = data[0]  # (6, H, W)
    ref_hr = data[1]
    input_lr = data[2]

    # Tile each into 256×256 patches
    patches_ref_lr, coords = tile_image(ref_lr)
    patches_ref_hr, _ = tile_image(ref_hr)
    patches_input_lr, _ = tile_image(input_lr)
    print("Total patches:", len(patches_ref_lr))

    # Load SwinSTFM model (no args!)
    model = SwinSTFM()
    model.to(device)
    model.eval()

    outputs = []

    # Loop over tiles
    with torch.no_grad():
        for i in range(len(patches_ref_lr)):
            p1 = patches_ref_lr[i].detach().clone().unsqueeze(0)  # (1, 6, 256, 256)
            p2 = patches_ref_hr[i].detach().clone().unsqueeze(0)
            p3 = patches_input_lr[i].detach().clone().unsqueeze(0)

            out = model(p1, p2, p3)[0].numpy()  # (6, 256, 256)
            outputs.append(out)

    # Reconstruct full image
    full_output = reconstruct_image(
        outputs,
        coords,
        H=ref_lr.shape[1],
        W=ref_lr.shape[2],
        C=ref_lr.shape[0]
    )

    print("Final output shape:", full_output.shape)  # (6, 2560, 3072)

    # Save GeoTIFF
    save_geotiff(output_path, full_output, reference_tif)


if __name__ == "__main__":
    run_inference(
        pt_path="/Users/frreyamehta/Documents/CV_Lab/mini_project/ls2s2/ls2s2_preprocessed/landsat_3dates.pt",
        reference_tif="/Users/frreyamehta/Documents/CV_Lab/mini_project/ls2s2/ls2s2_3_images/20220125_TM.tif",        
        output_path="/Users/frreyamehta/Documents/CV_Lab/mini_project/ls2s2/swin_output_full.tif",
        tile=256
    )
