import os
import numpy as np
import rasterio

# Base folder containing all date subfolders
#Do separately for MODIS and LANDSAT

#base = "/Users/frreyamehta/Documents/CV_Lab/mini_project/Landsat_and_MODIS_data_for_the_Lower_Gwydir_Catchment_study_site-31s6lPoF-_MODIS_unzipped/data/LGC/MODIS"  # CHANGE this to your actual folder

base = "/Users/frreyamehta/Documents/CV_Lab/mini_project/Landsat_and_MODIS_data_for_the_Lower_Gwydir_Catchment_study_site-LlVje3mV-_LANDSAT_unzipped/data/LGC/Landsat" 

# Loop through each date folder
for date_folder in os.listdir(base):
    date_path = os.path.join(base, date_folder)
    if not os.path.isdir(date_path):
        continue

    # Loop through all .int files in this folder
    for file in os.listdir(date_path):
        if file.endswith(".int"):
            int_path = os.path.join(date_path, file)
            
            # Load the .int file
            data = np.fromfile(int_path, dtype=np.int16)  # or dtype=np.float32 if needed
            # Assuming 6 bands for Landsat/MODIS as per README
            bands = 6
            rows = 2720  # adjust if needed
            cols = 3200  # adjust if needed
            data = data.reshape((bands, rows, cols))

            # Prepare output .tif path (same folder, same name)
            tif_path = os.path.join(date_path, file.replace(".int", ".tif"))

            # Save as GeoTIFF
            with rasterio.open(
                tif_path,
                "w",
                driver="GTiff",
                height=rows,
                width=cols,
                count=bands,
                dtype=data.dtype,
            ) as dst:
                for i in range(bands):
                    dst.write(data[i, :, :], i + 1)

            print(f"Converted {int_path} -> {tif_path}")
