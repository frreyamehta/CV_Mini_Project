import rasterio
with rasterio.open("/Users/frreyamehta/Documents/CV_Lab/mini_project/ls2s2_3_testing_input/20220705_TM.tif") as src:
    print(src.count)         # Number of bands
    print(src.indexes)       # Which bands are present
