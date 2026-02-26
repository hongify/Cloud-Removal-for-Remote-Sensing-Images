import rasterio
import numpy as np
import os
from dataset import CloudRemovalDataset

dataset = CloudRemovalDataset(root_dir='/home/user/Dataset/smilcr', split='train')
fname = dataset.filenames[390]
print(f"File name: {fname}")

with rasterio.open(os.path.join(dataset.cloud_dir, fname)) as src:
    cloud = src.read()
    print(f"Cloud has NaN: {np.isnan(cloud).any()}")

with rasterio.open(os.path.join(dataset.gt_dir, fname)) as src:
    gt = src.read()
    print(f"GT has NaN: {np.isnan(gt).any()}")
