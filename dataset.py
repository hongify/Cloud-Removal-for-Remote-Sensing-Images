import os
import torch
import numpy as np
import rasterio
from torch.utils.data import Dataset, DataLoader

class CloudRemovalDataset(Dataset):
    def __init__(self, root_dir, split='train'):

        self.root_dir = os.path.join(root_dir, split)
        self.cloud_dir = os.path.join(self.root_dir, 'CloudLandsat_2020')
        self.modis_dir = os.path.join(self.root_dir, 'MODIS_2020')
        self.s1_dir = os.path.join(self.root_dir, 'Sentinel-1_2020-De')
        self.mask_dir = os.path.join(self.root_dir, 'Mask')
        self.gt_dir = os.path.join(self.root_dir, 'GT')
        
        self.filenames = sorted([f for f in os.listdir(self.cloud_dir) if f.endswith('.tif')])

    def __len__(self):
        return len(self.filenames)

    def _load_tif(self, path):
        with rasterio.open(path) as src:
            img = src.read()
            img = img.astype(np.float32)
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
            return img

    def _normalize(self, img, method='min-max'):

        if method == 'min-max':
            min_val = img.min()
            max_val = img.max()
            if max_val - min_val > 1e-6:
                img = (img - min_val) / (max_val - min_val)
            else:
                img = img - min_val
        elif method == 'clip':
            img = np.clip(img, 0.0, 1.0)
        return img

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        
        cloud_img = self._load_tif(os.path.join(self.cloud_dir, fname))
        modis_img = self._load_tif(os.path.join(self.modis_dir, fname))
        s1_img = self._load_tif(os.path.join(self.s1_dir, fname))
        mask_img = self._load_tif(os.path.join(self.mask_dir, fname))
        gt_img = self._load_tif(os.path.join(self.gt_dir, fname))
        
        cloud_img = self._normalize(cloud_img, method='clip')
        gt_img = self._normalize(gt_img, method='clip')
        
        modis_img = self._normalize(modis_img, method='min-max')
        s1_img = self._normalize(s1_img, method='clip')
        mask_img = self._normalize(mask_img, method='min-max')
        
        input_tensor = np.concatenate([cloud_img, modis_img, s1_img, mask_img], axis=0)
        
        input_tensor = torch.from_numpy(input_tensor).float()
        gt_img = torch.from_numpy(gt_img).float()
        
        return input_tensor, gt_img

if __name__ == '__main__':
    dataset = CloudRemovalDataset(root_dir='/home/user/Dataset/smilcr', split='train')
    print(f"Total dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        inputs, gt = dataset[0]
        print(f"Input shape: {inputs.shape}, dtype: {inputs.dtype}, min: {inputs.min()}, max: {inputs.max()}")
        print(f"GT shape: {gt.shape}, dtype: {gt.dtype}, min: {gt.min()}, max: {gt.max()}")
