import torch
import numpy as np
import os
from dataset import CloudRemovalDataset
from tqdm import tqdm

def main():
    dataset = CloudRemovalDataset(root_dir='/home/user/Dataset/smilcr', split='train')
    print(f"Training Dataset size: {len(dataset)}")

    for i in tqdm(range(len(dataset))):
        inputs, gt = dataset[i]
        if torch.isnan(inputs).any() or torch.isnan(gt).any():
            print(f"NaN found at index {i}")
            if torch.isnan(inputs).any():
                print("NaN in inputs")
                for ch in range(inputs.shape[0]):
                    if torch.isnan(inputs[ch]).any():
                        print(f"  Channel {ch} has NaN")
            if torch.isnan(gt).any():
                print("NaN in gt")
            return
        if torch.isinf(inputs).any() or torch.isinf(gt).any():
            print(f"Inf found at index {i}")
            return

    print("Check finished. No NaN or Inf found in the training dataset.")

if __name__ == '__main__':
    main()
