# demo.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="rasterio")

from dataset import CloudRemovalDataset
from model import MultiBranchGenerator 

def visualize_results(inputs, targets, outputs, idx=0):
    """
    inputs: [C, H, W]
    targets: [C, H, W]
    outputs: [C, H, W]
    """

    rgb_indices = [2, 1, 0]
    
    cloud_img = inputs[:6]
    cloud_rgb = cloud_img[rgb_indices].cpu().numpy().transpose(1, 2, 0)
    
    gt_rgb = targets[rgb_indices].cpu().numpy().transpose(1, 2, 0)
    
    pred_rgb = outputs[rgb_indices].cpu().numpy().transpose(1, 2, 0)
    
    cloud_rgb = np.clip(cloud_rgb, 0, 1)
    gt_rgb = np.clip(gt_rgb, 0, 1)
    pred_rgb = np.clip(pred_rgb, 0, 1)
    
    cloud_nir = cloud_img[3].cpu().numpy()
    gt_nir = targets[3].cpu().numpy()
    pred_nir = outputs[3].cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(cloud_rgb)
    axes[0, 0].set_title("Input (Cloudy) - RGB")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gt_rgb)
    axes[0, 1].set_title("Target (Ground Truth) - RGB")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(pred_rgb)
    axes[0, 2].set_title("Prediction - RGB")
    axes[0, 2].axis('off')

    # NIR
    axes[1, 0].imshow(cloud_nir, cmap='gray')
    axes[1, 0].set_title("Input (Cloudy) - NIR")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(gt_nir, cmap='gray')
    axes[1, 1].set_title("Target (Ground Truth) - NIR")
    axes[1, 1].axis('off')

    axes[1, 2].imshow(pred_nir, cmap='gray')
    axes[1, 2].set_title("Prediction - NIR")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'demo_result_{idx}.png')
    print(f"Saved demo_result_{idx}.png")
    plt.close()

def main():
    root_dir = '/Dataset/smilecr'
    model_path = 'checkpoints/best_generator.pth' 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading dataset...")
    try:
        val_dataset = CloudRemovalDataset(root_dir=root_dir, split='test')
    except Exception as e:
        print(f"test dataset load failed: {e}")
        print("trying another val dataset.")
        val_dataset = CloudRemovalDataset(root_dir=root_dir, split='val')
    
    if len(val_dataset) == 0:
        print("dataset empty.")
        return
    
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    print("Loading model...")
    model = MultiBranchGenerator().to(device) 
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k # 'module.' 제거
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict)
        print("Model loaded successfully.")
    else:
        print(f"Model weight file not found at {model_path}.")
        print("Please train the model first or provide a valid checkpoint.")
        return

    model.eval()

    num_samples = 3
    print(f"Generating {num_samples} demo images...")

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):
            if i >= num_samples:
                break
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            visualize_results(inputs[0], targets[0], outputs[0], idx=i)

    print("Demo completed!")

if __name__ == "__main__":
    main()