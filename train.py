import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter
import subprocess
import time
import re
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="rasterio")

from dataset import CloudRemovalDataset
from model import MultiBranchGenerator, PatchDiscriminator
from loss import CloudRemovalLoss

def train():
    parser = argparse.ArgumentParser(description='Cloud Removal Training with GAN')
    parser.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training (e.g., checkpoints/checkpoint_epoch_10.pth)')
    args = parser.parse_args()

    root_dir = '/Dataset/smilecr'
    batch_size = 4
    lr_G = 2e-4 
    lr_D = 2e-4
    num_epochs = 50
    lambda_pixel = 100 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading datasets...")
    train_dataset = CloudRemovalDataset(root_dir=root_dir, split='train')
    val_dataset = CloudRemovalDataset(root_dir=root_dir, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    generator = MultiBranchGenerator().to(device)
    discriminator = PatchDiscriminator().to(device)

    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_pixel = CloudRemovalLoss().to(device)

    optimizer_G = optim.Adam(generator.parameters(), lr=lr_G, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.5, 0.999))

    scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=0.5, patience=3, verbose=True)
    scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=0.5, patience=3, verbose=True)

    start_epoch = 0
    best_val_loss = float('inf')
    os.makedirs('checkpoints', exist_ok=True)

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            
            generator.load_state_dict(checkpoint['generator_state_dict'])
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            
            optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
            scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
            
            print(f"=> Loaded checkpoint '{args.resume}' (Resuming from epoch {start_epoch})")
        else:
            print(f"=> No checkpoint found at '{args.resume}'")

    if torch.cuda.device_count() > 1:
        generator = torch.nn.DataParallel(generator)
        discriminator = torch.nn.DataParallel(discriminator)

    writer = None
    if args.tensorboard:
        print("Starting TensorBoard...")
        writer = SummaryWriter('runs/cloud_removal_gan')
        tb_process = subprocess.Popen(['tensorboard', '--logdir', 'runs', '--port', '6006', '--host', '127.0.0.1'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        cf_process = subprocess.Popen(['cloudflared', 'tunnel', '--url', 'http://127.0.0.1:6006'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        print("Waiting for Cloudflare Tunnel URL...")
        public_url = None
        for _ in range(15): 
            line = cf_process.stderr.readline()
            if 'trycloudflare.com' in line:
                match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
                if match:
                    public_url = match.group(0)
                    break
            time.sleep(1)
            
        if public_url:
            print(f"\n=======================================================")
            print(f"TensorBoard is publicly available at:")
            print(f"--> {public_url} <--")
            print(f"=======================================================\n")
        else:
            print("Failed to get Cloudflare Tunnel URL. Check if cloudflared is installed and working.")

    for epoch in range(start_epoch, num_epochs):
        generator.train()
        discriminator.train()
        
        train_loss_G = 0.0
        train_loss_D = 0.0
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        pbar = tqdm(train_loader, desc=f"Training")
        
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            mask = inputs[:, 14:15, :, :] 
            fakes = generator(inputs)

            optimizer_D.zero_grad()
            pred_real = discriminator(inputs, targets)
            loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real)) 
            
            pred_fake = discriminator(inputs, fakes.detach()) 
            loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake)) 
            
            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            pred_fake = discriminator(inputs, fakes)
            loss_G_adv = criterion_GAN(pred_fake, torch.ones_like(pred_fake)) 
            loss_pixel = criterion_pixel(fakes, targets, mask)
            
            loss_G = loss_G_adv + lambda_pixel * loss_pixel
            loss_G.backward()
            optimizer_G.step()

            train_loss_G += loss_G.item() * inputs.size(0)
            train_loss_D += loss_D.item() * inputs.size(0)
            pbar.set_postfix({'G_Loss': loss_G.item(), 'D_Loss': loss_D.item()})
            
        train_loss_G = train_loss_G / len(train_dataset)
        train_loss_D = train_loss_D / len(train_dataset)

        if writer:
            writer.add_scalar('Loss/Train_Generator', train_loss_G, epoch)
            writer.add_scalar('Loss/Train_Discriminator', train_loss_D, epoch)

        generator.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Validation"):
                inputs = inputs.to(device)
                targets = targets.to(device)
                mask = inputs[:, 14:15, :, :]

                fakes = generator(inputs)
                loss_pixel = criterion_pixel(fakes, targets, mask)
                val_loss += loss_pixel.item() * inputs.size(0)
                
        val_loss = val_loss / len(val_dataset)
        print(f"Train G_Loss: {train_loss_G:.4f} | Train D_Loss: {train_loss_D:.4f} | Val Pixel Loss: {val_loss:.4f}")

        if writer:
            writer.add_scalar('Loss/Validation_Pixel', val_loss, epoch)

        scheduler_G.step(val_loss)
        scheduler_D.step(val_loss)

        gen_state = generator.module.state_dict() if isinstance(generator, nn.DataParallel) else generator.state_dict()
        disc_state = discriminator.module.state_dict() if isinstance(discriminator, nn.DataParallel) else discriminator.state_dict()

        checkpoint_state = {
            'epoch': epoch,
            'generator_state_dict': gen_state,
            'discriminator_state_dict': disc_state,
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'scheduler_G_state_dict': scheduler_G.state_dict(),
            'scheduler_D_state_dict': scheduler_D.state_dict(),
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint_state, f'checkpoints/checkpoint_epoch_{epoch+1}.pth')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(gen_state, f'checkpoints/best_generator.pth')
            print(f"Best Generator saved (Val Loss: {val_loss:.4f})")
            
    print("Training Complete!")
    if writer:
        writer.close()

if __name__ == '__main__':
    train()