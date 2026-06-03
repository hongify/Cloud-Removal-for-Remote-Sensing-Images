## model.py
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)

class MultiBranchGenerator(nn.Module):
    def __init__(self):
        super(MultiBranchGenerator, self).__init__()
        
        self.enc_landsat = nn.Sequential(
            nn.Conv2d(7, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        

        self.enc_modis = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(inplace=True)
        )
        
        self.enc_sentinel = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(inplace=True)
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(320, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        res_blocks = []
        for _ in range(6):
            res_blocks.append(ResidualBlock(256))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 6, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        x_landsat = x[:, :6, :, :]
        x_modis = x[:, 6:12, :, :]
        x_sentinel = x[:, 12:14, :, :]
        x_mask = x[:, 14:15, :, :]
        
        in_landsat = torch.cat([x_landsat, x_mask], dim=1)
        
        feat_landsat = self.enc_landsat(in_landsat)
        feat_modis = self.enc_modis(x_modis)
        feat_sentinel = self.enc_sentinel(x_sentinel)
        
        merged = torch.cat([feat_landsat, feat_modis, feat_sentinel], dim=1)
        fused = self.fusion(merged)
        
        res_out = self.res_blocks(fused)
        
        out = self.decoder(res_out)
        return out
    
class PatchDiscriminator(nn.Module):
    def __init__(self, input_c=15, target_c=6):

        super(PatchDiscriminator, self).__init__()
        
        in_channels = input_c + target_c 
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),

            nn.Conv2d(512, 1, kernel_size=4, padding=1, bias=False)
        )

    def forward(self, condition, target):
        x = torch.cat((condition, target), dim=1) 
        return self.model(x)

if __name__ == "__main__":
    disc = PatchDiscriminator()
    cond = torch.randn(2, 15, 256, 256)
    real_or_fake = torch.randn(2, 6, 256, 256)
    out = disc(cond, real_or_fake)
    print("Discriminator Output shape:", out.shape)

if __name__ == "__main__":
    model = MultiBranchGenerator()
    x = torch.randn(2, 15, 256, 256)
    y = model(x)
    print("Generator Output shape:", y.shape) # Expect [2, 6, 256, 256]