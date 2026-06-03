import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.tensor(
        [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
         for x in range(window_size)],
        dtype=torch.float32
    )
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1d = gaussian(window_size, 1.5).unsqueeze(1)
    _2d = _1d @ _1d.t()
    window = _2d.unsqueeze(0).unsqueeze(0)
    window = window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean([1, 2, 3])


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.register_buffer("window", create_window(window_size, 1))
        self.channel = 1

    def forward(self, img1, img2):
        b, c, h, w = img1.shape

        if c != self.channel:
            window = create_window(self.window_size, c).to(img1.device).type_as(img1)
            self.window = window
            self.channel = c

        return 1 - _ssim(img1, img2, self.window, self.window_size, c, self.size_average)

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()

        kx = torch.tensor(
            [[-1., 0., 1.],
             [-2., 0., 2.],
             [-1., 0., 1.]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        ky = torch.tensor(
            [[-1., -2., -1.],
             [0., 0., 0.],
             [1., 2., 1.]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("weight_x", kx)
        self.register_buffer("weight_y", ky)

    def forward(self, pred, target):
        b, c, h, w = pred.shape

        weight_x = self.weight_x.repeat(c, 1, 1, 1)
        weight_y = self.weight_y.repeat(c, 1, 1, 1)

        pred_gx = F.conv2d(pred, weight_x, padding=1, groups=c)
        pred_gy = F.conv2d(pred, weight_y, padding=1, groups=c)

        target_gx = F.conv2d(target, weight_x, padding=1, groups=c)
        target_gy = F.conv2d(target, weight_y, padding=1, groups=c)

        pred_grad = torch.sqrt(pred_gx ** 2 + pred_gy ** 2 + 1e-6)
        target_grad = torch.sqrt(target_gx ** 2 + target_gy ** 2 + 1e-6)

        return F.l1_loss(pred_grad, target_grad)


class CloudRemovalLoss(nn.Module):
    def __init__(self,
                 band_weights=None,
                 alpha=0.84,
                 lambda_edge=0.5):

        super().__init__()

        if band_weights is None:
            band_weights = [1., 1., 1., 3., 1., 1.]

        self.register_buffer("band_weights",
                             torch.tensor(band_weights, dtype=torch.float32))

        self.l1_loss = nn.L1Loss(reduction='none')
        self.ssim_loss = SSIMLoss()
        self.edge_loss = EdgeLoss()

        self.alpha = alpha
        self.lambda_edge = lambda_edge

    def forward(self, pred, target, mask=None):
        weights = self.band_weights.view(1, -1, 1, 1)
        l1_diff = torch.abs(pred - target) * weights

        if mask is not None:
            mask_weight = 1.0 + 4.0 * mask
            l1_diff = l1_diff * mask_weight
        l1_loss = l1_diff.mean()

        ssim_loss = self.ssim_loss(pred, target)
        edge_loss = self.edge_loss(pred, target)
        base_loss = self.alpha * ssim_loss + (1 - self.alpha) * l1_loss
        
        total_loss = base_loss + self.lambda_edge * edge_loss

        return total_loss


if __name__ == "__main__":
    loss_fn = CloudRemovalLoss()

    pred = torch.randn(2, 6, 256, 256)
    target = torch.randn(2, 6, 256, 256)
    mask = torch.rand(2, 1, 256, 256)

    loss = loss_fn(pred, target, mask)
    print("Improved Combined Loss:", loss.item())