import math

import torch
from torch.nn import functional as F
from torchmetrics import Metric


class SSIM(Metric):

    def __init__(self, dist_sync_on_step=False, window_size=11, use_mask=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.window_size = window_size
        self.use_mask = use_mask
        window = create_window(window_size, 1)
        self.register_buffer('window', window)

        self.add_state('result', default=torch.tensor(0).float(), dist_reduce_fx='sum')
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx='sum')
        pass

    def update(self, out, target, mask=None):
        out = out.detach()
        out.clamp_(0, 1)
        out = out.unsqueeze(1)
        target = target.unsqueeze(1)

        ssim = _ssim(out, target, window=self.window, channel=1)
        ssim = ssim.view(ssim.size(0), -1)
        if self.use_mask:
            crop_size = (self.window_size - 1) // 2
            mask = mask[..., crop_size:-crop_size, crop_size:-crop_size].contiguous()
            mask = mask.view(target.size(0), -1)
            ssim = (ssim * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)
        else:
            ssim = ssim.mean(dim=-1)

        self.result += ssim.sum()
        self.total += ssim.numel()
        pass

    def compute(self):
        return self.result.float() / self.total


def _ssim(img1, img2, window, channel):
    mu1 = F.conv2d(img1, window, groups=channel)
    mu2 = F.conv2d(img2, window, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map


def create_window(window_size, channel, sigma=1.5):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()
