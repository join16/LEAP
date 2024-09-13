import torch
from torch.nn import functional as F
from torchmetrics import Metric


class PSNR(Metric):

    def __init__(self, dist_sync_on_step=False, use_mask=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.use_mask = use_mask
        self.add_state('result', default=torch.tensor(0).float(), dist_reduce_fx='sum')
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx='sum')
        pass

    def update(self, out, target, mask=None):
        out = out.detach()
        out.clamp_(0, 1)
        target.clamp_(0, 1)
        mse = F.mse_loss(out, target, reduction='none')
        mse = mse.view(mse.size(0), -1)
        if self.use_mask:
            mask = mask.view(target.size(0), -1)
            mse = (mse * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)
        else:
            mse = mse.mean(dim=-1)

        mask = mse == 0
        psnr = 10 * torch.log10(1. / mse)
        psnr[mask] = 0

        self.result += psnr.sum()
        self.total += psnr.numel()
        pass

    def compute(self):
        return self.result.float() / self.total
