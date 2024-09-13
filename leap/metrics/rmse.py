import torch
from torchmetrics import Metric


class RMSE(Metric):

    def __init__(self, dist_sync_on_step=False, scale_factor=1.0, use_mask=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.scale_factor = scale_factor
        self.use_mask = use_mask

        self.add_state('result', default=torch.tensor(0).float(), dist_reduce_fx='sum')
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx='sum')
        pass

    def update(self, out, target, mask=None):
        out = out.detach()
        out.clamp_(0, 1)

        mse = (out - target).abs() * self.scale_factor
        mse = mse.view(mse.size(0), -1)
        rmse = torch.sqrt(mse)
        if self.use_mask:
            mask = mask.view(target.size(0), -1)
            rmse = (rmse * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8)
        else:
            rmse = rmse.mean(dim=-1)

        self.result += rmse.sum()
        self.total += rmse.numel()
        pass

    def compute(self):
        return self.result.float() / self.total
