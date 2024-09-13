from typing import Any, Optional

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from .base import BaseSolver
from leap.data import NLOSBaseDataModule
from leap.models.rsd import (RSDConvolution, PS_BIN_TRAVEL_DIST,
                             make_wave_conv_kernels, wave_kernel_to_frequency_domain, compute_wave_freq_range)
from leap.models.utilis import to_frequency_feature
from leap.metrics import PSNR, SSIM, RMSE


class NLOSAutoEncoderSolver(BaseSolver):

    def __init__(self, dm: NLOSBaseDataModule, model: nn.Module,

                 ## optimizer, scheduler params
                 lr: float = 1e-4,
                 lr_step_size: int = -1,
                 lr_gamma: float = 0.1,
                 weight_decay: float = 0,

                 ## phasor params
                 target_wavelength: float = 1.5,
                 target_num_cycles: int = 5,
                 target_sigma: float = 0.3,
                 depth_threshold: float = 0.1,  # albedo threshold for depth map
                 ):
        super().__init__()

        self.dm = dm
        self.model = model

        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.depth_threshold = depth_threshold

        self.criterion = nn.L1Loss(reduction='none')
        self.psnr_metric = PSNR()
        self.ssim_metric = SSIM()
        self.rmse_metric = RMSE()

        ## build RSD related modules
        T, N = dm.transient_shape
        O = T // 2 + 1
        self.T = T
        self.O = O
        sample_size = dm.wall_size / N
        bin_length = dm.bin_resolution * PS_BIN_TRAVEL_DIST
        wavelength = torch.ones(1) * target_wavelength
        wave_kernel = make_wave_conv_kernels(wavelength, sample_size, bin_length,
                                             num_cycles=target_num_cycles, sigma=target_sigma)
        wave_kernel_f = wave_kernel_to_frequency_domain(wave_kernel, T=T)
        self.freq_range = compute_wave_freq_range(wave_kernel_f[0])
        self.rsd_conv = RSDConvolution(
            O=O, N=N, wall_size=dm.wall_size,
            light_coord=(0, 0, 0),
            is_confocal=dm.is_confocal,
            bin_length=bin_length,
            falloff=False,
            use_pad=True,
            freq_range=self.freq_range,
        )
        self.register_buffer('wave_kernel_f', wave_kernel_f.squeeze(0), persistent=False)

    def configure_optimizers(self):
        optimizers = [AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)]
        schedulers = []
        if self.lr_step_size > 0:
            schedulers = [StepLR(optimizers[0], step_size=self.lr_step_size, gamma=self.lr_gamma)]
        return optimizers, schedulers

    @property
    def _is_training(self):
        return (self.trainer is not None) and self.trainer.training

    def forward(self, batch):
        transient = batch['transient']
        metadata = batch['metadata']

        metadata['freq_range'] = self.freq_range
        outputs = self.model(transient, metadata)

        if not self._is_training:
            ## hidden scenes reconstruction
            out_aperture = outputs['phasor'].contiguous().float()
            albedo, depth = self._reconstruct_hidden_scenes(out_aperture)
            outputs['front'] = albedo
            outputs['depth'] = depth

        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        pred = outputs['phasor']
        label = batch['clean_transient']
        label = to_frequency_feature(label, self.freq_range)

        ## loss coefficients
        start, end = self.freq_range
        wave_coefficients = self.wave_kernel_f[start:end]
        wave_coefficients = wave_coefficients / wave_coefficients.max().item()
        wave_coefficients.clamp_min_(0.1)
        wave_coefficients = wave_coefficients.view(1, 1, wave_coefficients.size(0), 1, 1)

        diffs = self.criterion(pred, label)
        loss = (diffs * wave_coefficients).mean()
        self.log('train/loss', loss.item())

        return {
            'loss': loss,
        }

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)

        label = batch['clean_transient']
        label = to_frequency_feature(label, self.freq_range)
        front_label, _ = self._reconstruct_hidden_scenes(label)

        self.psnr_metric.update(outputs['front'], front_label)
        self.ssim_metric.update(outputs['front'], front_label)
        self.rmse_metric.update(outputs['depth'], batch['depth'])

    def on_validation_epoch_end(self):
        psnr = self.psnr_metric.compute()
        ssim = self.ssim_metric.compute()
        rmse = self.rmse_metric.compute() * 2

        self.log('val/psnr', psnr, rank_zero_only=True, on_epoch=True, sync_dist=True)
        self.log('val/ssim', ssim, rank_zero_only=True, on_epoch=True, sync_dist=True)
        self.log('val/rmse', rmse, rank_zero_only=True, on_epoch=True, sync_dist=True)
        self.track_score(psnr)
        print(f'psnr: {psnr:.1f}, ssim: {ssim:.1f}, rmse: {rmse:.4f}')

    def _reconstruct_hidden_scenes(self, out_aperture):
        # prepare wave kernel
        start, end = self.freq_range
        wave_filter = self.wave_kernel_f[start:end]
        wave_filter = wave_filter.view(1, 1, wave_filter.size(0), 1, 1)

        out_aperture = out_aperture * wave_filter
        out_aperture = out_aperture.unsqueeze(-1).transpose(1, -1).contiguous()
        volumes = self._run_diffraction(out_aperture)
        volumes = volumes.transpose(1, -1).squeeze(-1).contiguous()

        volumes = volumes.pow(2).sum(dim=1).sqrt()
        volumes[..., :4] = 0
        volumes[..., -4:] = 0

        B, D = volumes.size(0), volumes.size(-1)
        albedo, depth = torch.max(volumes, dim=-1)
        albedo = albedo.clamp_min(0)
        depth = (depth + 0.5) / D
        depth = (1 - depth)
        max_vals = albedo.view(B, -1).max(dim=-1)[0].view(B, 1, 1).clamp_min(1e-8)
        albedo = albedo / max_vals
        depth[albedo < self.depth_threshold] = 0
        depth[depth < 0.05] = 0
        depth[depth > 0.95] = 0

        return albedo, depth

    @torch.no_grad()
    def _run_diffraction(self, aperture):
        return self.rsd_conv(aperture)
