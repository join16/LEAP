import torch
from torch import nn

from .rsd import PS_BIN_TRAVEL_DIST, make_wave_conv_kernels, wave_kernel_to_frequency_domain


class WaveConvolutionLayer(nn.Module):

    def __init__(self, sampling_size, wavelength_params,

                 T=512,
                 bin_resolution=32, num_cycles=5, sigma=0.3,
                 normalize=False, threshold=-1,
                 ):
        super().__init__()
        wavelength_params = torch.tensor(wavelength_params)

        self.num_cycles = num_cycles
        self.sigma = sigma
        self.out_dim = wavelength_params.size(0) * 2
        self.threshold = threshold

        bin_length = bin_resolution * PS_BIN_TRAVEL_DIST
        all_wave_kernels = []
        base_wavelength = wavelength_params[0].item()
        for i, wl in enumerate(wavelength_params):
            wavelength = torch.tensor([wl])
            wave_kernels = make_wave_conv_kernels(wavelength, sampling_size, bin_length,
                                                  num_cycles=num_cycles,
                                                  sigma=sigma, base_wavelength=base_wavelength)
            kernel_f = wave_kernel_to_frequency_domain(wave_kernels, T=T, num_freqs=T)
            all_wave_kernels.append(kernel_f)
            pass

        all_wave_kernels = torch.cat(all_wave_kernels, dim=0)
        if normalize:
            all_wave_kernels = all_wave_kernels / all_wave_kernels.max(dim=-1, keepdim=True)[0]
        if self.threshold > 0:
            all_wave_kernels[all_wave_kernels < self.threshold] = 0
        N, O = all_wave_kernels.size()
        self.register_buffer('kernel', all_wave_kernels.view(1, N, O, 1, 1))
        pass

    def forward(self, x):
        x = torch.fft.fft(x, dim=2)
        x = x * self.kernel
        x = torch.fft.ifft(x, dim=2)
        x = torch.view_as_real(x).unsqueeze(1).transpose(1, -1).flatten(1, 2).contiguous().squeeze(-1)

        return x
