import torch
from torch import nn
from torch.nn import functional as F

C = 3e8
PS_BIN_TRAVEL_DIST = 0.0001 * 3


class RSDConvolution(nn.Module):
    """
    Reproduced version of RSD kernel convolution in "https://github.com/fmu2/nlos3d"
    """

    def __init__(self, O, N, wall_size,
                 light_coord=(0, 0, 0), is_confocal=True,
                 bin_length=0.0096,
                 falloff=False,
                 use_pad=True,
                 num_depth_planes=-1, z_min=0, z_max=2,
                 freq_range=None,
                 ):
        """
        :param O: number of frequencies
        :param N: number of pixels in each dimension
        :param wall_size: size of the wall
        :param light_coord: coordinate of the light source
        :param is_confocal: whether the light source is confocal
        :param num_depth_planes: number of depth planes (default = N)
        :param z_min: minimum depth
        :param z_max: maximum depth
        :param freq_range: frequency range to consider
        """
        super().__init__()
        if num_depth_planes <= 0:
            num_depth_planes = N
        if is_confocal:
            # light coords in the confocal mode will be handled differently
            light_coord = (0, 0, 0)

        self.N = N
        self.num_depth_planes = num_depth_planes
        self.wall_size = wall_size
        self.num_freqs = O
        self.light_coord = light_coord
        self.is_confocal = is_confocal
        self.falloff = falloff
        self.use_pad = use_pad
        self.freq_range = freq_range

        grid_z = (torch.arange(0, self.num_depth_planes, dtype=torch.float) + 0.5) / self.num_depth_planes
        grid_z = grid_z * (z_max - z_min) + z_min

        ## kernel
        bin_resolution = bin_length / C
        frequencies = torch.arange(0, self.num_freqs, dtype=torch.float) / ((self.num_freqs - 1) * 2)
        frequencies = frequencies / bin_resolution
        frequencies = 2 * torch.pi * frequencies
        kernel = self._make_rsd_kernel(grid_z, frequencies)
        kernel = kernel.unsqueeze(0).unsqueeze(0)

        if self.freq_range is not None:
            freq_start, freq_end = self.freq_range
            kernel = kernel[:, :, freq_start:freq_end]

        self.register_buffer('kernel', kernel, persistent=False)

        ## phase term
        if self.is_confocal:
            self.phase_term = None
        else:
            phase_term = (grid_z / C).unsqueeze(0)
            phase_term = torch.exp(1j * frequencies.unsqueeze(1) * phase_term)
            phase_term = phase_term.view(1, 1, self.num_freqs, self.num_depth_planes, 1, 1)
            phase_term = torch.view_as_real(phase_term).contiguous()
            if self.freq_range is not None:
                freq_start, freq_end = self.freq_range
                phase_term = phase_term[:, :, freq_start:freq_end]
            self.register_buffer('phase_term', phase_term, persistent=False)

    def _make_rsd_kernel(self, grid_z, frequencies):
        grid_x = (torch.arange(0, self.N, dtype=torch.float) + 0.5) / self.N
        grid_y = (torch.arange(0, self.N, dtype=torch.float) + 0.5) / self.N
        grid_x = (grid_x - 0.5) * self.wall_size
        grid_y = (grid_y - 0.5) * self.wall_size * -1

        grid_x, grid_y, grid_z = torch.meshgrid([grid_z, grid_x, grid_y], indexing='ij')

        light_x, light_y, light_z = self.light_coord
        dists = (grid_x - light_x) ** 2 + (grid_y - light_y) ** 2 + (grid_z - light_z) ** 2
        dists = dists.sqrt()
        scale_term = 1

        if self.is_confocal:
            scale_term = 2

        dists = dists.unsqueeze(0)
        O = frequencies.size(0)
        frequencies = frequencies.view(O, 1, 1, 1)

        # kernel dim starts with [Z, X, Y]
        kernel = torch.exp(1j * frequencies / C * dists * scale_term)
        if self.falloff:
            kernel = kernel / dists
        if self.use_pad:
            kernel = F.pad(kernel, (0, self.N, 0, self.N))
        kernel = torch.fft.fftn(kernel, s=(-1, -1))
        kernel = torch.view_as_real(kernel)

        return kernel

    def forward(self, x_f):
        """
        :param x_f: input frequency domain signal (B, C, O, H, W) or real tensor with last dimension = 2

        """
        dtype = x_f.dtype
        if len(x_f.size()) == 6:
            x_f = torch.view_as_complex(x_f.contiguous().float())

        B, C, O, H, W = x_f.size()
        kernel = torch.view_as_complex(self.kernel.contiguous())
        phase_term = torch.view_as_complex(self.phase_term.contiguous()) if not self.is_confocal else None

        if self.use_pad:
            x_f = F.pad(x_f, (0, W, 0, H), mode='constant', value=0)
        x_f = torch.fft.fftn(x_f, s=(-1, -1))
        x_f = x_f.unsqueeze(3)  # adding depth dim

        volume = self._convolution(x_f, kernel, phase_term)
        if self.use_pad:
            h_start, w_start = H // 2, W // 2
            h_end, w_end = h_start + H, w_start + W
            volume = volume[..., h_start:h_end, w_start:w_end]  # B, C, H, W, D
        else:
            volume = torch.fft.ifftshift(volume, dim=(-3, -2))

        volume = volume.permute(0, 1, 3, 4, 2)
        volume = torch.view_as_real(volume)

        # heuristic term to adjust feature scale
        volume = volume / self.num_freqs

        return volume

    def _convolution(self, x_f, kernel, phase_term=None):
        volume_f = x_f * kernel  # B, C, O, D, H, W
        if phase_term is not None:
            volume_f = volume_f * phase_term
        volume_f = volume_f.sum(dim=2)
        volume = torch.fft.ifftn(volume_f, s=(-1, -1))

        return volume


def make_wave_conv_kernels(wavelength_params: torch.Tensor,
                           sampling_size: float, bin_length: float,
                           base_wavelength: float = -1,
                           num_cycles: int = 5, sigma=0.3, eps=1e-8):
    virtual_wavelength = wavelength_params * (2 * sampling_size)
    if base_wavelength > 0:
        base_wavelength = base_wavelength * (2 * sampling_size)
    else:
        base_wavelength = torch.max(virtual_wavelength).item()
    num_samples = int(round(num_cycles * base_wavelength / bin_length))
    if num_samples % 2 == 0:
        num_samples += 1
    cycles = num_samples * bin_length / (virtual_wavelength + eps)
    cycles = cycles.unsqueeze(-1)

    grid_k = torch.arange(num_samples, dtype=torch.float32, device=wavelength_params.device) + 1
    grid_k = grid_k / num_samples
    grid_k = grid_k.unsqueeze(0)
    sin_wave = torch.sin(2 * torch.pi * cycles * grid_k)
    cos_save = torch.cos(2 * torch.pi * cycles * grid_k)
    windows = []
    for i in range(wavelength_params.size(0)):
        if base_wavelength > 0:
            sample_per_wavelength = num_samples
        else:
            sample_per_wavelength = int(round(num_cycles * virtual_wavelength[i].item() / bin_length))
            if sample_per_wavelength % 2 == 0:
                sample_per_wavelength += 1

        window = make_gaussian_window(sample_per_wavelength, 1.0 / sigma, device=wavelength_params.device)
        pad = (num_samples - sample_per_wavelength) // 2
        window = F.pad(window, (pad, pad), mode='constant', value=0)
        windows.append(window)
        pass

    window = torch.stack(windows, dim=0)
    virtual_sin_wave = sin_wave * window
    virtual_cos_wave = cos_save * window
    wave_conv_kernel = torch.stack([virtual_cos_wave, virtual_sin_wave], dim=1)  # n_wave x 2 x K

    return wave_conv_kernel


def wave_kernel_to_frequency_domain(wave_conv_kernel, T, num_freqs=-1):
    K = wave_conv_kernel.size(-1)
    wave_conv_kernel = wave_conv_kernel.transpose(1, 2)  # 2 to the last
    if K < T:
        padding = T - K
        wave_conv_kernel = F.pad(wave_conv_kernel, (0, 0, 0, padding), mode='constant', value=0)

    if num_freqs <= 0:
        num_freqs = T // 2 + 1
    wave_conv_kernel = torch.view_as_complex(wave_conv_kernel)
    wave_conv_kernel = torch.fft.fft(wave_conv_kernel, dim=-1)[:, :num_freqs]
    wave_conv_kernel = torch.abs(wave_conv_kernel) / K

    return wave_conv_kernel


def compute_wave_freq_range(wave_kernel, threshold=0.1):
    active_mask = (wave_kernel > wave_kernel.max().item() * threshold)
    freq_indices = torch.nonzero(active_mask)
    freq_range = freq_indices.min().item(), freq_indices.max().item()
    return freq_range


def make_gaussian_window(N, alpha, device):
    N_half = (N - 1) / 2.0
    grid = torch.arange(N, dtype=torch.float32, device=device) - N_half
    window = torch.exp(-0.5 * (alpha * grid / N_half) ** 2)
    return window
