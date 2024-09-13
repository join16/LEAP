import torch
from torch.nn import functional as F


def compute_crop_start_times(transient, crop_window_size):
    B, C, T, H, W = transient.size()
    kernel_size = crop_window_size
    intensity_sums = transient.view(B, C, T, -1).sum(dim=-1)
    intensity_sums = F.avg_pool1d(intensity_sums, kernel_size=(kernel_size,), stride=1).squeeze(1)
    start_times = torch.argmax(intensity_sums, dim=-1)

    start_times.clamp_(0, T - crop_window_size - 1)
    return start_times


def crop_transient(transient, start_times, crop_window_size):
    B, C, T, H, W = transient.size()
    transient = transient.transpose(1, 2)
    indices = torch.arange(crop_window_size, device=transient.device).unsqueeze(0).repeat(B, 1)
    indices += start_times.unsqueeze(-1)

    pad_t = indices.max().item() - T + 2
    if pad_t > 0:
        transient = torch.cat([transient, torch.zeros_like(transient[:, :pad_t])], dim=1)

    indices.clamp_(0, transient.size(1) - 1)
    b_indices = torch.arange(B, device=transient.device).unsqueeze(-1)
    cropped_transient = transient[b_indices, indices]
    cropped_transient = cropped_transient[:, :crop_window_size]
    cropped_transient = cropped_transient.transpose(1, 2).contiguous()

    return cropped_transient


def uncrop_transient(transient, start_times, T_target=512, down_factor=1):
    B, C, T, H, W = transient.size()
    if T == T_target:
        return transient
    if (start_times is None) or (start_times.sum() == 0):
        pad = T_target - T
        transient = F.pad(transient, (0, 0, 0, 0, 0, pad))
        return transient

    transient = transient.transpose(1, 2)
    start_times = (start_times / down_factor).floor().long()
    indices = torch.arange(T, device=transient.device).unsqueeze(0).repeat(B, 1) + start_times.unsqueeze(-1)
    T_max = max(indices.max().item() + 1, T_target)
    padded_transient = torch.zeros(B, T_max, C, H, W, device=transient.device, dtype=transient.dtype)
    b_indices = torch.arange(B, device=transient.device).unsqueeze(-1)

    padded_transient[b_indices, indices] = transient
    padded_transient = padded_transient[:, :T_target]
    padded_transient = padded_transient.transpose(1, 2).contiguous()
    return padded_transient


def to_frequency_feature(x, freq_range, fft_norm=None):
    x_f = torch.fft.rfft(x.float(), dim=2, norm=fft_norm)
    freq_start, freq_end = freq_range

    x_f = x_f[:, :, freq_start:freq_end]
    x_f = torch.view_as_real(x_f)
    x_f = x_f.unsqueeze(1).transpose(1, -1).flatten(1, 2).contiguous().squeeze(-1)
    return x_f
