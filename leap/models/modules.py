import torch
from torch import nn


def pixel_shuffle_transients(x, r):
    if isinstance(r, int):
        r = [r, r, r]
    r_t, r_h, r_w = r
    B, C, T, H, W = x.size()

    C_new = C // (r_t * r_h * r_w)
    x = x.reshape(B, C_new, r_t, r_h, r_w, T, H, W)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
    x = x.reshape(B, C_new, T * r_t, H * r_h, W * r_w)
    return x


class UpsampleBlock(nn.Module):

    def __init__(self, input_dim, out_dim, upsample_size, activation, norm_layer,
                 add_block_coords=False, num_refine_blocks=1):
        super().__init__()

        self.upsample_size = upsample_size
        r_t, r_n = upsample_size
        upsample_dim = out_dim * r_t * (r_n ** 2)
        self.linear_up = nn.Conv3d(input_dim, upsample_dim, kernel_size=1)
        self.blocks = nn.ModuleList()
        for _ in range(num_refine_blocks):
            block = ResidualBlock3d(out_dim, out_dim, activation, norm_layer,
                                    add_block_coords=add_block_coords)
            self.blocks.append(block)
        pass

    def forward(self, x, metadata):
        x = self.linear_up(x)
        r_f, r_n = self.upsample_size
        x = pixel_shuffle_transients(x, (r_f, r_n, r_n))
        for block in self.blocks:
            x = block(x, metadata)
        return x


class ResidualBlock3d(nn.Module):

    def __init__(self, input_dim, out_dim, activation,
                 norm_layer=nn.BatchNorm3d,
                 conv_layer=nn.Conv3d,
                 downsample=None, add_block_coords=False):
        super().__init__()

        self.add_block_coords = add_block_coords
        self.out_dim = out_dim
        self.downsample = downsample
        kernel_size, padding, stride = 3, 1, 1
        if downsample is not None:
            downsample = tuple([downsample] * 3) if isinstance(downsample, int) else downsample
            kernel_size = max([k * 2 - 1 for k in downsample])
            padding = (kernel_size - 1) // 2
            stride = downsample
            pass
        if self.add_block_coords:
            input_dim += 3
            pass

        self.conv = nn.Sequential(
            conv_layer(input_dim, out_dim, kernel_size=kernel_size, padding=padding, stride=stride),
            norm_layer(out_dim),
            activation,
            conv_layer(out_dim, out_dim, kernel_size=3, padding=1),
            norm_layer(out_dim),
        )

        self.shortcut = nn.Sequential(
            conv_layer(input_dim, out_dim, kernel_size=1, padding=0, stride=stride),
            norm_layer(out_dim),
        )
        self.activation = activation
        pass

    def forward(self, x, metadata=None):
        if self.add_block_coords:
            pos = self._create_temporal_pos(x, metadata)
            x = torch.cat([x, pos], dim=1)
            pass
        identity = x
        x = self.conv(x)
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        x = self.activation(x + identity)
        return x

    def _create_temporal_pos(self, x, metadata):
        B, C, T, H, W = x.size()
        down_factor = metadata['T_input'] // T
        T_full = metadata['T'] / down_factor
        grids = torch.meshgrid([
            torch.arange(T, device=x.device, dtype=x.dtype),
            torch.arange(H, device=x.device, dtype=x.dtype) / H,
            torch.arange(W, device=x.device, dtype=x.dtype) / W,
        ])
        start_times = metadata['start_times'] / down_factor
        grids = [x.unsqueeze(0).expand(B, -1, -1, -1) for x in grids]
        grids[0] = (grids[0] + start_times.view(B, 1, 1, 1)) / T_full
        grids = torch.stack(grids, dim=1)

        return grids
