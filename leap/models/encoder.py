import torch
from torch import nn

from .modules import ResidualBlock3d


class EnhancementEncoderCNN(nn.Module):

    def __init__(self, input_dim, activation, input_size,
                 blocks=((64, 2),
                         (128, (2, 1)),
                         (256, (2, 1)),),
                 add_block_coords=True,
                 ):
        super().__init__()

        self.input_dim = input_dim

        norm_layer = nn.BatchNorm3d
        last_dim = input_dim
        mid_t, mid_n = input_size
        self.blocks = nn.ModuleList()
        self.tail_layers = nn.ModuleList()
        for block_dim, downsample in blocks:
            if downsample == 0:
                downsample = None
            if isinstance(downsample, int):
                downsample = (downsample, downsample)
            downsample_size = None
            if downsample is not None:
                down_t, down_n = downsample
                downsample_size = (down_t, down_n, down_n)
                mid_t = mid_t // down_t
                mid_n = mid_n // down_n

            block = ResidualBlock3d(last_dim, block_dim, activation, norm_layer,
                                    downsample=downsample_size,
                                    add_block_coords=add_block_coords)
            self.blocks.append(block)
            last_dim = block_dim

        self.out_dim = last_dim
        pass

    def forward(self, x, metadata):
        for idx, block in enumerate(self.blocks):
            x = block(x, metadata)

        return x
