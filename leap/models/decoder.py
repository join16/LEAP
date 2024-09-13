import torch
from torch import nn
from torch.nn import functional as F

from .modules import UpsampleBlock


class EnhancementDecoderCNN(nn.Module):

    def __init__(self,
                 input_dim,
                 activation,
                 blocks=((64, (4, 2, 2), 2), (64, (4, 2, 2), 2)),
                 add_block_coords=True,
                 ):
        super().__init__()
        norm_layer = nn.BatchNorm3d
        self.blocks = nn.ModuleList()
        for (out_dim, upsample_size, num_refine_blocks) in blocks:
            block = UpsampleBlock(input_dim, out_dim, upsample_size, activation, norm_layer,
                                  add_block_coords=add_block_coords,
                                  num_refine_blocks=num_refine_blocks)
            self.blocks.append(block)
            input_dim = out_dim
        self.out_dim = input_dim

    def forward(self, x, metadata):
        for block in self.blocks:
            x = block(x, metadata)
        return x
