import functools

import torch
from torch import nn
from torch.nn import functional as F

from .waveconv import WaveConvolutionLayer
from .modules import ResidualBlock3d
from .encoder import EnhancementEncoderCNN
from .decoder import EnhancementDecoderCNN
from .utilis import compute_crop_start_times, crop_transient, uncrop_transient, to_frequency_feature
from .modules import pixel_shuffle_transients


class LEAP(nn.Module):

    def __init__(self,
                 input_size: int = 16,
                 output_size: int = 64,

                 stem_dim: int = 32,
                 wall_size: int = 2.,
                 T: int = 512,
                 crop_window_size: int = 256,

                 waveconv_params: dict = None,
                 encoder_params: dict = None,
                 decoder_params: dict = None,
                 ):
        waveconv_params = waveconv_params or {}
        encoder_params = encoder_params or {}
        decoder_params = decoder_params or {}
        activation = nn.GELU()

        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.sampling_size = wall_size / output_size
        self.crop_window_size = crop_window_size
        self.up_n = output_size // input_size
        self.T = T

        self.waveconv_layer = WaveConvolutionLayer(sampling_size=self.sampling_size, T=T,
                                                   **waveconv_params)
        self.stem = ResidualBlock3d(self.waveconv_layer.out_dim, stem_dim, activation=activation)
        self.encoder = EnhancementEncoderCNN(stem_dim, activation, (crop_window_size, input_size),
                                             **encoder_params)
        self.decoder = EnhancementDecoderCNN(self.encoder.out_dim, activation,
                                             **decoder_params)

        spatial_conv_3d = functools.partial(nn.Conv3d, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        last_dim = self.decoder.out_dim * 2
        out_dim = 2 * (self.up_n ** 2)

        self.last_layer = nn.Sequential(
            spatial_conv_3d(last_dim + 2, last_dim),
            activation,
            spatial_conv_3d(last_dim, last_dim),
            activation,
            nn.Conv3d(last_dim, out_dim, kernel_size=1, bias=False),
        )

    def forward(self, x, metadata):
        B = x.size(0)
        # removes unnecessary signals out of the crop window
        start_times = compute_crop_start_times(x, self.crop_window_size)
        x = crop_transient(x, start_times, self.crop_window_size)
        x = uncrop_transient(x, start_times, T_target=metadata['T'], down_factor=1)
        input_x = x

        x = self.waveconv_layer(x)
        x = crop_transient(x, start_times, self.crop_window_size)  # crop transients after input wave conv
        T_input = x.size(2)
        metadata['start_times'] = start_times
        metadata['T_input'] = x.size(2)

        x = self.stem(x)
        x = self.encoder(x, metadata)
        x = self.decoder(x, metadata)

        input_x = input_x / input_x.view(B, -1).max(dim=-1, keepdim=True)[0].view(B, 1, 1, 1, 1)
        input_x = to_frequency_feature(input_x, metadata['freq_range'])

        down_factor = T_input // x.size(2)
        x = uncrop_transient(x, metadata['start_times'], T_target=self.T, down_factor=down_factor)
        x_f = to_frequency_feature(x, metadata['freq_range'], fft_norm='ortho')
        input_x_up = self._pad_and_upsample(input_x, x_f.size(-1), metadata)
        x = torch.cat([x_f, input_x_up], dim=1)
        x = self.last_layer(x)

        out = pixel_shuffle_transients(x, (1, self.up_n, self.up_n))

        ## skip connection
        input_x_up = self._pad_and_upsample(input_x, out.size(-1), metadata)
        out = out + input_x_up

        return {
            'phasor': out,
        }

    def _pad_and_upsample(self, x, N_target, metadata):
        if metadata['crop_pad_ratio'] > 0:
            pad_ratio = metadata['crop_pad_ratio'] - 1
            pad_size = int(x.size(-1) * pad_ratio) // 2
            x = F.pad(x, (pad_size, pad_size, pad_size, pad_size, 0, 0), mode='constant', value=0)

        N_in = x.size(-1)
        if N_target > N_in:
            r = N_target // N_in
            x = F.interpolate(x, scale_factor=(1, r, r), mode='trilinear', align_corners=False)
        elif N_target < N_in:
            r = N_in // N_target
            x = F.avg_pool3d(x, kernel_size=(1, r, r), stride=(1, r, r))
        return x
