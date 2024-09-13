from torch.nn import functional as F
from lightning.pytorch import LightningDataModule


class NLOSBaseDataModule(LightningDataModule):
    is_confocal: bool = True
    transient_shape: tuple = (512, 64)
    wall_size: float = 2.0
    bin_resolution: float = 32  # in ps
    subsample: tuple = (1, 1)
    center_crop_size: int = -1
    name: str = 'none'

    def __init__(self, name):
        super().__init__()
        self.name = name

    def preprocess(self):
        pass

    def set_subsampling(self, subsample: tuple, center_crop_size: int):
        self.subsample = subsample
        self.center_crop_size = center_crop_size

    def handle_sparse_sampling(self, transient, sample_spacing: float, bin_resolution: float):
        if self.subsample is None:
            return transient, sample_spacing, bin_resolution

        ### subsampling
        down_T, down_N = self.subsample
        if down_T > 1:
            transient = F.avg_pool3d(transient, (down_T, 1, 1), stride=(down_T, 1, 1))
            bin_resolution *= down_T
        if down_N > 1:
            transient = transient[..., ::down_N, ::down_N]
            sample_spacing *= down_N

        transient = transient.contiguous()
        return transient, sample_spacing, bin_resolution

    def handle_center_crop(self, transient):
        if self.center_crop_size < 0:
            return transient, -1

        crop_pad_ratio = transient.size(-1) / self.center_crop_size
        half_size = self.center_crop_size // 2
        center = transient.size(-1) // 2
        start, end = center - half_size, center + half_size
        transient = transient[..., start:end, start:end].contiguous()
        return transient, crop_pad_ratio
