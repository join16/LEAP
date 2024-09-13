import os
from os import path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import h5py

from .base import NLOSBaseDataModule


class StanfordDataModule(NLOSBaseDataModule):

    def __init__(self, root_dir: str, raw_root_dir: str, exposure: int = 60):
        super().__init__('stanford')

        self.is_confocal = True
        self.wall_size = 2.0
        self.transient_shape = (512, 64)
        self.bin_resolution = 32

        self.data_dir = path.join(root_dir, f'exposure-{exposure}min')
        self.target_objects = ['bike', 'dragon', 'statue', 'teaser']
        self.exposure = exposure
        self.raw_root_dir = raw_root_dir  # original dataset dir from "http://www.computationalimaging.org/publications/nlos-fk/"

    def val_dataloader(self):
        dataset = StanfordDataset(self.data_dir, self.target_objects)
        return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int):
        transient = batch['transient'].unsqueeze(1)
        transient = transient / transient.max().item()
        full_transient = transient

        sample_spacing = self.wall_size / self.transient_shape[1]
        bin_resolution = self.bin_resolution
        transient, sample_spacing, bin_resolution = self.handle_sparse_sampling(transient,
                                                                                sample_spacing, bin_resolution)
        transient, crop_pad_ratio = self.handle_center_crop(transient)
        batch['transient'] = transient
        batch['metadata'] = {
            'sample_spacing': sample_spacing,
            'bin_resolution': bin_resolution,
            'T': transient.size(2),
            'crop_pad_ratio': crop_pad_ratio,
        }
        batch['object_name'] = self.target_objects[batch['idx'].item()]
        batch['clean_transient'] = full_transient
        return batch

    def preprocess(self):
        os.makedirs(self.data_dir, exist_ok=True)
        files = [x for x in os.listdir(self.data_dir) if x.endswith('.npy')]
        has_all_data = True
        for name in self.target_objects:
            if f'{name}.npy' not in files:
                has_all_data = False
                break
        if has_all_data:
            return

        ## start preprocessing
        print('start preprocessing Stanford dataset..')
        for name in tqdm(self.target_objects):
            transient = self._preprocess_one_transient(name)
            transient = torch.from_numpy(transient).unsqueeze(0).unsqueeze(0)
            transient = F.avg_pool3d(transient, kernel_size=(1, 2, 2))
            transient = transient.squeeze(0).squeeze(0)
            transient = transient.permute(0, 2, 1)
            transient = transient[..., 2::4, 2::4].contiguous()
            transient = transient.cpu().numpy()
            np.save(path.join(self.data_dir, f'{name}.npy'), transient)
        print('preprocessing finished.')

    def _preprocess_one_transient(self, name):
        object_dir = path.join(self.raw_root_dir, name)
        with h5py.File(path.join(object_dir, f'meas_{self.exposure}min.mat'), 'r') as file:
            transient = file['meas'][:]
        with h5py.File(path.join(object_dir, 'tof.mat'), 'r') as file:
            tof = file['tofgrid'][:]

        tof = np.floor(tof / self.bin_resolution).astype(np.int32)

        T, H, W = transient.shape
        crop_size = self.transient_shape[0]
        cropped = np.zeros((crop_size, H, W), dtype=transient.dtype)
        for h in range(H):
            for w in range(W):
                start = int(tof[h, w])
                end = min(start + crop_size, T - 1)
                cropped_length = end - start
                cropped[:cropped_length, h, w] = transient[start:end, h, w]

        return cropped[:crop_size]


class StanfordDataset(Dataset):

    def __init__(self, root_dir: str, object_names):
        super().__init__()
        self.root_dir = root_dir
        self.object_names = object_names

    def __len__(self):
        return len(self.object_names)

    def __getitem__(self, idx):
        file_path = path.join(self.root_dir, f'{self.object_names[idx]}.npy')
        transient = np.load(file_path)

        return {
            'idx': idx,
            'transient': torch.from_numpy(transient)
        }
