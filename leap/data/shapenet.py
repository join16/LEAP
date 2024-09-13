from os import path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from lightning.pytorch import LightningDataModule
import numpy as np

from .base import NLOSBaseDataModule
from leap.data.hdf5_dataset import HDF5Dataset


class ShapeNetDataModule(NLOSBaseDataModule):

    def __init__(self,
                 root_dir: str,
                 batch_size: int,
                 num_workers: int,
                 eval_batch_size: int = -1,

                 # scanning related params
                 is_confocal: bool = True,
                 transient_shape: tuple = (512, 64),  # (T, N)
                 wall_size: float = 2.0,
                 bin_resolution: float = 32,  # in ps

                 subsample: tuple = (1, 1),
                 center_crop_size: int = -1,

                 ### noise parameters
                 noise_b_range: tuple = (0.1, 0.2),
                 noise_c_range: tuple = (0.1, 1),
                 noise_p_range: tuple = (0, 0.2),
                 noise_b_sigma: float = 3,
                 noise_c_sigma: float = 0.1,
                 ):
        super().__init__('shapenet')

        self.is_confocal = is_confocal
        self.wall_size = wall_size
        self.transient_shape = transient_shape
        self.bin_resolution = bin_resolution
        self.subsample = subsample
        self.center_crop_size = center_crop_size

        self.root_dir = root_dir
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers

        ### noise parameters
        self.noise_b_range = noise_b_range
        self.noise_c_range = noise_c_range
        self.noise_p_range = noise_p_range
        self.noise_b_sigma = noise_b_sigma
        self.noise_c_sigma = noise_c_sigma

    def train_dataloader(self):
        dataset = self._get_dataset(is_train=True)
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          collate_fn=transient_collate_fn)

    def val_dataloader(self):
        batch_size = self.eval_batch_size if self.eval_batch_size > 0 else self.batch_size
        dataset = self._get_dataset(is_train=False)
        return DataLoader(dataset, batch_size=batch_size, num_workers=self.num_workers,
                          collate_fn=transient_collate_fn)

    def _get_dataset(self, is_train: bool = True):
        filename = 'train.h5' if is_train else 'val.h5'
        return NLOSShapeNetDataset(self.root_dir, filename, use_augmentation=is_train)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int):
        transient = self._sparse_to_dense_transient(batch)
        T_full = transient.size(1)

        sample_spacing = self.wall_size / self.transient_shape[1]
        bin_resolution = self.bin_resolution

        # ensure avg. min photon counts of the entire measurement (Supplement B.1)
        transient = transient * self._compute_min_photon_scale(transient)
        transient = transient.unsqueeze(1)
        clean_transient = transient.clone()

        transient, sample_spacing, bin_resolution = self.handle_sparse_sampling(transient,
                                                                                sample_spacing, bin_resolution)

        ### adding simulated noise
        noise_c, noise_b, noise_p = self._sample_noise(transient)
        transient = transient + noise_b  # background noise
        transient = transient * (1 - noise_p)  # simulating randomly missing photons
        transient = transient * noise_c  # exposure term
        transient = torch.poisson(transient) / noise_c

        transient, crop_pad_ratio = self.handle_center_crop(transient)

        ### normalize
        transient = self._max_normalize(transient)
        clean_transient = self._max_normalize(clean_transient)

        batch.pop('transient_indices')
        batch['transient'] = transient
        batch['clean_transient'] = clean_transient
        batch['metadata'] = {
            'sample_spacing': sample_spacing,
            'bin_resolution': bin_resolution,
            'T': T_full,
            'crop_pad_ratio': crop_pad_ratio,
        }
        return batch

    def _max_normalize(self, transient):
        B = transient.size(0)
        return transient / transient.view(B, -1).max(dim=-1)[0].view(B, 1, 1, 1, 1)

    def _compute_min_photon_scale(self, transient):
        B = transient.size(0)
        photon_target_val = 100
        top_val = torch.topk(transient.view(B, -1), k=10000, dim=-1)[0].mean(dim=-1)
        scale = photon_target_val / top_val.clamp_min(1)
        scale = scale.view(B, 1, 1, 1)
        mask = top_val < photon_target_val
        scale[mask.view(B, 1, 1, 1)] = 1

        return scale

    def _sparse_to_dense_transient(self, batch):
        ## gather sparse transient values to dense tensor
        B = batch['batch_size']
        T, N = self.transient_shape
        transient = torch.zeros(B, T, N, N, device=batch['transient'].device, dtype=batch['transient'].dtype)
        b = batch['batch_indices'].long()
        indices = batch['transient_indices'].long()
        values = batch['transient']
        t, h, w = indices[:, 0], indices[:, 1], indices[:, 2]
        valid_mask = t < T
        if valid_mask.float().sum().item() < valid_mask.size(0):
            b = b[valid_mask]
            t = t[valid_mask]
            h = h[valid_mask]
            w = w[valid_mask]
        b.clamp_(0, B - 1)
        t.clamp_(0, T - 1)
        h.clamp_(0, N - 1)
        w.clamp_(0, N - 1)
        transient[b, t, h, w] = values

        return transient

    def _sample_noise(self, transient):
        B = transient.size(0)
        noise_c = torch.rand(B, device=transient.device)
        noise_c = self._random_values_in_range(noise_c, self.noise_c_range)
        noise_c = noise_c.view(B, 1, 1, 1, 1)

        noise_b_base = torch.rand(B, device=transient.device)
        noise_b_base = self._random_values_in_range(noise_b_base, self.noise_b_range)
        noise_b_base = noise_b_base * transient.view(B, -1).max(dim=-1)[0]
        noise_b_base = noise_b_base.view(B, 1, 1, 1, 1)
        noise_b = torch.rand_like(transient)
        noise_b = self._random_values_in_range(noise_b, (-self.noise_b_sigma, self.noise_b_sigma))
        noise_b = noise_b_base + noise_b

        noise_p = torch.rand_like(transient)
        noise_p = self._random_values_in_range(noise_p, self.noise_p_range)

        return noise_c, noise_b, noise_p

    def _random_values_in_range(self, random_vals, ranges):
        val_min, val_max = ranges
        return random_vals * (val_max - val_min) + val_min


class NLOSShapeNetDataset(HDF5Dataset):

    def __init__(self, root_dir: str, filename: str, use_augmentation: bool = False):
        file_path = path.join(root_dir, filename)
        super().__init__(file_path)

        self.class_names = list(self.file.keys())
        self.object_names = self._get_all_object_names()
        self.num_items = len(self.object_names)
        self.use_augmentation = use_augmentation

        self.close_file()

    def __len__(self):
        return len(self.object_names)

    def __getitem__(self, idx):
        object_name = self.object_names[idx]
        group = self.file[object_name]

        sample_names = list(group.keys())
        if self.use_augmentation:
            sample_idx = np.random.randint(len(sample_names))
            sample_name = sample_names[sample_idx]
        else:
            sample_idx = 0
            sample_name = 'sample_0'

        group = group[sample_name]
        transient, transient_indices = self._get_transient(group)
        transient = torch.from_numpy(transient).float()  # N
        transient_indices = torch.from_numpy(transient_indices)

        item = {
            'idx': idx,
            'sample_idx': sample_idx,
            'transient': transient,
            'transient_indices': transient_indices,
        }
        if 'depth' in group:
            depth = group['depth'][:]
            depth = torch.from_numpy(depth).float()
            item['depth'] = depth

        return item

    def _get_all_object_names(self):
        all_object_names = []
        for cls_name in self.file.keys():
            all_object_names.extend([f'{cls_name}/{x}' for x in self.file[cls_name].keys()])

        object_names = []
        for object_name in all_object_names:
            group = self.file[object_name]
            sample_names = list(group.keys())
            if len(sample_names) == 0:
                continue

            object_names.append(object_name)

        return object_names

    def _get_transient(self, group):
        base_key = 'transient'
        indices = group[f'{base_key}_pos'][:].copy().astype(np.int32)
        indices = indices.transpose((1, 0))

        transients = group[f'{base_key}_values']
        transients = transients[:].copy().astype(np.float32)

        return transients, indices


def transient_collate_fn(items):
    ### merging sparse transients
    total = sum([x['transient'].size(0) for x in items])
    transients = torch.zeros(total, dtype=torch.float32)
    transient_indices = torch.zeros(total, 3, dtype=torch.int)
    batch_indices = torch.zeros(total, dtype=torch.int)

    last_idx = 0
    for i, item in enumerate(items):
        num_transient = item['transient_indices'].size(0)
        next_idx = last_idx + num_transient

        transients[last_idx:next_idx] = item['transient']
        transient_indices[last_idx:next_idx] = item['transient_indices']
        batch_indices[last_idx:next_idx] = i
        last_idx = next_idx

        del item['transient']
        del item['transient_indices']
        pass

    batch = default_collate(items)

    batch['transient'] = transients
    batch['transient_indices'] = transient_indices
    batch['batch_indices'] = batch_indices
    batch['batch_size'] = len(items)

    return batch
