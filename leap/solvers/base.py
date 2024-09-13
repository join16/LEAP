import os
import functools
import math
from typing import List

import torch
from torch import nn
from lightning.pytorch import LightningModule, Callback
from torchmetrics import Metric

import engine

class BaseSolver(LightningModule):

    def __init__(self):
        super().__init__()
        self._best_score = None
        self._additional_callbacks: List[Callback] = [_DefaultTaskCallback()]
        self._metrics = None
        self.checkpoint_epoch = -1
        self.output_dir = engine.to_experiment_dir('outputs')
        os.makedirs(self.output_dir, exist_ok=True)

    def configure_callbacks(self):
        return self._additional_callbacks

    def add_callback(self, callback: Callback):
        self._additional_callbacks.append(callback)

    def reset_parameters(self):
        init_fn = functools.partial(nn.init.kaiming_uniform_, a=math.sqrt(5))
        for m in self.modules():
            if m == self:
                continue
            elif hasattr(m, 'skip_reset_parameters') and m.skip_reset_parameters:
                continue
            elif hasattr(m, 'reset_parameters'):
                m.reset_parameters()
                continue
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                init_fn(m.weight)
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @property
    def best_score(self):
        return self._best_score

    def track_score(self, score: any):
        if (self._best_score is None) or score > self._best_score:
            self._best_score = score

        self.log('score', score, rank_zero_only=True, on_epoch=True, sync_dist=True)
        self.log('best', self._best_score, rank_zero_only=True, on_epoch=True, sync_dist=True)

    def restore_checkpoint(self, checkpoint_path: str):
        print(f'checkpoint loaded: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['state_dict']
        self.load_state_dict(state_dict, strict=False)
        self.checkpoint_epoch = checkpoint['epoch']

    def reset_metrics(self):
        if self._metrics is None:
            self._metrics = [value for value in self.modules() if isinstance(value, Metric)]

        for metric in self._metrics:
            metric.reset()


class _DefaultTaskCallback(Callback):
    def on_sanity_check_end(self, trainer, solver):
        if not isinstance(solver, BaseSolver):
            return
        solver._best_score = None

    def on_validation_epoch_start(self, trainer, solver) -> None:
        if not isinstance(solver, BaseSolver):
            return
        solver.reset_metrics()
