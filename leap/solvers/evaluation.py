import os
from os import path

import numpy as np
import cv2

import engine
from .base import BaseSolver
from leap.data import NLOSBaseDataModule


class EvaluationSolver(BaseSolver):

    def __init__(self, dm: NLOSBaseDataModule, solver: BaseSolver):
        super().__init__()

        self.dm = dm
        self.solver = solver

        self.output_dir = engine.to_experiment_dir('outputs', self.dm.name)
        os.makedirs(self.output_dir, exist_ok=True)

    def validation_step(self, batch, batch_idx):
        outputs = self.solver(batch)
        front = outputs['front']
        object_name = batch['object_name']

        self._save_img(object_name, front[0])

    def _save_img(self, filename, x):
        img = (x.cpu().numpy() * 255).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_HOT)
        cv2.imwrite(path.join(self.output_dir, f'{filename}.png'), img)
