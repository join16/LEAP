from os import path
from argparse import ArgumentParser

from omegaconf import OmegaConf
import lightning.pytorch as pl

import engine

parser = ArgumentParser('NLOS LEAP training script')
parser.add_argument('config', type=str, help='Path to the config file')
parser.add_argument('--name', '-n', type=str, default=None, help='Name of the experiment')
parser.add_argument('--debug', '-d', action='store_true', default=False, help='debug mode (for sanity check)')
parser.add_argument('--gpus', '-g', default='-1',
                    help='GPU to use (num. GPU or gpu ids, follow pytorch-lightning convention). e.g., "-1" (all), "2" (2 GPU), "0,1" (GPU id 0, 1), "[0]" (GPU id 0)')


def main():
    args = parser.parse_args()
    cfg = engine.load_config(args.config)
    pl.seed_everything(cfg.get('seed', 123456))
    if args.debug:
        args.name = 'debugging'
    engine.create_experiment_context(cfg.get('output_dir', None), args.name)
    with open(path.join(engine.to_experiment_dir('config.yaml')), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    model = engine.instantiate(cfg.model)
    dm = engine.instantiate(cfg.data)
    solver = engine.instantiate(cfg.solver, dm=dm, model=model)

    trainer: pl.Trainer = engine.prepare_trainer(cfg, gpus=args.gpus, debug=args.debug)
    trainer.fit(model=solver, datamodule=dm)


if __name__ == '__main__':
    main()
