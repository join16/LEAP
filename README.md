# LEAP: Learning to Enhance Aperture Phasor Field for Non-Line-of-Sight Imaging
In Cho, Hyunbo Shim, [Seon Joo Kim](https://sites.google.com/site/seonjookim/)

[[`arXiv`](https://arxiv.org/abs/2407.18574)] [[`Project`](https://join16.github.io/leap-page/)] [[`BibTeX`](#Citation)]

This is an official implementation of the paper [Learning to Enhance Aperture Phasor Field for Non-Line-of-Sight Imaging](https://arxiv.org/abs/2407.18574).

## Installation
### Using docker
We provide a prebuilt [docker image](), which contains all the dependencies required to run the code.
```bash
docker pull join16/join16/nlos-leap:py39-cu113
```
You can also build your own docker image by running the following command.
```bash
docker build -t nlos-leap:py39-cu113 .
```
### Using pip
You can install the required dependencies using pip. We recommend using a virtual environment to avoid conflicts with other packages.
```bash
pip install -r requirements.txt
```
We tested our code on Python 3.9, torch 2.0.1, and CUDA 11.3.

## Dataset
### Synthetic dataset 
To generate synthetic datasets, we reproduce the NLOS renderer provided by [LFE](https://github.com/princeton-computational-imaging/NLOSFeatureEmbeddings) with headless rendering and multi GPU (multi-process) support.
Our reproduced renderer will be released soon.
You can alternatively use the original renderer to generate synthetic datasets.

### Real dataset
We use the Stanford dataset provided by [FK](https://github.com/computational-imaging/nlos-fk).
Download the original data, and modify `raw_root_dir` in the `config/data/stanford.yaml` to the path of the downloaded data.
Our evaluation script will automatically preprocess the data.

## Training
To train the model, run the following command.
```bash
python train.py config/train_n16.yaml
```
Available command line arguments:
- `--name`, `-n`: name of the experiment. Logs and checkpoints will be saved in `logs/{name}`.
- `--gpus`, `-g`: GPUs to use. This follows the pytorch-lightning style. Examples:  "-1" (all), "2" (2 GPU), "0,1" (GPU id 0, 1), "[0]" (GPU id 0)
- `--debug`, `-d`: Running in debug mode (run one step with a single GPU).

## Evaluation
Once the model is trained, you can evaluate the model using the following command.
```bash
python3 evaluate.py logs/{experiment_name}
```

## Acknowledgements
Our code is built upon the [Pytorch Lightning](https://lightning.ai/docs/pytorch/stable/).
We sincerely appreciate the authors for sharing their code and data, which greatly helped our research.
Our RSD implementation is based on the [original MATLAB code](https://www.nature.com/articles/s41467-020-15157-4) and the [pytorch implementation](https://github.com/fmu2/nlos3d).

## <a name="citation"></a> Citation
```BibTex
@article{cho2024leap,
  author    = {Cho, In and Shim, Hyunbo and Kim, Seon Joo},
  title     = {Learning to Enhance Aperture Phasor Field for Non-Line-of-Sight Imaging},
  journal   = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2024},
}
```