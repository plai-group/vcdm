# Code release for Visual Chain-of-Thought Diffusion Models [[Paper]](https://arxiv.org/abs/2303.16187) [[Blog]](https://plai.cs.ubc.ca/2023/06/17/visual-chain-of-thought-diffusion-models/)

![Images sampled with VCDM.](https://www.cs.ubc.ca/~wsgh/images/afhq-blog.gif)

Presented at [CVPR 2023's Generative Models for Computer Vision  workshop](https://generative-vision.github.io/workshop-CVPR-23/), [ICML 2023 Workshop on Structured Probabilistic Inference & Generative Modeling](https://spigmworkshop.github.io/)

This repository is based on [the code release](https://github.com/NVlabs/edm) for [Elucidating the design space of diffusion-based generative models](https://arxiv.org/abs/2206.00364). Thank you to the authors for making this accessible.

# Setting up

#### Python environment
We use Python 3.9.6, and can install the required packages in a fresh installation with:
```
pip install torch torchvision wandb tqdm matplotlib scikit-learn einops einops_exts git+https://github.com/openai/CLIP.git rotary-embedding-torch
```
Alternatively, see `requirements.txt` for a full specification of our requirements.

#### Environment variables for wandb logging
We log data from our training runs to [Weights & Biases](https://wandb.ai/). To set up logging, we use the following environment variables.
```
export WANDB_ENTITY="<YOUR WANDB USERNAME>"
export WANDB_PROJECT="<NAME OF PROJECT TO LOG TO>"
```
Optionally, if you are running on compute nodes without an internet connection, you can use `wandb offline` to deactivate logging (and optionally sync the logs later with `wandb sync`).

#### Download datasets
We dowload AFHQ, FFHQ, and ImageNet following the instructions in [the EDM repo](https://github.com/NVlabs/edm). For convenience we copy and paste them below:

**FFHQ:** Download the [Flickr-Faces-HQ dataset](https://github.com/NVlabs/ffhq-dataset) as 1024x1024 images and convert to ZIP archive at 64x64 resolution:

```.bash
python dataset_tool.py --source=downloads/ffhq/images1024x1024 \
    --dest=datasets/ffhq-64x64.zip --resolution=64x64
python fid.py ref --data=datasets/ffhq-64x64.zip --dest=fid-refs/ffhq-64x64.npz
```

**AFHQv2:** Download the updated [Animal Faces-HQ dataset](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq) (`afhq-v2-dataset`) and convert to ZIP archive at 64x64 resolution:

```.bash
python dataset_tool.py --source=downloads/afhqv2 \
    --dest=datasets/afhqv2-64x64.zip --resolution=64x64
python fid.py ref --data=datasets/afhqv2-64x64.zip --dest=fid-refs/afhqv2-64x64.npz
```

**ImageNet:** Download the [ImageNet Object Localization Challenge](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data) and convert to ZIP archive at 64x64 resolution:

```.bash
python dataset_tool.py --source=downloads/imagenet/ILSVRC/Data/CLS-LOC/train \
    --dest=datasets/imagenet-64x64.zip --resolution=64x64 --transform=center-crop
python fid.py ref --data=datasets/imagenet-64x64.zip --dest=fid-refs/imagenet-64x64.npz
```

# Example training commands

### AFHQ from scratch:
**Train model of CLIP embeddings (called *auxiliary model* in paper)**

Run on 1 GPU:
```
torchrun --standalone --nproc_per_node=1 train.py --path=datasets/afhqv2-64x64.zip --data_class CLIPDataset --batch 256 --seed 1 --exist 0,1 --observed 0,0 --arch=dalle2 --lr=1e-4 --augment=0.15 --pred_x0 1
```
**Train model of images given CLIP (called *conditional image model* in paper):**

Run on 4 GPUs (see the [torchrun documentation](https://pytorch.org/docs/stable/elastic/run.html) for help with parallel training):
```
torchrun --standalone --nproc_per_node=4 train.py --path=datasets/afhqv2-64x64.zip --data_class CLIPDataset --batch 128 --seed 1 --exist 1,1 --observed 0,1 --arch=ddpmpp --cres=1,2,2,2 --lr=1e-4 --dropout=0.05 --augment=0.15
```

### FFHQ from pretrained
**Downloading pretrained models**

We use pretrained models from [the EDM repo](https://github.com/NVlabs/edm). To initialize FFHQ from a pretrained model, we specifically download https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl with:
```
mkdir ./pretrained/
wget https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-ffhq-64x64-uncond-vp.pkl ./pretrained/
```
**Train model of CLIP embeddings (called *auxiliary model* in paper)**

Run on 1 GPU:
```
torchrun --standalone --nproc_per_node=1 train.py --path=datasets/ffhq-64x64.zip --data_class CLIPDataset --batch 256 --seed 1 --exist 0,1 --observed 0,0 --arch=dalle2 --lr=1e-4 --augment=0.15 --pred_x0 1
```
**Train model of images given CLIP, finetuned from EDM's unconditional model (called *conditional image model* in paper):**

Run on 4 GPUs (see the [torchrun documentation](https://pytorch.org/docs/stable/elastic/run.html) for help with parallel training):
```
torchrun --standalone --nproc_per_node=4 train.py --path=datasets/ffhq-64x64.zip --data_class CLIPDataset --batch 128 --seed 1 --exist 1,1 --observed 0,1 --arch=ddpmpp --cres=1,2,2,2 --lr=1e-4 --dropout=0.05 --augment=0.15 --pretrained_weights pretrained/edm-ffhq-64x64-uncond-vp.pkl
```

# Sampling and evaluating trained models
After training a model, you can find the saved checkpoint at the path `training-runs/<WANDB ID>/network-snapshot-<KIMG>.pkl`, where `<WANDB ID>` is a randomly-generated identifier used by the run and its logs on [Weights & Biases](https://wandb.ai/) and `<KIMG>` is the training progress, measured by how many thousands of training images have been seen. We sample from VCDM by first sampling CLIP embeddings with
```
torchrun --standalone --nproc_per_node=1 generate.py --seeds=0-19999 --batch=64 --network=training-runs/<AUXILIARY MODEL WANDB ID>/network-snapshot-<AUXILIARY MODEL KIMG>.pkl --steps 40 --S_churn 50 --S_noise 1.007
```
This samples CLIP embeddings and saves them to `results/<AUXILIARY MODEL WANDB ID>/network-snapshot-<AUXILIARY MODEL KIMG>/S_churn-50.0_S_max-inf_S_min-0.0_S_noise-1.007_class-None_discretization-None_num_steps-40_rho-7.0_scaling-None_schedule-None_sigma_max-None_sigma_min-None_solver-None/samples-0-19999/`. We then generate images given these CLIP embeddings with:

```
torchrun --standalone --nproc_per_node=1 generate.py --seeds=0-19999 --batch=64 --network=training-runs/<CONDITIONAL IMAGE MODEL WANDB ID>/network-snapshot-<CONDITIONAL IMAGE MODEL KIMG>.pkl --steps 40 --S_churn 50 --S_noise 1.007 --load_obs_from results/<AUXILIARY MODEL WANDB ID>/network-snapshot-<AUXILIARY MODEL KIMG>/S_churn-50.0_S_max-inf_S_min-0.0_S_noise-1.007_class-None_discretization-None_num_steps-40_rho-7.0_scaling-None_schedule-None_sigma_max-None_sigma_min-None_solver-None/samples-0-19999/
```
The images will be saved to `results/<CONDITIONAL IMAGE MODEL WANDB ID>/network-snapshot-<CONDITIONAL IMAGE MODEL KIMG>/S_churn-50.0_S_max-inf_S_min-0.0_S_noise-1.007_class-None_discretization-None_num_steps-40_rho-7.0_scaling-None_schedule-None_sigma_max-None_sigma_min-None_solver-None/results_<CONDITIONAL IMAGE MODEL WANDB ID>_network-snapshot-<AUXILIARY MODEL KIMG>_S_churn-50.0_S_max-inf_S_min-0.0_S_noise-1.007_class-None_discretization-None_num_steps-40_rho-7.0_scaling-None_schedule-None_sigma_max-None_sigma_min-None_solver-None_samples-0-19999/samples-0-19999`.

For evaluation, we use the `fid.py` script similarly to how it is described in [the EDM repo](https://github.com/NVlabs/edm). E.g., for FFHQ, we download `https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/ffhq-64x64.npz` to `./fid-refs/` and then run
```
torchrun --standalone --nproc_per_node=1 fid.py calc --images=<SAMPLED IMAGES DIR> --ref=fid-refs/ffhq-64x64.npz --num 20000
```

# Results directory structure

Following [this repo](https://github.com/wsgharvey/video-diffusion), we sampled images and CLIP embeddings are saved with the following directory structure:
```bash
results
├── <wandb_id>
│   ├── <checkpoint_name>
│   │   ├── <sample_args>
│   │   │   ├── samples-[n1]-[n2]
│   │   │   │  ├── [n1].png
│   │   │   │  ├── ...
│   │   │   │  └── [n2].png
│   │   │   ├── samples-[n3]-[n4]
│   │   │   │  └── ...
│   │   │   └── ...
│   │   └── ... (samples from same checkpoint with different sampler arguments)
│   └── ... (other checkpoints of the same run)
└── ... (other wandb runs)
```

# Citation

```
@article{harvey2023visual,
  title={Visual Chain-of-Thought Diffusion Models},
  author={Harvey, William and Wood, Frank},
  journal={arXiv preprint arXiv:2303.16187},
  year={2023}
}
```
