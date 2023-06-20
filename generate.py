# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import json
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from pathlib import Path
from torch_utils import distributed as dist
from training.sampler import edm_sampler, StackedRandomGenerator

#----------------------------------------------------------------------------
# Generalized ablation sampler, representing the superset of all sampling
# methods discussed in the paper.

def ablation_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='heun', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.1, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.1, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == 'heun'
            denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels).to(torch.float64)
            d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

    return x_next

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=False)
@click.option('--data_path',               help='Optionally use different data path to that used in training.',     type=str, default=None)
@click.option('--load_obs_from',           help='Path from which to load previously-sampled observes.',             type=str, default=None)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--intermediates', 'plot_intermediates', help='Plot intermediate state',                              is_flag=True)
@click.option('--invert_dataset_transform', help='Invert normalisation/PCA before saving.',                         type=bool, default=False)
@click.option('--increment_seeds',         help='Increment seed. Useful for drawing multiple samples with same observations but different seed.', type=int, default=0)
@click.option('--parallel_id',         help='Identifier for "rank" so that we can run simultaneous copies of this script and have them generate the images in different orders to avoid clashes.', type=int, default=0)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)

@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))

def main(network_pkl, outdir, data_path, load_obs_from, subdirs, seeds, class_idx, max_batch_size, device=torch.device('cuda'), plot_intermediates=False, invert_dataset_transform=False, increment_seeds=0, parallel_id=0, **sampler_kwargs):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Generate 64 images and save them as out/*.png
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    """
    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    if parallel_id == 0:
        pass
    elif parallel_id == 1:
        rank_batches = rank_batches[::-1]
    elif parallel_id >= 2:
        perm = torch.randperm(len(rank_batches))
        rank_batches = [rank_batches[i] for i in perm]

    # Infer outdir
    if outdir is None:
        print('outdir not specified, inferring outdir from network path')
        runs_dir, wandb_id, fname = network_pkl.split('/')
        sample_str = '_'.join(f'{k}-{v}' for k, v in sorted({**sampler_kwargs, 'class': class_idx}.items())) + ('_inverse_transformed'  if invert_dataset_transform else '') + (f'_{increment_seeds}' if increment_seeds > 0 else '')
        outdir = Path('results') / Path(wandb_id) / Path(fname).stem / Path(sample_str)
        if load_obs_from is not None:
            outdir = outdir / '_'.join(Path(load_obs_from).parts)
        outdir = outdir / Path(f'samples-{seeds[0]}-{seeds[-1]}')

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        data = pickle.load(f)
        net = data['ema']
        embedder_state_dict = data['embedder'] if 'embedder' in data else None
    # -- below is required to avoid bug which I think is due to model calling isinstance(...) on its layers -------
    from training.networks import EDMPrecond
    from torch_utils import misc

    init_kwargs = net.init_kwargs
    if 'pretrained_weights' in init_kwargs:
        init_kwargs = {k: v for k, v in init_kwargs.items() if k != 'pretrained_weights'}
    new_net = EDMPrecond(*net.init_args, **init_kwargs)
    misc.copy_params_and_buffers(net, new_net, require_all=True)
    net = new_net
    net = net.to(device).eval().requires_grad_(False)
    # ----------------------------------------------------------------------
    structure = net.structure

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Construct dataset obj.
    training_options_path = os.path.join(os.path.dirname(network_pkl), 'training_options.json')
    training_options = json.load(open(training_options_path, 'r'))
    dataset_kwargs = training_options['dataset_kwargs']
    if data_path is not None:
        dataset_kwargs['path'] = data_path
    if 'train_embedder' in dataset_kwargs:
        dataset_kwargs['train_embedder'] = False  # ensures gradients are not computed for embedder during sampling
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    if embedder_state_dict is not None:
        dataset_obj.load_network_state_dict(embedder_state_dict)

    image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
    os.makedirs(image_dir, exist_ok=True)
    existing_paths = set(os.path.join(image_dir, fname) for fname in os.listdir(image_dir))

    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Check if images already exist.
        image_paths = []
        saving_images = structure.exist[0]
        for seed in batch_seeds:
            ext = 'png' if saving_images else 'pt'
            image_path = os.path.join(image_dir, f'{seed:06d}.{ext}')
            image_paths.append(image_path)
        if all((path in existing_paths) for path in image_paths):
            print(f'All images already exist: {image_paths[0]} ... {image_paths[-1]}')
            continue

        # Pick latents and labels.
        if load_obs_from is None:
            rnd = StackedRandomGenerator(device, batch_seeds)
            indices = rnd.randint(len(dataset_obj), size=[batch_size, 1], device=device)
            rnd = StackedRandomGenerator(device, batch_seeds+increment_seeds)
            unstacked_data = [dataset_obj.__getitem__(i, will_augment=False) for i in indices]
        else:
            unstacked_data = [torch.load(os.path.join(load_obs_from, f'{i:06d}.pt')) for i in batch_seeds]  # use seeds as indices
            # increment batch seeds for use as seeds in case it is weird to use same seed for generating emb as for generating image given emb
            rnd = StackedRandomGenerator(device, [s+1+increment_seeds for s in batch_seeds])
        n_tensors = len(unstacked_data[0])
        data = tuple(torch.stack([datum[t] for datum in unstacked_data]).to(device) if unstacked_data[0][t] is not None else None for t in range(n_tensors))
        y = structure.flatten_obs(data, contains_marg=True)
        xT = rnd.randn([batch_size, structure.latent_dim], device=device)

        # Generate images.
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        have_ablation_kwargs = any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
        sampler_fn = ablation_sampler if have_ablation_kwargs else edm_sampler
        x0 = sampler_fn(net, xT, y, randn_like=rnd.randn_like, **sampler_kwargs, return_intermediates=plot_intermediates)
        data = structure.unflatten_batch(x0, y, pad_marg=True)

        # Save images.
        if structure.exist[0]:
            images_np = dataset_obj.get_images(data)
            for path, image_np in zip(image_paths, images_np):
                if image_np.shape[2] == 1:
                    PIL.Image.fromarray(image_np[:, :, 0], 'L').save(path)
                else:
                    PIL.Image.fromarray(image_np, 'RGB').save(path)
        else:
            # There are no images, save everything else instead.
            if invert_dataset_transform:
                data = dataset_obj.inverse_transform(data)  # undo normalisation or PCA
            for i, path in enumerate(image_paths):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                to_save = [None if t is None else t[i].cpu() for t in data]
                torch.save(to_save, path)
              
    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')




#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
