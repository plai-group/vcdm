# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import json
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training import training_loop
from training.dataset import datasets_to_kwargs, kwargs_gettable_from_dataset
import wandb

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides')  # False warning printed by PyTorch 1.12.

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    print('parse_int_list', s)
    if isinstance(s, tuple): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return tuple(ranges)

def parse_float_list(s):
    if isinstance(s, tuple): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(float(m.group(1)), float(m.group(2))+1))
        else:
            ranges.append(float(p))
    return tuple(ranges)

#----------------------------------------------------------------------------

def dataset_specific_options(f):
    # print(datasets_to_kwargs.values())
    dataset_specific_kwargs = set.union(*datasets_to_kwargs.values())
    for name, type_str, default in dataset_specific_kwargs:
        f = click.option(f'--{name}', help=f'{name} for dataset', type=eval(type_str), default=default, show_default=True)(f)
    return f

@click.command()

# Structural options (TODO remove duplication)
@click.option('--exist',          help='List of 1s/0s to specify which tensors to use (i.e. to not marginalise).', metavar='LIST',  type=parse_int_list, default=None, show_default=True)  # TODO implement for any dataset
@click.option('--observed',      help='Which dataset tensors are observed', metavar='LIST',         type=parse_int_list, required=True)
# Main options.
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, default='training-runs', show_default=True)
@click.option('--data_class',    help='Dataset class to use',                                       type=click.Choice(datasets_to_kwargs.keys()), required=True)
@click.option('--precond',       help='Preconditioning & loss function', metavar='vp|ve|edm',       type=click.Choice(['vp', 've', 'edm']), default='edm', show_default=True)
@click.option('--just_visualize', help='Whether to just visualize the dataset.', metavar='BOOL',    type=bool, default=False, show_default=True)
@click.option('--noise_mult',    help='Multipliers for amount of noise added to each tensor.', metavar='LIST', type=parse_float_list, default=(1.,))
@click.option('--equally_weight_tensors', help='Equally weight loss for each tensor.',            type=bool, default=False)
@click.option('--pretrained_weights', help='Path to pickle file to load initialization from.',            type=str, default=None)
@click.option('--freeze_pretrained', help='Whether to freeze pretrained weights.',            type=bool, default=False)
@click.option('--embedder_path', help='Ignored argument kept for compatibility.',            type=str, default=None)
# Architecture options.
@click.option('--arch',          help='Network architecture',                                       type=click.Choice(['concatunet', 'gsdm', 'tran', 'adm', 'control-adm', 'film-ddpmpp', 'ddpmpp', 'dalle2']), default='concatunet', show_default=True)
@click.option('--pred_x0',       help='List describing whether to predict x0 for each tensor.', metavar='LIST', type=parse_int_list, default=(0,))
@click.option('--softmax_onehot', help='Whether to use softmax on model output for onehots. Only makes sense with --pred_x0 on for the onehots.', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--channel_mult_emb', help='Channel multiplier for vector embeddings.', metavar='INT', type=int, default=4, show_default=True)
@click.option('--channel_mult_noise', help='Channel multiplier for initial noise (and label+vector) embeddings.', metavar='INT', type=int, default=4, show_default=True)
@click.option('--emb_layer_type', help='Architecture for embedding layers when they are modelled jointly.', type=click.Choice(['attn', 'shitty', 'attn-skip', 'shitty-skip']), default='shitty', show_default=True)
@click.option('--num_transformer_blocks', help='Number of blocks to use in transformer architecture.', metavar='INT', type=int, default=4, show_default=True)

@dataset_specific_options

# Hyperparameters.
@click.option('--duration',      help='Training duration', metavar='MIMG',                          type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',       type=int)
@click.option('--cres',          help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list)
@click.option('--lr',            help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=10e-4, show_default=True)
@click.option('--ema',           help='EMA half-life', metavar='MIMG',                              type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option('--dropout',       help='Dropout probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.13, show_default=True)
@click.option('--augment',       help='Augment probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.12, show_default=True)

# Performance-related.
@click.option('--fp16',          help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--ls',            help='Loss scaling', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=len(os.sched_getaffinity(0)), show_default=True)

# I/O-related.
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--tick',          help='How often to print progress', metavar='KIMG',                type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--snap',          help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--dump',          help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--sample',        help='How often to log images', metavar='TICKS',                   type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--transfer',      help='Transfer learning from network pickle', metavar='PKL|URL',   type=str)
@click.option('--resume',        help='Resume from previous training state', metavar='PT',          type=str)
@click.option('-n', '--dry-run', help='Print training options and exit',                            is_flag=True)

# wandb
@click.option('--wandb_dir',     help='Where to save the wandb results', metavar='DIR',             type=str, default='.')
@click.option('--resume_id',     help='Wandb id to resume from', metavar='ID',                      type=str, default=None)

def main(**kwargs):
    """Train diffusion-based generative model using the techniques described in the
    paper "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Train DDPM++ model for class-conditional CIFAR-10 using 8 GPUs
    torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \\
        --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp
    """
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn')
    dist.init()

    # Initialize config dict.
    c = dnnlib.EasyDict()
    dataset_class_name = 'training.dataset.' + opts.data_class
    c.dataset_kwargs = dnnlib.EasyDict(class_name=dataset_class_name)
    for kwarg_name, _, _ in datasets_to_kwargs[opts.data_class]:
        c.dataset_kwargs[kwarg_name] = opts[kwarg_name]
    if 'dataset_cache_dir' in c.dataset_kwargs and c.dataset_kwargs.dataset_cache_dir is None:
        data_fname = os.path.splitext(os.path.basename(opts.path))[0]
        c.dataset_kwargs.dataset_cache_dir = f"datasets/cache/{opts.data_class}_{data_fname}_{opts.seed}"

    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.loss_kwargs = dnnlib.EasyDict(noise_mult=opts.noise_mult, pred_x0=opts.pred_x0)
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9,0.999], eps=1e-8)
    c.just_visualize = opts.just_visualize
    c.equally_weight_tensors = opts.equally_weight_tensors
    c.freeze_pretrained = opts.freeze_pretrained

    # Validate dataset options.
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        print("Datset size: ", len(dataset_obj))
        dataset_name = dataset_obj.name
        for kwarg_name, getter in kwargs_gettable_from_dataset[opts.data_class]:
            # sets e.g. max_size and resolution for image datasets
            setattr(c.dataset_kwargs, kwarg_name, getter(dataset_obj))  # sets
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    c.structure_kwargs = dnnlib.EasyDict(exist=opts.exist, observed=opts.observed)

    # Network architecture.
    c.network_kwargs = dnnlib.EasyDict(pred_x0=opts.pred_x0, noise_mult=opts.noise_mult, pretrained_weights=opts.pretrained_weights)  # required by EDMPrecond
    if opts.arch == 'concatunet':
        c.network_kwargs.update(model_type='ConcatUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard',
                                resample_filter=[1,1], model_channels=128, channel_mult=[2,2,2])
        c.network_kwargs.update(channel_mult_emb=opts.channel_mult_emb, channel_mult_noise=opts.channel_mult_noise,
                                softmax_onehot=opts.softmax_onehot, emb_layer_type=opts.emb_layer_type)
    elif opts.arch == 'gsdm':
        c.network_kwargs.update(model_type='GraphicallyStructuredModel', model_channels=128, channel_mult_noise=opts.channel_mult_noise, num_blocks=opts.num_transformer_blocks,
                                softmax_onehot=opts.softmax_onehot)
        c.network_kwargs.update(augment_dim=-1)  # turn off augmentation, since we don't use it on GSDM-style problems
    elif opts.arch == 'tran':
        c.network_kwargs.update(model_type='TransformerModel', model_channels=128, channel_mult_noise=opts.channel_mult_noise, num_blocks=opts.num_transformer_blocks, softmax_onehot=opts.softmax_onehot)
    elif opts.arch == 'adm':
        c.network_kwargs.update(model_type='WrappedDhariwalUNet', model_channels=192, channel_mult=[1,2,3,4])
        c.network_kwargs.update(softmax_onehot=opts.softmax_onehot)
    elif opts.arch == 'control-adm':
        c.network_kwargs.update(model_type='ControlledDhariwalUNet', model_channels=192, channel_mult=[1,2,3,4])
        c.network_kwargs.update(softmax_onehot=opts.softmax_onehot)
    elif opts.arch == 'ddpmpp':
        c.network_kwargs.update(model_type='WrappedOriginalSongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[2,2,2])
        c.network_kwargs.update(softmax_onehot=opts.softmax_onehot)
    elif opts.arch == 'film-ddpmpp':
        c.network_kwargs.update(model_type='FilmWrappedOriginalSongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[2,2,2])
        c.network_kwargs.update(softmax_onehot=opts.softmax_onehot)
    elif opts.arch == 'dalle2':
        c.network_kwargs.update(model_type='DALLE2CLIPModel', dim=512, depth=6, dim_head=64, heads=8)
    else:
        raise Exception
    # elif opts.arch == 'ncsnpp':
    #     c.network_kwargs.update(model_type='OriginalSongUNet', embedding_type='fourier', encoder_type='residual', decoder_type='standard')
    #     c.network_kwargs.update(channel_mult_noise=2, resample_filter=[1,3,3,1], model_channels=128, channel_mult=[2,2,2])

    # Preconditioning & loss function.
    if opts.precond == 'vp':
        c.network_kwargs.class_name = 'training.networks.VPPrecond'
        c.loss_kwargs.class_name = 'training.loss.VPLoss'
    elif opts.precond == 've':
        c.network_kwargs.class_name = 'training.networks.VEPrecond'
        c.loss_kwargs.class_name = 'training.loss.VELoss'
    else:
        assert opts.precond == 'edm'
        c.network_kwargs.class_name = 'training.networks.EDMPrecond'
        c.loss_kwargs.class_name = 'training.loss.EDMLoss'

    # Network options.
    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres
    if opts.augment:
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', p=opts.augment)
        c.augment_kwargs.update(xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
        c.network_kwargs.augment_dim = 9
    c.network_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16)

    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump, log_img_ticks=opts.sample)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Transfer learning and resume.
    if opts.transfer is not None:
        if opts.resume is not None:
            raise click.ClickException('--transfer and --resume cannot be specified at the same time')
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
    elif opts.resume is not None:
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    # Initialize wandb
    if dist.get_rank() == 0:
        wandb.init(
            entity=os.environ['WANDB_ENTITY'], project=os.environ['WANDB_PROJECT'], config=c,
            dir=opts.wandb_dir, id=opts.resume_id, resume=opts.resume_id is not None,
        )

    # Description string.
    cond_str = 'cond-' + ''.join(map(str, opts.observed))
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    desc = f'{dataset_name:s}-{cond_str:s}-{opts.arch:s}-{opts.precond:s}-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-{dtype_str:s}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{wandb.run.id}')
        assert not os.path.exists(c.run_dir)

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    training_loop.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
