# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import itertools as it
import copy
import numpy as np
import torch
from torch_utils import persistence
from torch.nn.functional import silu
from torch.nn import SiLU, Sequential
from training.structure import StructuredArgument
from training.networks.original import *
from training.networks.modified_unet import ModifiedUNetBlock

#----------------------------------------------------------------------------
# Reimplementation of the DDPM++ and NCSN++ architectures from the paper
# "Score-Based Generative Modeling through Stochastic Differential
# Equations". By Karras et al. but edited to be able to output a `label' vector
# in addition to the image.

@persistence.persistent_class
class SongUNet(torch.nn.Module):
    def __init__(self,
        img_resolution,                     # Image resolution at input/output.
        in_channels,                        # Number of color channels at input.
        out_channels,                       # Number of color channels at output.
        label_in_dim        = 0,            # Number of class labels, 0 = unconditional.
        label_out_dim       = 0,            # Number of latent class labels, >0 means jointly generating label.
        augment_dim         = 0,            # Augmentation label dimensionality, 0 = no augmentation.

        model_channels      = 128,          # Base multiplier for the number of channels.
        channel_mult        = [1,2,2,2],    # Per-resolution multipliers for the number of channels.
        channel_mult_emb    = 4,            # Multiplier for the dimensionality of the embedding vector.
        num_blocks          = 4,            # Number of residual blocks per resolution.
        attn_resolutions    = [16],         # List of resolutions with self-attention.
        dropout             = 0.10,         # Dropout probability of intermediate activations.
        label_dropout       = 0,            # Dropout probability of class labels for classifier-free guidance.

        embedding_type      = 'positional', # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
        channel_mult_noise  = 1,            # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        encoder_type        = 'standard',   # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
        decoder_type        = 'standard',   # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
        resample_filter     = [1,1],        # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.

        emb_layer_type      = None,
    ):
        assert embedding_type in ['fourier', 'positional']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']

        super().__init__()
        self.in_channels = in_channels
        self.img_resolution = img_resolution
        self.label_dropout = label_dropout
        self.label_in_dim = label_in_dim
        self.label_out_dim = label_out_dim
        self.skips_for_emb = 'skip' in emb_layer_type
        # self.label_cond = label_dim > 0 and observed[1] == 1
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))
        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn,
        )
        if self.label_out_dim > 0:
            if 'attn' in emb_layer_type:
                block_kwargs['num_emb_queries'] = 8
                UNetBlock_ = ModifiedUNetBlock
            elif 'shitty' in emb_layer_type:
                UNetBlock_ = UNetBlockShittyModifyEmb
            else:
                raise ValueError(f'Unknown emb_layer_type: {emb_layer_type}')
        else:
            UNetBlock_ = UNetBlock

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=noise_channels, endpoint=True) if embedding_type == 'positional' else FourierEmbedding(num_channels=noise_channels)
        self.map_label = Linear(in_features=label_in_dim, out_features=noise_channels, **init) if label_in_dim > 0 else None
        self.map_augment = Linear(in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None
        self.map_layer0 = Linear(in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        self.label_out = Linear(in_features=emb_channels, out_features=label_out_dim, **init_zero) if label_out_dim > 0 else None

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f'{res}x{res}_conv'] = Conv2d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f'{res}x{res}_down_unet'] = UNetBlock_(in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{res}x{res}_aux_down'] = Conv2d(in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter)
                    self.enc[f'{res}x{res}_aux_skip'] = Conv2d(in_channels=caux, out_channels=cout, kernel=1, **init)
                if encoder_type == 'residual':
                    self.enc[f'{res}x{res}_aux_residual'] = Conv2d(in_channels=caux, out_channels=cout, kernel=3, down=True, resample_filter=resample_filter, fused_resample=True, **init)
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = (res in attn_resolutions)
                self.enc[f'{res}x{res}_block{idx}_unet'] = UNetBlock_(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
        skips = [block.out_channels for name, block in self.enc.items() if 'aux' not in name]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0_unet'] = UNetBlock_(in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1_unet'] = UNetBlock_(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up_unet'] = UNetBlock_(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = (idx == num_blocks and res in attn_resolutions)
                if self.label_out_dim > 0:
                    emb_channels_in_kwarg = {'emb_channels_in': 2*emb_channels if 'skip' in emb_layer_type else emb_channels}
                else:
                    emb_channels_in_kwarg = {}
                self.dec[f'{res}x{res}_block{idx}_unet'] = UNetBlock_(in_channels=cin, out_channels=cout, attention=attn, **block_kwargs, **emb_channels_in_kwarg)
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(channel_mult) - 1:
                    self.dec[f'{res}x{res}_aux_up'] = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel=0, up=True, resample_filter=resample_filter)
                self.dec[f'{res}x{res}_aux_norm'] = GroupNorm(num_channels=cout, eps=1e-6)
                self.dec[f'{res}x{res}_aux_conv'] = Conv2d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(self, img, label, noise_labels, augment_labels=None):
        x = img
        
        # Mapping.
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # swap sin/cos
        if self.map_label is not None:
            tmp = label
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = silu(self.map_layer1(emb))

        # Encoder.
        skips = []
        emb_skips = []
        aux = x
        for name, block in self.enc.items():
            if 'aux_down' in name:
                aux = block(aux)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                if 'unet' in name:
                    block_out = block(x, emb)
                    if (self.label_out_dim > 0):
                        x, emb = block_out
                    else:
                        x = block_out
                else:
                    x = block(x)
                skips.append(x)
                emb_skips.append(emb)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux)
            elif 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_conv' in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    x = torch.cat([x, skips.pop()], dim=1)
                if (self.label_out_dim > 0) and emb.shape[1] != block.emb_channels_in:
                    emb = torch.cat([emb, emb_skips.pop()], dim=1)
                block_out = block(x, emb)
                if (self.label_out_dim > 0):
                    x, emb = block_out
                else:
                    x = block_out

        if self.label_out is not None:
            label = self.label_out(emb)
        return aux, label

# ----------------------------------------------------------------------------
# ConcatUNet - wrapper around the SongUNet that helps out with data consisting
# of multiple image-shaped and vector-shaped tensors.

@persistence.persistent_class
class ConcatUNet(SongUNet):
    def __init__(self, structure, softmax_onehot, *args, **kwargs):
        # check everything is either image or vector-shaped
        self.structure = structure
        shapes = structure.shapes
        observed = structure.observed
        self.softmax_onehot = softmax_onehot
        self.image_shaped = [len(s) == 3 for s in shapes]
        self.vector_shaped = [len(s) == 1 for s in shapes]
        assert all(i ^ v for i, v in zip(self.image_shaped, self.vector_shaped))
        label_in_dim = sum(s[0] for s, v in zip(shapes, self.vector_shaped) if v)  # sum of all vector dimensions
        label_out_dim = sum(s[0] for s, v, o in zip(shapes, self.vector_shaped, observed) if v and not o)  # sum of all latent vector dimensions
        in_channels = sum(s[0] for s, i in zip(shapes, self.image_shaped) if i)  # sum of all image channels
        out_channels = sum(s[0] for s, i, o in zip(shapes, self.image_shaped, observed) if i and not o)  # sum of all latent image channels
        # check all images are same H, W
        img_resolutions = [s[2] for s, i in zip(shapes, self.image_shaped) if i]
        if sum(self.image_shaped) == 0:
            in_channels = 1  # dummy 1x32x32 image so that the network can still run
            out_channels = 1
            img_resolutions = [32]
        assert all(r == img_resolutions[0] for r in img_resolutions)
        super().__init__(*args, img_resolution=img_resolutions[0], in_channels=in_channels, out_channels=out_channels, label_in_dim=label_in_dim, label_out_dim=label_out_dim, **kwargs)

    def forward(self, x, y, noise_labels, augment_labels=None):
        tensors = self.structure.unflatten_batch(x, y, pad_marg=False)
        img_shaped_thing = torch.cat([t for t, i in zip(tensors, self.image_shaped) if i], dim=1) if sum(self.image_shaped) > 0 else torch.zeros([x.shape[0], 1, 32, 32], device=x.device)
        vec_shaped_thing = torch.cat([t for t, v in zip(tensors, self.vector_shaped) if v], dim=1) if sum(self.vector_shaped) > 0 else None
        img_shaped_thing, vec_shaped_thing = super().forward(img_shaped_thing, vec_shaped_thing, noise_labels, augment_labels=augment_labels)
        # merge img_ and vec_shaped_thing back into `tensors`
        if sum(self.image_shaped) == 0:
            vec_shaped_thing = vec_shaped_thing + 0 * img_shaped_thing.sum()  # dummy sum to avoid unused variable warning
        tensors = []
        for i, v, shape, observed, onehot in zip(self.image_shaped, self.vector_shaped, self.structure.shapes, self.structure.observed, self.structure.is_onehot):
            c = shape[0]
            if observed:
                append = None
            elif i:
                append, img_shaped_thing = img_shaped_thing[:, :c], img_shaped_thing[:, c:]
            elif v:
                append, vec_shaped_thing = vec_shaped_thing[:, :c], vec_shaped_thing[:, c:]
            else:
                raise Exception
            if self.softmax_onehot and onehot:
                append = torch.softmax(append, dim=1)
            tensors.append(append)
        return self.structure.flatten_latents(tensors, contains_marg=False)

    def set_requires_grad(self, val, freeze_pretrained=False):
        for param in self.parameters():
            param.requires_grad = val


@persistence.persistent_class
class WrappedDhariwalUNet(DhariwalUNet):
    def edit_pretrained_state_dict(self, state_dict):
        """
        Assume we will always just be converting label-conditional to
        label and embedding-conditional.
        So check if we have an embedding and, if so, add more dimensions to
        the label embedder.
        """
        # figure out structure of conditioning vector
        own_state_dict = self.state_dict()
        obs_dims = [s[0] for s, o in zip(self.structure.shapes, self.structure.observed) if len(s) == 1 and o]
        is_orig_label = [od == 1000 for od in obs_dims]
        assert state_dict['model.map_label.weight'].shape[1] == 1000  # pretrained checkpoints for this are all conditioned on just class label, can reimplement if that changes
        if sum(is_orig_label) == 1:
            orig_label_idx = is_orig_label.index(True)
            dims_before = sum(obs_dims[:orig_label_idx])
            new_map_label = torch.zeros_like(own_state_dict['map_label.weight'])
            new_map_label[:, dims_before:dims_before+1000] = state_dict['model.map_label.weight']
            state_dict['model.map_label.weight'] = new_map_label
        else:
            raise NotImplementedError('Checkpoint is class-conditional, not defined how to load unto non-class-conditional model.')
            assert sum(is_orig_label) == 0
            # state_dict['model.map_label.weight'] = torch.zeros_like(own_state_dict['map_label.weight'])
        return state_dict
        # pretrained_label_dim = state_dict['label_in.weight'].shape[0]

    def __init__(self, structure, softmax_onehot, label_dim=None, *args, **kwargs):
        self.structure = structure
        assert softmax_onehot == False
        channels, img_resolution, _ = structure.shapes[0]
        if label_dim is None:
            if len(structure.shapes) == 1:
                label_dim = 0
            elif len(structure.shapes) >= 2:
                # concatenate all other tensors into a "label"
                assert all(len(s) == 1 for s in structure.shapes[1:])
                label_dim = sum(s[0] for s in structure.shapes[1:])
            else:
                raise Exception
        super().__init__(img_resolution=img_resolution, in_channels=channels, out_channels=channels, label_dim=label_dim, *args, **kwargs)

    def forward(self, x, y, noise_labels, augment_labels=None):
        data = self.structure.unflatten_batch(x, y, pad_marg=False)
        if len(data) == 1:
            image = data[0]
            label = None
        elif len(data) >= 2:
            image, *labels = data
            label = torch.cat(labels, dim=1)
            assert (not self.structure.observed[0]) and all(self.structure.observed[1:])
        image = super().forward(x=image, noise_labels=noise_labels, class_labels=label, augment_labels=augment_labels)
        data = [image, None]
        return self.structure.flatten_latents(data, contains_marg=False)

    def set_requires_grad(self, val, freeze_pretrained=False):
        for name, param in self.named_parameters():
            param.requires_grad = val


@persistence.persistent_class
class ControlledDhariwalUNet(WrappedDhariwalUNet):
    def __init__(self, structure, **kwargs):
        # set flags
        self.loaded_parameters = False
        self.added_control = False
        # figure out structure of conditioning vector
        obs_dims = [s[0] for s, o in zip(structure.shapes, structure.observed) if len(s) == 1 and o]
        self.obs_new = [od != 1000 for od in obs_dims]
        self.obs_class_label = [od == 1000 for od in obs_dims]
        self.obs_new_dims = sum(od for od, o in zip(obs_dims, self.obs_new) if o)
        self.obs_class_label_dims = sum(od for od, o in zip(obs_dims, self.obs_class_label) if o)
        if sum(self.obs_class_label) == 0:
            # pretrained model is trained with class labels sometimes dropped out, so we can handle this
            # case by feeding in a dummy all-zero class label in each forward pass
            raise Exception('Check that class labels are dropped out with prob > 0 for original model. I think they\'re not.')
            self.condition_on_class_labels = False
        elif sum(self.obs_class_label) == 1:
            # pretrained model is conditioned on class labels, so this is handled easily
            self.condition_on_class_labels = True
        else:
            raise Exception
        # initialise original model
        kwargs['label_dim'] = self.obs_class_label_dims
        super().__init__(structure=structure, **kwargs)
        assert self.map_label.in_features == self.obs_class_label_dims

    def edit_pretrained_state_dict(self, state_dict):
        """
        We add extra control parameters later (in add_control)
        """
        assert not self.added_control
        self.loaded_parameters = True
        return state_dict

    def add_control(self):
        device = next(self.parameters()).device
        assert not self.added_control
        self.added_control = True
        print('Adding control to DhariwalUNet.')
        self.control_enc = copy.deepcopy(self.enc)
        self.control_middle = torch.nn.ModuleList([copy.deepcopy(d) for d, _ in zip(self.dec.values(), range(2))]).to(device)
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        self.control_convs = torch.nn.ModuleList([
            Conv2d(b.out_channels, b.out_channels, 1, **init_zero) for b in self.enc.values()
        ])
        self.control_convs_middle = torch.nn.ModuleList([
            Conv2d(b.out_channels, b.out_channels, 1, **init_zero) for b, _ in zip(self.dec.values(), range(2))
        ]).to(device)
        emb_channels = self.map_label.out_features
        control_channels = self.obs_new_dims
        self.control_in = Linear(control_channels, emb_channels, **init_zero).to(device)

    def set_requires_grad(self, val, freeze_pretrained=False):
        """
        Freezes pretrained parameters, regardless of freeze_pretrained flag.
        """
        for name, param in self.named_parameters():
            param.requires_grad = val and ('control' in name)

    def forward(self, x, y, noise_labels, augment_labels=None):
        if self.condition_on_class_labels:
            # extract class label from y
            y_orig = torch.cat([t for t, c in zip(y, self.obs_class_label) if c], dim=1)
        else:
            # "dropped out" class label
            y_orig = torch.zeros([x.shape[0], 1000], device=x.device)
        y_control = torch.cat([t for t, n in zip(y, self.obs_new) if n], dim=1)
        x, = self.structure.unflatten_latents(x)
        # Mapping.
        emb = self.map_noise(noise_labels)
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        emb = silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        if self.map_label is not None:
            tmp = y_orig
            if self.training and self.label_dropout:
                tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
            emb = emb + self.map_label(tmp)
        emb = silu(emb)

        # Encoder.
        skips = []
        control_x = x
        control_emb = emb + self.control_in(y_control)
        for block, control_block, control_conv in zip(self.enc.values(), self.control_enc.values(), self.control_convs):
            kwargs = dict(emb=emb) if isinstance(block, UNetBlock) else {}
            control_kwargs = dict(emb=control_emb) if isinstance(control_block, UNetBlock) else {}
            x = block(x, **kwargs)
            control_x = control_block(control_x, **control_kwargs)
            skips.append(x + control_conv(control_x))


        # Decoder.
        for b, block in enumerate(self.dec.values()):
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)
            if b < 2:
                # also add control - these don't need skip connections
                control_x = self.control_middle[b](control_x, control_emb)
                x = x + self.control_convs_middle[b](control_x)
        x = self.out_conv(silu(self.out_norm(x)))
        data = [x,] + [None,] * sum(self.structure.observed)
        return self.structure.flatten_latents(data, contains_marg=False)


@persistence.persistent_class
class WrappedOriginalSongUNet(OriginalSongUNet):

    def edit_pretrained_state_dict(self, state_dict):
        """
        Method for editing pretrained state dict if we change what we wish to condition on.
        """
        # add model.map_label.weight and model.map_label.bias
        if self.structure.observed[1]:
            own_state_dict = self.state_dict()
            own_params = [name for name in own_state_dict.keys() if 'map_label' in name]
            for own_name in own_params:
                other_name = f'model.{own_name}'
                if other_name not in state_dict: 
                    state_dict[other_name] = torch.zeros_like(own_state_dict[own_name])
        return state_dict

    def set_requires_grad(self, val, freeze_pretrained=False):
        for name, param in self.named_parameters():
            if 'map_label' in name:
                param.requires_grad = val
            else:
                param.requires_grad = val and not freeze_pretrained

    def __init__(self, structure, softmax_onehot, *args, **kwargs):
        self.structure = structure
        assert softmax_onehot == False
        channels, img_resolution, _ = structure.shapes[0]
        if len(structure.shapes) == 1:
            label_dim = 0
        elif len(structure.shapes) == 2:
            assert (not self.structure.observed[0]) and self.structure.observed[1]
            label_dim = structure.shapes[1][0]
        elif len(structure.shapes) == 3:
             implemented_obs = [0, 1, 1]
             for o, io in zip(structure.observed, implemented_obs):
                assert o == io
             label_dim = structure.shapes[1][0] + structure.shapes[2][0]
        else:
            raise Exception
        super().__init__(img_resolution=img_resolution, in_channels=channels, out_channels=channels, label_dim=label_dim, *args, **kwargs)

    def forward(self, x, y, noise_labels, augment_labels=None):
        data = self.structure.unflatten_batch(x, y, pad_marg=False)
        if len(data) == 1:
            image = data[0]
            label = None
        elif len(data) == 2:
            image, label = data
        elif len(data) == 3:
            image, y1, y2 = data
            label = torch.cat([y1, y2], dim=1)
        image = super().forward(x=image, noise_labels=noise_labels, class_labels=label, augment_labels=augment_labels)
        data = [image,]
        return self.structure.flatten_latents(data, contains_marg=False)


@persistence.persistent_class
class FilmWrappedOriginalSongUNet(OriginalSongUNet):

    def edit_pretrained_state_dict(self, state_dict):
        """
        Method for editing pretrained state dict if we change what we wish to condition on.
        """
        # add model.map_label.weight and model.map_label.bias
        if self.structure.observed[1]:
            own_state_dict = self.state_dict()
            own_params = [name for name in own_state_dict.keys() if 'get_gammas_betas' in name]
            for own_name in own_params:
                other_name = f'model.{own_name}'
                if other_name not in state_dict: 
                    state_dict[other_name] = own_state_dict[own_name]
        return state_dict

    def set_requires_grad(self, val, freeze_pretrained=False):
        for name, param in self.named_parameters():
            if 'get_gammas_betas' in name:
                param.requires_grad = val
            else:
                param.requires_grad = val and not freeze_pretrained

    def __init__(self, structure, softmax_onehot, *args, **kwargs):
        self.structure = structure
        assert softmax_onehot == False
        channels, img_resolution, _ = structure.shapes[0]
        super().__init__(img_resolution=img_resolution, in_channels=channels, out_channels=channels, label_dim=0, *args, **kwargs)
        if len(structure.shapes) == 1:
            pass
        elif len(structure.shapes) == 2:
            assert (not self.structure.observed[0]) and self.structure.observed[1]
            self.unet_channels = [block.out_channels for block in it.chain(self.enc.values(), self.dec.values()) if isinstance(block, UNetBlock)]
            cond_dim = structure.shapes[1][0]
            emb_channels = self.map_layer0.out_features
            init = dict(init_mode='xavier_uniform')
            init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
            self.get_gammas_betas = Sequential(
                Linear(cond_dim, emb_channels, **init),
                GroupNorm(emb_channels),
                SiLU(),
                Linear(emb_channels, emb_channels, **init),
                GroupNorm(emb_channels,),
                SiLU(),
                Linear(emb_channels, 2*sum(self.unet_channels), **init_zero),
            )
        else:
            raise Exception

    def forward(self, x, y, noise_labels, augment_labels=None):
        data = self.structure.unflatten_batch(x, y, pad_marg=False)
        if len(data) == 1:
            image = data[0]
            label = None
        elif len(data) == 2:
            image, label = data
            _gammas_betas = self.get_gammas_betas(label).unsqueeze(2).unsqueeze(3)
            gamma_betas = []
            for n_channels in self.unet_channels:
                gamma, beta = _gammas_betas[:, :n_channels], _gammas_betas[:, n_channels:2*n_channels]
                _gammas_betas = _gammas_betas[:, 2*n_channels:]
                gamma_betas.append((gamma, beta))
        image = super().forward(x=image, noise_labels=noise_labels, class_labels=None, augment_labels=augment_labels, modulation_gamma_betas=gamma_betas)
        data = [image,]
        return self.structure.flatten_latents(data, contains_marg=False)
