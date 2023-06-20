import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import time
from .original import PositionalEmbedding


def get_timestep_embedding(timesteps, embedding_dim, max_timesteps=10000):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(max_timesteps) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class ResnetBlock(nn.Module):
    def __init__(self, *, channels,
                 dropout, temb_channels=512, node_emb_channels=None):
        super().__init__()
        make_linear = lambda a, b: nn.Conv1d(a, b, kernel_size=1, stride=1, padding=0)
        self.channels = channels
        self.norm1 = Normalize(channels)
        # rewrite the below using make_linear
        self.conv1 = make_linear(channels, channels)
        self.temb_proj = make_linear(temb_channels, channels)
        if node_emb_channels is not None:
            self.var_proj = make_linear(node_emb_channels, channels)
        self.norm2 = Normalize(channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = make_linear(channels, channels)

    def forward(self, x, temb, node_emb=None):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = h + self.temb_proj(nonlinearity(temb))
        if node_emb is not None:
            h = h + self.var_proj(node_emb)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels, n_heads=1, attn_dim_reduce=1):
        super().__init__()
        self.in_channels = in_channels
        self.n_heads = n_heads

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv1d(in_channels,
                                 in_channels//attn_dim_reduce,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv1d(in_channels,
                                 in_channels//attn_dim_reduce,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv1d(in_channels,
                                 in_channels//attn_dim_reduce,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv1d(in_channels//attn_dim_reduce,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def sparse_forward(self, x, sparse_attention_mask_and_indices):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, n = q.shape
        heads = self.n_heads
        reshape_for_transformer = lambda t: t.reshape(b, heads, c//heads, n)
        # beta = (int(c//heads)**(-0.5)) # standard attention scaling
        # unnormalized attention
        beta = 1
        q = reshape_for_transformer(q)
        k = reshape_for_transformer(k)
        v = reshape_for_transformer(v)

        valid_indices_mask, attendable_indices = sparse_attention_mask_and_indices
        nq, max_attendable_keys = valid_indices_mask.shape
        attendable_indices = attendable_indices.view(1, 1, nq, max_attendable_keys)\
                                               .expand(b, heads, nq, max_attendable_keys)
        def get_keys_or_values(t, indices):
            *batch_shape, nd, nv = t.shape
            t = t.transpose(-1, -2)\
                .view(*batch_shape, nv, 1, nd)\
                .expand(*batch_shape, nv, max_attendable_keys, nd)
            index = indices.view(*batch_shape, nv, max_attendable_keys, 1)\
                .expand(-1, -1, -1, -1, c//heads)
            return t.gather(dim=2, index=index)

        attended_keys = get_keys_or_values(k, indices=attendable_indices)   # b x heads x h*w x max_attendable_keys x c
        attended_values = get_keys_or_values(v, indices=attendable_indices)

        weights = beta * torch.einsum('bhqkc,bhcq->bhqk', attended_keys, q)
        inf_matrix = torch.zeros_like(valid_indices_mask)
        inf_matrix[valid_indices_mask==0] = torch.inf
        weights = weights - inf_matrix.view(1, 1, nq, max_attendable_keys)
        weights = weights.softmax(dim=-1)

        h_ = torch.einsum('bhqk,bhqkc->bhqc', weights, attended_values)
        h_ = h_.permute(0, 3, 1, 2).reshape(b, c, n)
        h_ = self.proj_out(h_)
        out = x+h_
        return out, None

    def forward(self, x, sparsity_matrix=None, sparse_attention_mask_and_indices=None, return_w=False):
        if sparse_attention_mask_and_indices is not None:
            out, w_ = self.sparse_forward(x, sparse_attention_mask_and_indices)
            return (out, w_) if return_w else out
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, n = q.shape
        heads = self.n_heads
        reshape_for_transformer = lambda t: t.reshape(b, heads, c//heads, n)
        q = reshape_for_transformer(q)
        k = reshape_for_transformer(k)
        v = reshape_for_transformer(v)
        w_ = torch.einsum('bhdk,bhdq->bhqk', k, q)
        w_ = w_ * (int(c//heads)**(-0.5))
        if sparsity_matrix is not None:
            inf_matrix = torch.zeros_like(sparsity_matrix)
            inf_matrix[sparsity_matrix==0] = torch.inf
            w_ = w_ - inf_matrix.view(-1, 1, n, n)
        w_ = torch.nn.functional.softmax(w_, dim=3)
        h_ = torch.einsum('bhdk,bhqk->bhdq', v, w_)
        h_ = h_.view(b, c, n)
        h_ = self.proj_out(h_)
        out = x+h_
        return (out, w_) if return_w else out


class GraphicallyStructuredModel(nn.Module):

    def __init__(self,
        structure,
        softmax_onehot,
        model_channels,                     # Base multiplier for the number of channels.
        channel_mult_noise,                 # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
        num_blocks,                         # Number of residual blocks per resolution.
        node_embeddings='EE',
        sparse_attention=True,
        augment_dim         = -1,           # Not used, but required for compatibility with SongUNet/ConcatUNet.
        # ------------------------------------------------------------------------------------------------------
        channel_mult_emb    = -1,           # Multiplier for the dimensionality of the embedding vector for conditioning, augmentation, and eventually noise.
        attn_resolutions    = -1,           # Not used, but required for compatibility with SongUNet/ConcatUNet.
        dropout             = -1,           # Not used, but required for compatibility with SongUNet/ConcatUNet.
        label_dropout       = -1,           # Not used, but required for compatibility with SongUNet/ConcatUNet.
        embedding_type      = -1,           # Not used, but required for compatibility with SongUNet/ConcatUNet.
        encoder_type        = -1,           # Not used, but required for compatibility with SongUNet/ConcatUNet.
        decoder_type        = -1,           # Not used, but required for compatibility with SongUNet/ConcatUNet.
        resample_filter     = -1,           # Not used, but required for compatibility with SongUNet/ConcatUNet.
        img_resolution      = -1,           # Not used, but required for compatibility with SongUNet/ConcatUNet.
        in_channels         = -1,           # Not used, but required for compatibility with SongUNet/ConcatUNet.
        out_channels        = -1,           # Not used, but required for compatibility with SongUNet/ConcatUNet.
        label_in_dim        = -1,           # Not used, but required for compatibility with SongUNet/ConcatUNet.
        label_out_dim       = -1,           # Not used, but required for compatibility with SongUNet/ConcatUNet.
        channel_mult        = -1,           # Not used, but required for compatibility with SongUNet/ConcatUNet.
    ):
        super().__init__()
        self.structure = structure
        # use_faithful_inverse/shared_var_embed options ?
        # n_variables (gettable from structure)
        self.emb_dim = model_channels
        self.temb_dim = model_channels * channel_mult_noise
        self.num_blocks = num_blocks
        self.n_discrete_options = set(shape[0] if onehot else 0 for shape, onehot in zip(structure.shapes, structure.is_onehot))
        self.n_discrete_options_latent = set(shape[0] if onehot else 0 for shape, onehot, obs in zip(structure.shapes, structure.is_onehot, structure.observed) if not obs)
        self.attn_dim_reduce = 1                # TODO can later edit to reduce memory consumption when there are many variables
        self.n_heads = 8                        # TODO think about this
        self.sparse_attention = sparse_attention
        self.sparse_attention_mask_and_indices = tuple([t.clone() for t in self.structure.graphical_model_mask_and_indices]) if self.sparse_attention else None
        # if self.sparse_attention:
        #     self.plot_sparse_attention_mask_and_indices()
        self.node_embeddings = node_embeddings
        self.shareable_embedding_indices = self.structure.shareable_embedding_indices.clone() if self.node_embeddings == 'EE' else None
        self.softmax_onehot = softmax_onehot

        # time embedding
        self.temb = nn.Sequential(
            PositionalEmbedding(self.temb_dim),
            nn.SiLU(),
            nn.Linear(self.temb_dim, self.temb_dim),
            nn.SiLU(),
            nn.Linear(self.temb_dim, self.temb_dim),
        )
        # in and out projections
        if 0 in self.n_discrete_options:
            self.cont_in_proj = nn.Conv1d(1, self.emb_dim, kernel_size=1)
        if 0 in self.n_discrete_options_latent:
            self.cont_out_proj = nn.Conv1d(self.emb_dim, 1, kernel_size=1)
        self.disc_in_projs = nn.ModuleDict({
            str(n_options): nn.Conv1d(n_options, self.emb_dim, kernel_size=1) for n_options in set(self.n_discrete_options) if n_options > 0
        })
        self.disc_out_projs = nn.ModuleDict({
            str(n_options): nn.Conv1d(self.emb_dim, n_options, kernel_size=1) for n_options in set(self.n_discrete_options_latent) if n_options > 0
        })
        # observe embedding
        self.obs_emb = nn.Parameter(torch.randn(self.emb_dim), requires_grad=True) if sum(self.structure.observed) > 0 else None
        # augment embedding
        if augment_dim > 0:
            self.augment_layer = nn.Linear(augment_dim, self.emb_dim, bias=False)
        # core transformer/resnet blocks
        self.transformers = nn.ModuleList([
            AttnBlock(self.emb_dim, n_heads=self.n_heads, attn_dim_reduce=self.attn_dim_reduce)
            for _ in range(self.num_blocks)
        ])
        self.res_blocks = nn.ModuleList([
            ResnetBlock(channels=self.emb_dim, temb_channels=self.temb_dim, node_emb_channels=self.emb_dim, dropout=False)
            for _ in range(self.num_blocks)
        ])
        # node embeddings
        if node_embeddings == 'IE':
            n_node_embeddings = sum((np.prod(shape[1:]) if len(shape) > 1 else 1) if onehot else np.prod(shape) for shape, onehot in zip(structure.shapes, structure.is_onehot))
        elif node_embeddings == 'EE':
            n_node_embeddings = len(self.shareable_embedding_indices.unique())
        else:
            raise ValueError(f'Invalid node_embeddings option: {node_embeddings}')
        self.grouped_node_embeddings = nn.Parameter(torch.randn(self.emb_dim, n_node_embeddings), requires_grad=True)

    def project_data_to_emb(self, x, y):
        data = self.structure.unflatten_batch(x, y, pad_marg=False)
        embedded_data = []
        for (data_i, shape, onehot) in zip(data, self.structure.shapes, self.structure.is_onehot):
            embedder = self.disc_in_projs[str(shape[0])] if onehot else self.cont_in_proj
            if not onehot:
                data_i = data_i.unsqueeze(1)  # add channel dim
            B, C, *_ = data_i.shape
            data_i = data_i.view(B, C, -1)  #  flattened data should match ordering given by structure.flatten_batch
            embedded_data.append(embedder(data_i))
        return embedded_data

    def project_emb_to_x(self, emb):
        data = []
        for i, (shape, onehot, obs) in enumerate(zip(self.structure.shapes, self.structure.is_onehot, self.structure.observed)):
            if onehot:
                n_options, n_nodes = shape[0], (np.prod(shape[1:]) if len(shape) > 1 else 1)
            else:
                n_nodes = np.prod(shape)
            # take next chunk of emb
            emb_chunk, emb = emb[:, :, :n_nodes], emb[:, :, n_nodes:]
            if obs:
                out_tensor = None
            else:
                proj = self.disc_out_projs[str(n_options)] if onehot else self.cont_out_proj
                out_tensor = proj(emb_chunk)
                if self.softmax_onehot and onehot:
                    out_tensor = out_tensor.softmax(dim=1)
            data.append(out_tensor)
        return self.structure.flatten_latents(data, contains_marg=False)

    def get_node_embeddings(self):
        if self.node_embeddings == 'IE':
            return self.grouped_node_embeddings
        elif self.node_embeddings == 'EE':
            self.shareable_embedding_indices = self.shareable_embedding_indices.to(self.grouped_node_embeddings.device)
            return self.grouped_node_embeddings[:, self.shareable_embedding_indices]
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")

    def forward(self, x, y, sigma, augment_labels=None):
        # move non-parameters to device
        self.shareable_embedding_indices = self.shareable_embedding_indices.to(x.device) if self.shareable_embedding_indices is not None else None
        self.sparse_attention_mask_and_indices = tuple([t.to(x.device) for t in self.sparse_attention_mask_and_indices]) if self.sparse_attention_mask_and_indices is not None else None
        # assert augment_labels is None, "Augment labels not implemented for this model."
        # print(f"Augment labels: {augment_labels}")
        embs = self.project_data_to_emb(x, y)
        # add obs embeddings
        for i in [i for i, obs in enumerate(self.structure.observed) if obs]:
            embs[i] = embs[i] + self.obs_emb.view(1, -1, 1)
        # concatenate embs into single tensor
        embs = torch.cat(embs, dim=-1)  # B x C x n_nodes
        # add augment labels
        if augment_labels is not None:
            embs = embs + self.augment_layer(augment_labels).unsqueeze(-1)
        # get time embeddings
        tembs = self.temb(sigma).unsqueeze(-1)  # B x C x 1
        # get node embeddings
        node_embs = self.get_node_embeddings().unsqueeze(0)  # 1 x C x n_nodes
        # run stacked resnet/transfomer blocks
        for l, (res_block, transformer) in enumerate(zip(self.res_blocks, self.transformers)):
            embs = res_block(embs, tembs, node_embs)  # expects B x C x n_nodes
            embs = transformer(embs, sparse_attention_mask_and_indices=self.sparse_attention_mask_and_indices)  # expects B x C x n_nodes
        # run output projections and return
        return self.project_emb_to_x(embs)

    def plot_sparse_attention_mask_and_indices(self):
        n_nodes, max_attendable = self.sparse_attention_mask_and_indices[0].shape
        matrix = torch.zeros(n_nodes, n_nodes)
        for r, (attendable, indices) in enumerate(zip(*self.sparse_attention_mask_and_indices)):
            n_attendable = attendable.sum()
            if n_attendable == 1:
                matrix[r] = 0.5
            matrix[r, indices[:n_attendable.long().item()]] = 1.
        # save matrix as png
        import imageio
        imageio.imwrite('attention_mask.png', matrix)

    def set_requires_grad(self, val, freeze_pretrained=False):
        for param in self.parameters():
            param.requires_grad = val


class TransformerModel(GraphicallyStructuredModel):
    def __init__(self, **kwargs):
        super().__init__(node_embeddings='IE', sparse_attention=False, **kwargs)
