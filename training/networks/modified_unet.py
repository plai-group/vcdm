from torch_utils import persistence
from torch.nn.functional import silu
import torch
from .original import *


@persistence.persistent_class
class ModifiedUNetBlock(UNetBlock):
    """
    UNetBlock modified to operate on an embedding vector as well as on an object with spatial dimensions.
    """
    def __init__(self,
        out_channels, eps, num_heads, init, init_zero, emb_channels, emb_channels_in=None,
        num_emb_queries=1,
        **kwargs  # TODO use emb_channels_in
    ):
        super().__init__(out_channels=out_channels, emb_channels=emb_channels,
                         num_heads=num_heads, eps=eps, init=init, init_zero=init_zero, **kwargs)
        self.num_emb_queries = num_emb_queries
        self.emb_channels_in = emb_channels_in if emb_channels_in is not None else emb_channels

        # recreate affine later with emb_channels_in instead of emb_channels
        # self.affine = Linear(in_features=self.emb_channels_in, out_features=out_channels*(2 if self.adaptive_scale else 1), **init)

        # resnet layer for emb
        self.emb_norm1 = GroupNorm(num_channels=self.emb_channels_in, eps=eps)
        self.emb_layer1 = Linear(in_features=self.emb_channels_in, out_features=self.emb_channels, **init)
        self.emb_norm2 = GroupNorm(num_channels=self.emb_channels, eps=eps)
        self.emb_layer2 = Linear(in_features=self.emb_channels, out_features=self.emb_channels, **init_zero)
        if self.emb_channels_in != self.emb_channels:
            self.emb_skip = Linear(in_features=self.emb_channels_in, out_features=self.emb_channels, **init)

        if self.num_heads:
            # attention stuff for emb
            self.emb_q = Linear(emb_channels, out_channels*self.num_heads*self.num_emb_queries)
            self.emb_proj = Linear(out_channels*self.num_heads*self.num_emb_queries, self.emb_channels)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        # ResNet-style transformation for emb
        skipped_emb = self.emb_skip(emb) if self.emb_channels_in != self.emb_channels else emb 
        emb = silu(self.emb_norm1(emb))
        emb = self.emb_layer1(emb)
        emb = silu(self.emb_norm2(emb))
        emb = self.emb_layer2(emb)
        emb = self.skip_scale * (skipped_emb + emb)

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            print('We reached this point???')
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(torch.nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            B, C, *_ = x.shape
            H = self.num_heads
            emb_q = self.emb_q(emb).view(B*H, C//H, self.num_emb_queries)
            q, k, v = self.qkv(self.norm2(x)).reshape(B*H, C//H, 3, -1).unbind(2)
            q = torch.cat([emb_q, q], dim=2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            emb_a, a = a[..., :self.num_emb_queries], a[..., self.num_emb_queries:]
            emb = self.skip_scale * (emb + self.emb_proj(emb_a.reshape(B, H*C*self.num_emb_queries)))
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale

        return x, emb