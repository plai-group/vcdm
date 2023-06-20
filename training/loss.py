# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence
from training.structure import StructuredArgument

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, structure, noise_mult, pred_x0, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.structure = structure
        self.noise_mult = StructuredArgument(noise_mult, structure=structure, dtype=torch.float32)
        self.pred_x0 = StructuredArgument(pred_x0, structure=structure, dtype=torch.uint8)
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, x, y, augment_labels):
        B = x.shape[0]
        rnd_normal = torch.randn([B, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        sigma_full = sigma * self.noise_mult.lats.to(sigma.device)
        n = torch.randn_like(x) * sigma_full
        D_xn = net(x+n, y=y, sigma=sigma, augment_labels=augment_labels)
        weight = (sigma_full ** 2 + self.sigma_data ** 2) / (sigma_full * self.sigma_data) ** 2
        pred_x0_lats = self.pred_x0.lats.to(x.device).view(1, -1)
        if pred_x0_lats.bool().any():
            # weight computed above is ~1/sigma**2 for whichever sigma is smallest. For small sigma_full,
            # this means weight is very large. When predicting x0 for onehots, we probably don't want this
            # scaling, so set all weights for onehots to 1.
            weight = weight * (1-pred_x0_lats) + torch.ones_like(weight) * pred_x0_lats
        loss = weight * ((D_xn - x) ** 2)
        return loss

#----------------------------------------------------------------------------
