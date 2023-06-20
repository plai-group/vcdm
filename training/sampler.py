import numpy as np
import torch

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])


#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).

def edm_sampler(
    net, xT, y, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    return_intermediates=False,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=xT.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    noise_mult = net.noise_mult.lats.to(xT.device)

    # Main sampling loop.
    intermediates = {'denoised': [], 'xt': []} if return_intermediates else None
    x_next = xT.to(torch.float64) * t_steps[0] * noise_mult
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next
        t_cur_full = t_cur * noise_mult
        t_next_full = t_next * noise_mult

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma((t_cur + gamma * t_cur))
        t_hat_full = t_hat * noise_mult
        x_hat = x_cur + (t_hat_full ** 2 - t_cur_full ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x=x_hat, y=y, sigma=t_hat).to(torch.float64)
        if return_intermediates:
            intermediates['denoised'].append(denoised)
        d_cur = (x_hat - denoised) / t_hat_full
        x_next = x_hat + (t_next_full - t_hat_full) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x=x_next, y=y, sigma=t_next).to(torch.float64)
            d_prime = (x_next - denoised) / t_next_full
            x_next = x_hat + (t_next_full - t_hat_full) * (0.5 * d_cur + 0.5 * d_prime)

        if return_intermediates:
            intermediates['xt'].append(x_next.type(torch.float32))

    return intermediates if return_intermediates else x_next.type(torch.float32)
