# Modified from: https://github.com/lucidrains/denoising-diffusion-pytorch/

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .module import *
from functools import partial
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim = 32,
        input_dim = 3,
        out_dim = 1,
        dim_mults=(1, 2),
        resnet_block_groups = 4,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
    ):
        """
        Params:
            dim: base feature dimension
            hidden_dim: feature dimension of hidden state
            input_dim: feature dimension of input
            out_dim: output dimension
            dim_mults: scaling factor of feature dimension during downsampling
        """
        super().__init__()

        self.out_dim = out_dim
        init_dim = dim
        self.init_conv = nn.Conv2d(input_dim, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features
            )
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        # time embeddings
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # downsampling
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in,
                                                                    dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]

        # GRU
        self.gru = SepConvGRU(hidden_dim, mid_dim)
        self.mid = block_klass(hidden_dim, mid_dim)

        # upsampling
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out,
                                                                         dim_in, 3, padding = 1)
            ]))

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, 1, 1)
        self.conf = nn.Conv2d(dim, 1, 1)

    def forward(self, x, hidden, time):
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)

        h = []
        # downsampling
        for block1, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = downsample(x)

        # GRU
        hidden = self.gru(hidden, x)
        x = hidden
        x = self.mid(x, t)

        # upsampling
        for block1, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)
            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        delta = self.final_conv(x)
        confidence = self.conf(x)
        confidence = torch.sigmoid(confidence)
        return hidden, delta, confidence

class ConditionEncoder(nn.Module):
    def __init__(self, num_sample, cost_dim, hidden_dim, out_chs):
        super().__init__()
        self.out_chs = out_chs
        # for cost volume
        self.convc1 = nn.Conv2d(cost_dim, hidden_dim, 3, padding=1)
        self.convc2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        # # for depth samples
        self.convd1 = nn.Conv2d(num_sample, hidden_dim, 3, padding=1)
        self.convd2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)

        self.output = nn.Conv2d(2 * hidden_dim, out_chs - 1, 3, padding=1)

    def forward(self, depth, depth_values, cost_volume):
        c_feat = F.relu(self.convc1(cost_volume))
        c_feat = F.relu(self.convc2(c_feat))
        d_feat = F.relu(self.convd1(depth_values))
        d_feat = F.relu(self.convd2(d_feat))
        feat = torch.cat([c_feat, d_feat], dim=1)

        output = F.relu(self.output(feat))
        return torch.cat([output, depth], dim=1)

class DiffusionUpdateBlockDepth(nn.Module):
    def __init__(
        self,
        args,
        dim = 16,
        dim_mults = (1, 2),
        hidden_dim = 32,
        num_sample = 4,
        cost_dim = 16,
        context_dim = 32,
        stage_idx = 0,
        iters = 3,
        ratio = 2,
    ):
        """
        Params:
            args: arguments from parser
            dim: base feature dimension of UNet
            dim_mults: scaling factor of feature dimension during downsampling in UNet
            hidden_dim: feature dimension of hidden state
            num_sample: number of depth samples
            cost_dim: dimension of lcoal cost volume
            context_dim: dimension of context feature
            stage_idx: index of current stage
            ratio: upsample ratio of depth map
        """
        super(DiffusionUpdateBlockDepth, self).__init__()
        self.iters = iters
        self.encoder = ConditionEncoder(
            num_sample=num_sample,
            cost_dim=cost_dim,
            hidden_dim=context_dim,
            out_chs=context_dim,
        )

        # mask for upsampling
        self.mask = nn.Sequential(
            nn.Conv2d(context_dim, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, ratio * ratio * 9, 1, padding=0)
        )

        self.unet = Unet(
            dim=dim,
            hidden_dim=hidden_dim,
            input_dim=self.encoder.out_chs+context_dim,
            out_dim=1,
            dim_mults=dim_mults,
        )

        self.stage_idx = stage_idx
        timesteps = args.timesteps[stage_idx]
        sampling_timesteps = args.sampling_timesteps[stage_idx]
        self.timesteps = timesteps

        # define beta schedule
        betas = cosine_beta_schedule(timesteps=timesteps).float()

        # sampling related parameters
        # default num sampling timesteps to number of timesteps at training
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = args.ddim_eta[stage_idx]
        self.scale = args.scale[stage_idx]

        # define alphas 
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        self.register_buffer('log_one_minus_alphas_cumprod', log_one_minus_alphas_cumprod)
        self.register_buffer('sqrt_recip_alphas', sqrt_recip_alphas)
        self.register_buffer('sqrt_recip_alphas_cumprod', sqrt_recip_alphas_cumprod)
        self.register_buffer('sqrt_recipm1_alphas_cumprod', sqrt_recipm1_alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = self.scale*torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod,
                                                  t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def forward(
            self,
            depth_cost_func,
            inv_depth,
            hidden,
            context,
            gt_inv_depth = None,
            inv_init_depth = None
        ):
        """
        Params:
            depth_cost_func: function for computing local cost volume
            inv_depth: inverse depth map
        """
        B = inv_depth.shape[0]

        if self.training:
            inv_depth_list = []
            conf_list = []

            gt_inv_depth = torch.where(torch.isinf(gt_inv_depth),
                                       inv_init_depth, gt_inv_depth)
            gt_delta_inv_depth = gt_inv_depth - inv_depth
            gt_delta_inv_depth = gt_delta_inv_depth.detach()

            t = torch.randint(0, self.timesteps, (B,), device=inv_depth.device).long()
            noise = (self.scale * torch.randn_like(gt_delta_inv_depth)).float()

            delta_inv_depth = self.q_sample(x_start=gt_delta_inv_depth, t=t, noise=noise)
            inv_depth_new = inv_depth + delta_inv_depth
            inv_depth_new = torch.clamp(inv_depth_new, min=0, max=1)
            delta_inv_depth = inv_depth_new - inv_depth

            confidence = None
            for i in range(self.iters):
                delta_inv_depth = delta_inv_depth.detach()
                if confidence is not None:
                    confidence = confidence.detach()
                inv_depth_new = inv_depth_new.detach()
                
                cost, inverse_depth_samples = depth_cost_func(inv_depth_new, 
                                                    confidence=confidence)
                input_features = self.encoder(inv_depth_new, 
                                              inverse_depth_samples, cost)

                input_unet = torch.cat([context, input_features], dim=1)
                hidden, update, confidence = self.unet(input_unet, hidden, t)
                confidence = confidence.squeeze(1)
                delta_inv_depth = delta_inv_depth + update
                conf_list.append(confidence)

                inv_depth_new = inv_depth + delta_inv_depth
                inv_depth_new = torch.clamp(inv_depth_new, min=0, max=1)
                delta_inv_depth = inv_depth_new - inv_depth
                inv_depth_list.append(inv_depth_new)

            mask = .25 * self.mask(context)
            return mask, hidden, inv_depth_list, conf_list

        else:
            batch, device, total_timesteps, sampling_timesteps, eta = B, inv_depth.device, self.timesteps, self.sampling_timesteps, self.ddim_sampling_eta

            times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1) 
            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), ..., (1, 0), (0, -1)]
            img = (self.scale * torch.randn_like(inv_depth)).float()
            mask = .25 * self.mask(context)

            for time, time_next in time_pairs:
                t = torch.full((batch,), time, device=device, dtype=torch.long)
                inv_depth_list = []
                conf_list = []
                delta_inv_depth = img
                inv_depth_new = inv_depth + delta_inv_depth
                inv_depth_new = torch.clamp(inv_depth_new, min=0, max=1)
                delta_inv_depth = inv_depth_new - inv_depth
                img = delta_inv_depth
                
                cur_hidden = hidden
                confidence = None
                for _ in range(self.iters):
                    cost, inverse_depth_samples = depth_cost_func(inv_depth_new, 
                                                    confidence=confidence)
                    input_features = self.encoder(inv_depth_new, 
                                                inverse_depth_samples,
                                                cost)
                    input_unet = torch.cat([context, input_features], dim=1)
                    cur_hidden, update, confidence = self.unet(input_unet, cur_hidden, t)
                    confidence = confidence.squeeze(1)
                    delta_inv_depth = delta_inv_depth + update
                    conf_list.append(confidence)

                    inv_depth_new = inv_depth + delta_inv_depth
                    inv_depth_new = torch.clamp(inv_depth_new, min=0, max=1)
                    delta_inv_depth = inv_depth_new - inv_depth
                    inv_depth_list.append(inv_depth_new)
                
                pred_noise = self.predict_noise_from_start(img, t, delta_inv_depth)

                if time_next < 0:
                    continue

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = (self.scale * torch.randn_like(inv_depth)).float()

                img = delta_inv_depth * alpha_next.sqrt() + \
                      c * pred_noise + \
                      sigma * noise

            return mask, cur_hidden, inv_depth_list, conf_list
