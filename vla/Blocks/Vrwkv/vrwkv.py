import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from functools import lru_cache

from einops import rearrange

__all__ = ['RwkvBlock_BiV4']


@lru_cache(maxsize=1)
# def _load_vrwkv_cuda_kernel():
#     # arch有对应 86=3090 89=4090
#     abspath = os.path.dirname(os.path.abspath(__file__))
#     wkv_cuda = load(name="bi_wkv", sources=[os.path.join(abspath, "cuda/bi_wkv.cpp"),os.path.join(abspath, "cuda/bi_wkv_kernel.cu") ],
#                     verbose=True,
#                     extra_cuda_cflags=['-res-usage', '--maxrregcount=60', '--use_fast_math', '/02', '-Xptxas=/02'])
#     return wkv_cuda
def _load_vrwkv_cuda_kernel():
    abspath = os.path.dirname(os.path.abspath(__file__))

    # Corrected: List the renamed C++ file as a .cu file.
    # You must rename the file on your disk from bi_wkv.cpp to bi_wkv.cu
    source_files = [
        os.path.join(abspath, "cuda\\bi_wkv.cpp"),
        os.path.join(abspath, "cuda\\bi_wkv_kernel.cu")
    ]

    # --- Compiler Flags (these are correct from the last version) ---
    extra_cuda_cflags = [
        '-res-usage',
        '--maxrregcount=60',
        '--use_fast_math',
        '-Xptxas', '-O3',  # Note: Capital 'O', not zero '0'
    ]

    gencode_flags = [
        '-gencode', 'arch=compute_75,code=sm_75',
        '-gencode', 'arch=compute_80,code=sm_80',
        '-gencode', 'arch=compute_86,code=sm_86',
        '-gencode', 'arch=compute_89,code=sm_89',
        '-gencode', 'arch=compute_90,code=sm_90',
    ]
    extra_cuda_cflags.extend(gencode_flags)

    import sys
    if sys.platform == "win32":
        extra_cflags = ["/O2"]
    else:
        extra_cflags = ["-O3"]

    wkv_cuda = load(
        name="bi_wkv",
        sources=source_files,
        verbose=True,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags
    )
    return wkv_cuda

class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, u, k, v):
        wkv_cuda = _load_vrwkv_cuda_kernel()
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = wkv_cuda.bi_wkv_forward(w, u, k, v)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        wkv_cuda = _load_vrwkv_cuda_kernel()
        w, u, k, v = ctx.saved_tensors
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        gw, gu, gk, gv = wkv_cuda.bi_wkv_backward(w.float().contiguous(),
                                                  u.float().contiguous(),
                                                  k.float().contiguous(),
                                                  v.float().contiguous(),
                                                  gy.float().contiguous())
        if half_mode:
            return (gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            return (gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            return (gw, gu, gk, gv)


def RUN_CUDA(w, u, k, v):
    return WKV.apply(w.cuda(), u.cuda(), k.cuda(), v.cuda())


def q_shift(input, shift_pixel=1, gamma=1 / 4, patch_resolution=None):
    # vanilla qshift in VisionRwkv, https://github.com/OpenGVLab/Vision-RWKV
    assert gamma <= 1 / 4
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, 0:int(C * gamma), :, shift_pixel:W] = input[:, 0:int(C * gamma), :, 0:W - shift_pixel]
    output[:, int(C * gamma):int(C * gamma * 2), :, 0:W - shift_pixel] = input[:, int(C * gamma):int(C * gamma * 2), :,
                                                                         shift_pixel:W]
    output[:, int(C * gamma * 2):int(C * gamma * 3), shift_pixel:H, :] = input[:, int(C * gamma * 2):int(C * gamma * 3),
                                                                         0:H - shift_pixel, :]
    output[:, int(C * gamma * 3):int(C * gamma * 4), 0:H - shift_pixel, :] = input[:,
                                                                             int(C * gamma * 3):int(C * gamma * 4),
                                                                             shift_pixel:H, :]
    output[:, int(C * gamma * 4):, ...] = input[:, int(C * gamma * 4):, ...]
    return output


class OmniShift(nn.Module):
    # Reparameterized 5x5 depth-wise convolution,
    # from RestoreRWKV, https://github.com/Yaziwel/Restore-RWKV

    def __init__(self, dim):
        super(OmniShift, self).__init__()
        # Define the layers for training
        self.conv1x1 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=1, groups=dim, bias=False
        )
        self.conv3x3 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            groups=dim,
            bias=False,
        )
        self.conv5x5 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=5,
            padding=2,
            groups=dim,
            bias=False,
        )
        self.alpha = nn.Parameter(torch.randn(4), requires_grad=True)

        # Define the layers for testing
        self.conv5x5_reparam = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=5,
            padding=2,
            groups=dim,
            bias=False,
        )
        self.repram_flag = True

    def forward_train(self, x):
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x)

        out = (
                self.alpha[0] * x
                + self.alpha[1] * out1x1
                + self.alpha[2] * out3x3
                + self.alpha[3] * out5x5
        )
        return out

    def reparam_5x5(self):
        # Combine the parameters of conv1x1, conv3x3, and conv5x5 to form a single 5x5 depth-wise convolution

        padded_weight_1x1 = F.pad(self.conv1x1.weight, (2, 2, 2, 2))
        padded_weight_3x3 = F.pad(self.conv3x3.weight, (1, 1, 1, 1))
        identity_weight = F.pad(torch.ones_like(self.conv1x1.weight), (2, 2, 2, 2))

        combined_weight = (
                self.alpha[0] * identity_weight
                + self.alpha[1] * padded_weight_1x1
                + self.alpha[2] * padded_weight_3x3
                + self.alpha[3] * self.conv5x5.weight
        )
        device = self.conv5x5_reparam.weight.device
        combined_weight = combined_weight.to(device)
        self.conv5x5_reparam.weight = nn.Parameter(combined_weight)

    def forward(self, x):
        if self.training:
            self.repram_flag = True
            out = self.forward_train(x)
        elif self.training is False and self.repram_flag is True:
            self.reparam_5x5()
            self.repram_flag = False
            out = self.conv5x5_reparam(x)
        elif self.training is False and self.repram_flag is False:
            out = self.conv5x5_reparam(x)

        return out


class SpatialMix(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        attn_dim = dim

        self.shift = q_shift
        self.key = nn.Linear(dim, attn_dim, bias=False)
        self.value = nn.Linear(dim, attn_dim, bias=False)
        self.receptance = nn.Linear(dim, attn_dim, bias=False)
        self.output = nn.Linear(attn_dim, dim, bias=False)

        self.decay = nn.Parameter(torch.randn((self.dim,)))
        self.boost = nn.Parameter(torch.randn((self.dim,)))

    def jit_func(self, x, resolution):
        H, W = resolution
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W) 
        x = self.shift(x, patch_resolution=resolution)
        x = rearrange(x, "b c h w -> b (h w) c")

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, resolution):
        B, T, C = x.size()
        sr, k, v = self.jit_func(x, resolution)
        x = RUN_CUDA(self.decay / T, self.boost / T, k, v)
        x = sr * x
        x = self.output(x)
        return x


class ChannelMix(nn.Module):
    def __init__(self, dim, hidden_rate=4):
        super().__init__()
        self.n_embd = dim
        hidden_dim = int(hidden_rate * dim)

        self.shift = q_shift
        self.key = nn.Linear(dim, hidden_dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x, resolution):
        H, W = resolution
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        x = self.shift(x, patch_resolution=resolution)
        x = rearrange(x, "b c h w -> b (h w) c")

        k = self.key(x)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(x)) * kv
        return x


class RwkvBlock_BiV4(nn.Module):
    def __init__(self, dim, hidden_rate=4):
        super().__init__()

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.att = SpatialMix(dim)
        self.ffn = ChannelMix(dim, hidden_rate)
        self.gamma1 = nn.Parameter(torch.ones((dim)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        assert x.dim() == 4, 'Invalid dimension'
        B, C, H, W = x.shape
        resolution = (H, W)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = x + self.gamma1 * self.att(self.ln1(x), resolution)
        x = x + self.gamma2 * self.ffn(self.ln2(x), resolution)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        return x
