import math

import torch
import torch.nn as nn

from einops import rearrange
from timm.layers import Mlp

from .utils import cross_scan_fn, selective_scan_chunk_fn, cross_merge_fn

from vla.ops import DropPath


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class VmambaMixer(nn.Module):
    def __init__(self,
                 d_model,
                 d_state,
                 d_conv,
                 expand=2,
                 dt_rank="auto",
                 dropout=0.0,
                 conv_bias=True,
                 bias=False):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_proj = self.d_inner * 2

        k_group = 4
        self.in_proj = nn.Linear(self.d_model, self.d_proj, bias=bias)
        self.x_proj = [
            nn.Linear(self.d_inner, int(self.dt_rank + self.d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj
        self.act = nn.SiLU()

        # out proj =======================================
        self.out_act = nn.GELU()
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.Ds = nn.Parameter(torch.ones((k_group, self.dt_rank, int(self.d_inner //  self.dt_rank))))
        self.A_logs = nn.Parameter(torch.zeros((k_group,  self.dt_rank)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
        self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, self.dt_rank)))

        self.conv2d = nn.Sequential(
                Permute(0, 3, 1, 2),
                nn.Conv2d(
                    in_channels=self.d_inner,
                    out_channels=self.d_inner,
                    groups=self.d_inner,
                    bias=conv_bias,
                    kernel_size=d_conv,
                    padding=(d_conv - 1) // 2,
                ),
                Permute(0, 2, 3, 1),
            )
        self.out_norm = nn.LayerNorm(self.d_inner)

    def forward_core(self, x):
        B, H, W, RD = x.shape
        K, R, D = self.Ds.shape
        assert RD == R * D
        L = H * W
        KR = K * R
        x_proj_bias = getattr(self, "x_proj_bias", None)

        initial_state = None
        xs = cross_scan_fn(x.view(B, H, W, RD), in_channel_first=False, out_channel_first=False, scans=0,
                           force_torch=False)
        x_dbl = torch.einsum("b l k d, k c d -> b l k c", xs, self.x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, -1, K, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, self.d_state, self.d_state], dim=3)
        xs = xs.contiguous().view(B, L, KR, D)
        dts = dts.contiguous().view(B, L, KR)
        Bs = Bs.contiguous().view(B, L, K, self.d_state)
        Cs = Cs.contiguous().view(B, L, K, self.d_state)

        As = -self.A_logs.to(torch.float).exp().view(KR)
        Ds = self.Ds.to(torch.float).view(KR, D)
        dt_bias = self.dt_projs_bias.view(KR)

        ys, final_state = selective_scan_chunk_fn(
            xs, dts, As, Bs, Cs, chunk_size=64, D=Ds, dt_bias=dt_bias,
            initial_states=initial_state, dt_softplus=True, return_final_states=True,
            backend='triton',
        )
        y: torch.Tensor = cross_merge_fn(ys.view(B, H, W, K, RD), in_channel_first=False, out_channel_first=False,
                                         scans=0, force_torch=False)
        y = self.out_norm(y.view(B, H, W, -1))
        return y.to(x.dtype)

    def forward(self, x):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1)  # (b, h, w, d)
        z = self.act(z)
        x = self.conv2d(x)
        x = self.act(x)
        y = self.forward_core(x)
        y = self.out_act(y)
        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class VmambaBlock(nn.Module):
    def __init__(
            self,
            dim,
            hidden_rate=4,
            mlp_drop=0.,
            drop_path=0.,
            init_value=None,
            act_layer=nn.GELU):
        super().__init__()
        self.mixer = VmambaMixer(d_model=dim, d_state=8, d_conv=3, expand=1)
        hidden_dim = int(dim * hidden_rate)
        self.mlp = Mlp(in_features=dim, hidden_features=hidden_dim, act_layer=act_layer, drop=mlp_drop)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale = (init_value is not None)
        if self.layer_scale:
            # original MambaVisionMixer doesn't require gradient
            self.gamma_1 = nn.Parameter(init_value * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        assert x.dim()==4, 'Invalid dimension'
        # assume the input follows [b, c, h, w]
        x = rearrange(x, 'b c h w -> b h w c')

        if self.layer_scale:
            x = x + self.drop_path(self.gamma_1 * self.mixer(self.ln1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.ln2(x)))
        else:
            x = x + self.drop_path(self.mixer(self.ln1(x)))
            x = x + self.drop_path(self.mlp(self.ln2(x)))

        x = rearrange(x, 'b h w c -> b c h w')
        return x
