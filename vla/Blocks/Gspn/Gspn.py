# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch.nn as nn
# from timm.layers import Mlp

from einops import rearrange
from vla.ops import DropPath

from .utils import Linear2d, LayerNorm2d, GRN, normalize_w, GateRecurrent2dnoind, Mlp


class Gspn(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            feat_size,
            items_each_chunk=8,
            d_model=96,
            expand=2.0,
            d_state=16,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # ======================
            n_directions=4,
            # ======================
            channel_first=True,
            is_glayers=False,
            force_fp32=True,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()

        self.d_state = d_state
        self.channel_first = channel_first

        self.c_group = 12
        self.n_directions = n_directions

        d_inner = int(expand * d_model)

        Linear = Linear2d if channel_first else nn.Linear
        LayerNorm = LayerNorm2d if channel_first else nn.LayerNorm


        if is_glayers:
            self.items_each_chunk = feat_size
        else:
            self.items_each_chunk = items_each_chunk

        # in proj =======================================
        d_proj = d_inner
        self.in_proj = Linear(d_model, d_proj, bias=bias)
        self.act: nn.Module = act_layer()

        # conv =======================================
        self.d_spn = d_inner

        self.conv2d = nn.Conv2d(
            in_channels=self.d_spn,
            out_channels=self.d_spn,
            groups=self.d_spn,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        # spn module ============================
        self.spn_core = GateRecurrent2dnoind(self.items_each_chunk)

        # w, l, u proj ============================
        ks = 1
        self.x_conv_down = nn.Conv2d(self.d_spn, self.d_state, kernel_size=ks, padding=(ks - 1) // 2, bias=False)
        self.w_conv_up = nn.Conv2d(self.d_state, self.c_group * self.d_spn, kernel_size=ks, padding=(ks - 1) // 2,
                                   bias=False)
        self.l_conv_up = nn.Conv2d(self.d_state, self.n_directions * self.d_spn, kernel_size=ks, padding=(ks - 1) // 2,
                                   bias=False)
        self.u_conv_up = nn.Conv2d(self.d_state, self.n_directions * self.d_spn, kernel_size=ks, padding=(ks - 1) // 2,
                                   bias=False)
        self.d_conv = nn.Conv2d(self.d_state, self.n_directions * self.d_spn, kernel_size=ks, padding=(ks - 1) // 2,
                                bias=False)
        self.m_conv = nn.Conv2d(self.n_directions, 1, kernel_size=1, bias=False)
        # out proj =======================================
        self.grn = GRN(d_inner)
        self.out_act = nn.Identity()
        self.out_norm = LayerNorm(self.d_spn)
        self.out_proj = Linear(d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def spn_block(self, X, l, u, Gl, Gm, Gr, D=None, spn_module=None):
        Gl, Gm, Gr = normalize_w(Gl, Gm, Gr)

        Gl = Gl.to(X.dtype)
        Gm = Gm.to(X.dtype)
        Gr = Gr.to(X.dtype)

        out = spn_module(X, l, Gl, Gm, Gr)
        if D is not None:
            out = out * u + X * D
        else:
            out = out * u
        return out

    def forward_core(self, x: torch.Tensor = None, **kwargs):
        B, D, H, W = x.shape

        x_proxy = self.x_conv_down(x)
        ws = self.w_conv_up(x_proxy)
        Ls = self.l_conv_up(x_proxy).contiguous()
        Us = self.u_conv_up(x_proxy).contiguous()
        Ds = self.d_conv(x_proxy).contiguous()

        x_hwwh = torch.stack([x, x.transpose(2, 3).contiguous()], dim=1)
        xs = torch.cat([x_hwwh, x_hwwh.flip(dims=[-1]).contiguous()], dim=1)  # (b, k, d, h, w)
        xs = xs.view(B, -1, H, W)  # (b, k, d, h, w)
        xs = xs.contiguous()

        Gs = torch.split(ws, D * self.n_directions, dim=1)  # 3 * (b, d, h, w)
        G3 = [g.contiguous() for g in Gs]

        out_y = self.spn_block(xs, Ls, Us, G3[0], G3[1], G3[2], Ds, self.spn_core)

        out_y = out_y.view(B, self.n_directions, D * H, W)
        out_y = self.m_conv(out_y).view(B, D, H, W)

        y = self.out_norm(out_y)

        return y

    def forward(self, x: torch.Tensor):
        x = self.in_proj(x)
        x = self.conv2d(x)  # (b, d, h, w)
        y = self.forward_core(x)
        y = self.out_act(y)
        y = self.grn(y)
        out = self.dropout(self.out_proj(y))
        return out


class GSPNBlock(nn.Module):
    '''
    Generalized Spatial Propagation Network proposed by NVIDIA
    Hongjun Wang, Wonmin Byeon, Jiarui Xu, Jinwei Gu, Ka Chun Cheung, Xiaolong Wang, Kai Han, Jan Kautz, Sifei Liu.
    "Parallel Sequence Modeling via Generalized Spatial Propagation Network" In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 4473-4483. https://doi.org/10.48550/arXiv.2501.12381.
    '''
    def __init__(
            self,
            feat_size,
            items_each_chunk,
            dim,
            hidden_rate=4,
            mlp_drop=0.,
            drop_path=0.,
            init_value=None,
            act_layer=nn.GELU):
        super().__init__()
        self.mixer = Gspn(feat_size=feat_size, items_each_chunk=items_each_chunk,d_model=dim, d_state=8, d_conv=3, expand=1)
        hidden_dim = int(dim * hidden_rate)
        self.mlp = Mlp(in_features=dim, hidden_features=hidden_dim, act_layer=act_layer, drop=mlp_drop, channels_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale = (init_value is not None)
        if self.layer_scale:
            # original MambaVisionMixer doesn't require gradient
            self.gamma_1 = nn.Parameter(init_value * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        assert x.dim() == 4, 'Invalid dimension'
        print(x.size())

        if self.layer_scale:
            x = x + self.drop_path(self.gamma_1 * self.mixer(self.ln1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.ln2(x)))
        else:
            x = x + self.drop_path(self.mixer(x))
            print(x.size())
            x = x + self.drop_path(self.mlp(x))

        return x
