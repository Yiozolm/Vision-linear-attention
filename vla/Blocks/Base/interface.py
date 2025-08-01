# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import Mlp

from vla.ops import DropPath

# A Unified Interface for Linear Attentions in fla

__all__ = ['BasicBlock']


class BasicBlock(nn.Module):
    def __init__(
            self,
            dim,
            hidden_rate,
            SpatialMixer: nn.Module,
            mlp_drop=0.,
            drop_rate=0.,
            init_value=None,
            act_layer=nn.GELU,
            **kwargs):
        super(BasicBlock, self).__init__()
        mixer_args = {'hidden_size': dim}
        if 'expand_ratio' in kwargs:
            mixer_args['expand_ratio'] = kwargs['expand_ratio']
        if 'num_heads' in kwargs:
            mixer_args['num_heads'] = kwargs['num_heads']
        if 'num_key_value_heads' in kwargs:
            mixer_args['num_key_value_heads'] = kwargs['num_key_value_heads']
        self.SpatialMixer = SpatialMixer(**mixer_args)

        hidden_dim = int(dim * hidden_rate)
        self.ChannelMixer = Mlp(in_features=dim, hidden_features=hidden_dim, out_features=dim, act_layer=act_layer,
                                drop=mlp_drop)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_prob=drop_rate) if drop_rate > 0 else nn.Identity()
        self.layer_scale = (init_value is not None)
        if self.layer_scale:
            self.gamma_1 = nn.Parameter(init_value * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_value * torch.ones(dim))

    def forward(
            self,
            x: torch.Tensor,
            resolution: Optional[Tuple] = None
    ) -> torch.Tensor:
        assert x.dim() in [3, 4], 'Invalid dimension'
        xdim = x.dim()
        if xdim == 4:
            _, _, H, W = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c')

        if self.layer_scale:
            x = x + self.drop_path(self.gamma_1 * self.SpatialMixer(self.ln1(x))[0])
            x = x + self.drop_path(self.gamma_2 * self.ChannelMixer(self.ln2(x)))
        else:
            x = x + self.drop_path(self.SpatialMixer(self.ln1(x))[0])
            x = x + self.drop_path(self.ChannelMixer(self.ln2(x)))

        if xdim == 4:
            x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x