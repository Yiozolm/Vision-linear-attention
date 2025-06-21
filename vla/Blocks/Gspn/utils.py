import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'normalize_w',
    'GRN',
    'Linear2d',
    'LayerNorm2d',
    'GateRecurrent2dnoind',
]


def normalize_w(Gl, Gm, Gr):
    Gl_s = torch.sigmoid(Gl)
    Gm_s = torch.sigmoid(Gm)
    Gr_s = torch.sigmoid(Gr)

    sum_s = Gl_s + Gm_s + Gr_s

    sum_s[:, :, 0, :] = Gm_s[:, :, 0, :] + Gr_s[:, :, 0, :]
    sum_s[:, :, -1, :] = Gl_s[:, :, -1, :] + Gm_s[:, :, -1, :]

    sum_s = sum_s.clamp(min=1e-7)

    return Gl_s / sum_s, Gm_s / sum_s, Gr_s / sum_s


class Linear2d(nn.Linear):
    def forward(self, x: torch.Tensor):
        # B, C, H, W = x.shape
        return F.conv2d(x, self.weight[:, :, None, None], self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        state_dict[prefix + "weight"] = state_dict[prefix + "weight"].view(self.weight.shape)
        return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                             error_msgs)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor):
        Gx = torch.norm(x, p=2, dim=[2, 3], keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


def gaterecurrent2dnoind_pytorch(X, B, G1, G2, G3):
    """PyTorch implementation of GateRecurrent2dnoind"""
    batch_size, channels, height, width = X.size()
    H = torch.zeros_like(X)

    # Forward pass from left to right
    for w in range(width):
        for h in range(height):
            # Get current inputs
            x_t = X[..., h, w]
            b_t = B[..., h, w]

            # Calculate gated connections from previous positions
            if w > 0:
                # Top-left connection (h-1, w-1)
                if h > 0:
                    h1_prev = H[..., h - 1, w - 1].clone()
                    g1 = G1[..., h, w]  # Gate from current position
                    h1_gated = g1 * h1_prev
                else:
                    h1_gated = 0

                # Left connection (h, w-1)
                h2_prev = H[..., h, w - 1].clone()
                g2 = G2[..., h, w]  # Gate from current position
                h2_gated = g2 * h2_prev

                # Bottom-left connection (h+1, w-1)
                if h < height - 1:
                    h3_prev = H[..., h + 1, w - 1].clone()
                    g3 = G3[..., h, w]  # Gate from current position
                    h3_gated = g3 * h3_prev
                else:
                    h3_gated = 0

                # Combine all gated connections
                h_sum = h1_gated + h2_gated + h3_gated
            else:
                h_sum = 0

            # Update current hidden state
            H[..., h, w] = b_t * x_t + h_sum

    return H


class GateRecurrent2dnoind(nn.Module):
    def __init__(self, items_each_chunk_, backend='pytorch'):
        super(GateRecurrent2dnoind, self).__init__()
        self.items_each_chunk = items_each_chunk_
        assert backend in ['cuda'], f"Backend {backend} not supported"  # , 'triton', 'pytorch'
        self.backend = backend

    def forward(self, X, B, G1, G2, G3):
        if self.backend == 'pytorch':
            return gaterecurrent2dnoind_pytorch(X, B, G1, G2, G3)
        # elif self.backend == 'triton':
        #     return gaterecurrent_triton(X, B, G1, G2, G3, self.items_each_chunk)
        # else:  # cuda backend
        #     return gaterecurrent(X, B, G1, G2, G3, self.items_each_chunk)
        else:
            raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(backend={self.backend})"
