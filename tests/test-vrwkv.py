from vla.Blocks.Vrwkv import Vrwkv
from torchsummary import summary

if __name__ == "__main__":
    model = Vrwkv(
        dim=64,
    ).cuda()

    summary(model, (64, 128, 128), device='cuda')

'''
Device: RTX4090
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
         LayerNorm-1            [-1, 16384, 64]             128
            Linear-2            [-1, 16384, 64]           4,096
            Linear-3            [-1, 16384, 64]           4,096
            Linear-4            [-1, 16384, 64]           4,096
            Linear-5            [-1, 16384, 64]           4,096
        SpatialMix-6            [-1, 16384, 64]               0
         LayerNorm-7            [-1, 16384, 64]             128
            Linear-8           [-1, 16384, 256]          16,384
            Linear-9            [-1, 16384, 64]          16,384
           Linear-10            [-1, 16384, 64]           4,096
       ChannelMix-11            [-1, 16384, 64]               0
================================================================
Total params: 53,504
Trainable params: 53,504
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 4.00
Forward/backward pass size (MB): 112.00
Params size (MB): 0.20
Estimated Total Size (MB): 116.20
----------------------------------------------------------------
'''