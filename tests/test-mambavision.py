from vla.Blocks.MambaVisionMixer import MambaVisionBlock
import torch
from torchsummary import summary

if __name__=="__main__":
    model = MambaVisionBlock(
        dim=64,
    ).cuda()

    # a = torch.randn([1, 64, 128, 128]).cuda()
    # b = model(a)
    # print(b.shape)
    summary(model, (64, 128, 128), device='cuda')

'''
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
         LayerNorm-1            [-1, 16384, 64]             128
            Linear-2            [-1, 16384, 64]           4,096
            Linear-3                   [-1, 20]             640
            Linear-4                   [-1, 32]             160
            Linear-5            [-1, 16384, 64]           4,096
  MambaVisionMixer-6            [-1, 16384, 64]               0
          Identity-7            [-1, 16384, 64]               0
         LayerNorm-8            [-1, 16384, 64]             128
            Linear-9           [-1, 16384, 256]          16,640
             GELU-10           [-1, 16384, 256]               0
          Dropout-11           [-1, 16384, 256]               0
         Identity-12           [-1, 16384, 256]               0
           Linear-13            [-1, 16384, 64]          16,448
          Dropout-14            [-1, 16384, 64]               0
              Mlp-15            [-1, 16384, 64]               0
         Identity-16            [-1, 16384, 64]               0
================================================================
Total params: 42,336
Trainable params: 42,336
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 4.00
Forward/backward pass size (MB): 208.00
Params size (MB): 0.16
Estimated Total Size (MB): 212.16
----------------------------------------------------------------
'''