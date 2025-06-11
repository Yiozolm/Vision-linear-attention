from vla.Blocks.Vmamba import VmambaBlock
import torch
from torchsummary import summary

if __name__ == "__main__":
    model = VmambaBlock(
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
         LayerNorm-1         [-1, 128, 128, 64]             128
            Linear-2        [-1, 128, 128, 128]           8,192
              SiLU-3         [-1, 128, 128, 64]               0
           Permute-4         [-1, 64, 128, 128]               0
            Conv2d-5         [-1, 64, 128, 128]             640
           Permute-6         [-1, 128, 128, 64]               0
              SiLU-7         [-1, 128, 128, 64]               0
         LayerNorm-8         [-1, 128, 128, 64]             128
              GELU-9         [-1, 128, 128, 64]               0
           Linear-10         [-1, 128, 128, 64]           4,096
         Identity-11         [-1, 128, 128, 64]               0
      VmambaMixer-12         [-1, 128, 128, 64]               0
         Identity-13         [-1, 128, 128, 64]               0
        LayerNorm-14         [-1, 128, 128, 64]             128
           Linear-15        [-1, 128, 128, 256]          16,640
             GELU-16        [-1, 128, 128, 256]               0
          Dropout-17        [-1, 128, 128, 256]               0
         Identity-18        [-1, 128, 128, 256]               0
           Linear-19         [-1, 128, 128, 64]          16,448
          Dropout-20         [-1, 128, 128, 64]               0
              Mlp-21         [-1, 128, 128, 64]               0
         Identity-22         [-1, 128, 128, 64]               0
================================================================
Total params: 46,400
Trainable params: 46,400
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 4.00
Forward/backward pass size (MB): 280.00
Params size (MB): 0.18
Estimated Total Size (MB): 284.18
----------------------------------------------------------------
'''