from vla.Blocks.Gspn import GSPNBlock
from torchsummary import summary

if __name__ == "__main__":
    model = GSPNBlock(
        feat_size=56,
        items_each_chunk=8,
        dim=64,
    ).cuda()

    summary(model, (64, 128, 128), device='cuda')
'''
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
          Linear2d-1         [-1, 64, 128, 128]           4,096
            Conv2d-2         [-1, 64, 128, 128]             640
            Conv2d-3          [-1, 8, 128, 128]             512
            Conv2d-4        [-1, 768, 128, 128]           6,144
            Conv2d-5        [-1, 256, 128, 128]           2,048
            Conv2d-6        [-1, 256, 128, 128]           2,048
            Conv2d-7        [-1, 256, 128, 128]           2,048
GateRecurrent2dnoind-8        [-1, 256, 128, 128]               0
            Conv2d-9         [-1, 1, 8192, 128]               4
      LayerNorm2d-10         [-1, 64, 128, 128]             128
         Identity-11         [-1, 64, 128, 128]               0
              GRN-12         [-1, 64, 128, 128]               0
         Linear2d-13         [-1, 64, 128, 128]           4,096
         Identity-14         [-1, 64, 128, 128]               0
             Gspn-15         [-1, 64, 128, 128]               0
         Identity-16         [-1, 64, 128, 128]               0
         Linear2d-17        [-1, 256, 128, 128]          16,640
             GELU-18        [-1, 256, 128, 128]               0
          Dropout-19        [-1, 256, 128, 128]               0
         Linear2d-20         [-1, 64, 128, 128]          16,448
          Dropout-21         [-1, 64, 128, 128]               0
              Mlp-22         [-1, 64, 128, 128]               0
         Identity-23         [-1, 64, 128, 128]               0
================================================================
Total params: 54,852
Trainable params: 54,852
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 4.00
Forward/backward pass size (MB): 433.00
Params size (MB): 0.21
Estimated Total Size (MB): 437.21
----------------------------------------------------------------
'''