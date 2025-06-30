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
       LayerNorm2d-1         [-1, 64, 128, 128]             128
          Linear2d-2         [-1, 64, 128, 128]           4,096
            Conv2d-3         [-1, 64, 128, 128]             640
            Conv2d-4          [-1, 8, 128, 128]             512
            Conv2d-5        [-1, 768, 128, 128]           6,144
            Conv2d-6        [-1, 256, 128, 128]           2,048
            Conv2d-7        [-1, 256, 128, 128]           2,048
            Conv2d-8        [-1, 256, 128, 128]           2,048
GateRecurrent2dnoind-9        [-1, 256, 128, 128]               0
           Conv2d-10         [-1, 1, 8192, 128]               4
      LayerNorm2d-11         [-1, 64, 128, 128]             128
         Identity-12         [-1, 64, 128, 128]               0
              GRN-13         [-1, 64, 128, 128]               0
         Linear2d-14         [-1, 64, 128, 128]           4,096
         Identity-15         [-1, 64, 128, 128]               0
             Gspn-16         [-1, 64, 128, 128]               0
         Identity-17         [-1, 64, 128, 128]               0
      LayerNorm2d-18         [-1, 64, 128, 128]             128
         Linear2d-19        [-1, 256, 128, 128]          16,640
             GELU-20        [-1, 256, 128, 128]               0
          Dropout-21        [-1, 256, 128, 128]               0
         Linear2d-22         [-1, 64, 128, 128]          16,448
          Dropout-23         [-1, 64, 128, 128]               0
              Mlp-24         [-1, 64, 128, 128]               0
         Identity-25         [-1, 64, 128, 128]               0
================================================================
Total params: 55,108
Trainable params: 55,108
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 4.00
Forward/backward pass size (MB): 449.00
Params size (MB): 0.21
Estimated Total Size (MB): 453.21
----------------------------------------------------------------
'''