        Layer (type)               Output Shape         Param #
================================================================
         LayerNorm-1            [-1, 16384, 64]             128
            Linear-2            [-1, 16384, 64]           4,096
            Linear-3            [-1, 16384, 64]           4,096
            Linear-4            [-1, 16384, 64]           4,096
            Linear-5            [-1, 16384, 64]           4,096
   SwishFeatureMap-6         [-1, 16384, 4, 16]               0
   SwishFeatureMap-7         [-1, 16384, 4, 16]               0
GatedSlotAttention-8          [[-1, 16384, 64]]               0
          Identity-9            [-1, 16384, 64]               0
        LayerNorm-10            [-1, 16384, 64]             128
           Linear-11           [-1, 16384, 256]          16,640
             GELU-12           [-1, 16384, 256]               0
          Dropout-13           [-1, 16384, 256]               0
         Identity-14           [-1, 16384, 256]               0
           Linear-15            [-1, 16384, 64]          16,448
          Dropout-16            [-1, 16384, 64]               0
              Mlp-17            [-1, 16384, 64]               0
         Identity-18            [-1, 16384, 64]               0
================================================================
Total params: 49,728
Trainable params: 49,728
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 4.00 [64, 128, 128]
Forward/backward pass size (MB): 240.00
Params size (MB): 0.19
Estimated Total Size (MB): 244.19
----------------------------------------------------------------