        Layer (type)               Output Shape         Param #
================================================================
         LayerNorm-1            [-1, 16384, 64]             128
            Linear-2            [-1, 16384, 64]           4,096
            Linear-3            [-1, 16384, 64]           4,096
            Linear-4            [-1, 16384, 64]           4,096
 RebasedFeatureMap-5         [-1, 16384, 4, 16]               0
 RebasedFeatureMap-6         [-1, 16384, 4, 16]               0
            Linear-7            [-1, 16384, 64]           4,096
          Identity-8            [-1, 16384, 64]               0
ReBasedLinearAttention-9            [-1, 16384, 64]               0
         Identity-10                   [-1, 64]               0
        LayerNorm-11            [-1, 16384, 64]             128
           Linear-12           [-1, 16384, 256]          16,640
             GELU-13           [-1, 16384, 256]               0
          Dropout-14           [-1, 16384, 256]               0
         Identity-15           [-1, 16384, 256]               0
           Linear-16            [-1, 16384, 64]          16,448
          Dropout-17            [-1, 16384, 64]               0
              Mlp-18            [-1, 16384, 64]               0
         Identity-19            [-1, 16384, 64]               0
================================================================
Total params: 49,728
Trainable params: 49,728
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 4.00
Forward/backward pass size (MB): 240.00
Params size (MB): 0.19
Estimated Total Size (MB): 244.19
----------------------------------------------------------------