from vla.Blocks.Base import BasicBlock
from fla.layers.multiscale_retention import MultiScaleRetention
from torchsummary import summary

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    model = BasicBlock(
        dim=64,
        hidden_rate=4,
        SpatialMixer=MultiScaleRetention
    ).cuda()

    summary(model, (64, 128, 128), device='cuda')

'''
Write your own summary
- [ ]   ABC **triton.compiler.errors.CompilationError: at 26:14:**
- [ ]   Based
- [ ]   Comba **triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 147456, Hardware limit: 101376. Reducing block sizes or `num_stages` may help.**
- [x]   DeltaNet **change your model & data to bfloat16**
- [x]   DeltaProduct **change your model & data to bfloat16**
- [x]   Fox
- [x]   GLA
- [x]   GSA **slow**
- [x]   HGRN
- [x]   HGRN2 **change your input channel to 128$\cdot$k **
- [x]   Lightnet **numhead_param**
- [x]   Mamba
- [x]   MesaNet
- [ ]   PathAttn
- [ ]   ReBased
- [x]   RetNet **Crazy digits**
- [x]   Rodimus **chunk_gla Problem**
- [x]   Rwkv6 
- [ ]   Rwkv7 need v_first
'''