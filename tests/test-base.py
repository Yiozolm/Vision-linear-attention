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
- [x]   ABC **triton.compiler.errors.CompilationError: at 26:14:**
- [x]   Based
- [ ]   Comba **triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 147456, Hardware limit: 101376. Reducing block sizes or `num_stages` may help.**
- [x]   DeltaNet
- [x]   DeltaProduct
- [x]   Fox
- [x]   Gated DeltaNet
- [x]   GLA
- [x]   GSA **slow**
- [x]   HGRN
- [x]   HGRN2
- [x]   Lightnet 
- [x]   Mamba
- [x]   MesaNet
- [x]   PathAttn
- [x]   ReBased
- [x]   RetNet 
- [x]   Rodimus 
- [x]   Rwkv6 
- [ ]   Rwkv7 need v_first
'''