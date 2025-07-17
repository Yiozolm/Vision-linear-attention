from vla.Blocks.Base import BasicBlock
from fla.layers.comba import Comba
from torchsummary import summary

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    model = BasicBlock(
        dim=64,
        hidden_rate=4,
        SpatialMixer=Comba
    ).cuda()

    summary(model, (64, 128, 128), device='cuda')

'''
Write your own summary
- [x]   Fox
- [ ]   DeltaNet **fp32**
- [x]   HGRN
- [ ]   HGRN2 **numhead_param**
- [ ]   lightnet **numhead_param**
- [x]   Gsa **slow**
- [x]   Mamba
- [x]   MesaNet
- [x]   Rodimus **chunk_gla Problem**
- [x]   Rwkv6 
- [ ]   Rwkv7 need v_first
- [ ]   Comba **IndexError: list index out of range**
'''