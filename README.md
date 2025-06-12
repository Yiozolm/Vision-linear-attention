# Vision Linear Attention

**Note!**: This branch aims at building vls module on window pc for short time debugging not long-term stable training

This repo aims at providing a collection of vision linear attention models

## ToDo
- [ ] Vmamba **mamba_ssm to be solved**
- [x] MambaVision 
- [x] Vision-Rwkv **add init mode in the future**

## Windows Testing Enviroment
- Win11 23H2
- CUDA12.8
- torch2.7.1
- Visual Studio 2022
- GPU: RTX3060ti

## Installation
The following requirements should be satisfied
- [PyTorch](https://pytorch.org/) >= 2.5 (CUDA>=12.4)
- [Triton-windows](https://github.com/woct0rdho/triton-windows) >= 3.0
- [einops](https://github.com/arogozhnikov/einops)
- [Timm](https://github.com/huggingface/pytorch-image-models)

You can install `vla` with pip:
```bash
git clone https://github.com/Yiozolm/Vision-linear-attention.git
cd Vision-linear-attention
pip install -e .
```
## Models

| Year | Venue   | Model       | Paper                                                                                                                  | Code                                                 | 
|:-----|:--------|:------------|:-----------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------| 
| 2024 | NeurIPS | Vmamba      | [VMamba: Visual State Space Model](https://arxiv.org/abs/2401.10166)                                                   | [official](https://github.com/MzeroMiko/VMamba)      | 
| 2025 | ICLR    | MambaVision | [MambaVision: A Hybrid Mamba-Transformer Vision Backbone](https://arxiv.org/abs/2407.08083)                            | [official](https://github.com/NVlabs/MambaVision)    | 
| 2025 | ICLR    | Vision-Rwkv | [Vision-RWKV: Efficient and Scalable Visual Perception with RWKV-Like Architectures](https://arxiv.org/abs/2403.02308) | [official](https://github.com/OpenGVLab/Vision-RWKV) | 

## Acknowledgments
We would like to express our deepest respect to [Songlin](https://github.com/sustcsonglin) and other maintainers of the [fla](https://github.com/fla-org/flash-linear-attention) library.