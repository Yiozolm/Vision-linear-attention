# Vision Linear Attention

This repo aims at providing a collection of vision linear attention models

## ToDo
- [x] Vmamba 
- [x] MambaVision 
- [x] Vision-Rwkv **add init mode in the future**
- [ ] GSPN **torch & triton kernel**
- [ ] GroupMamba **To be tested**

## Installation
The following requirements should be satisfied
- [PyTorch](https://pytorch.org/) >= 2.5 (CUDA>=12.4)
- [Triton](https://github.com/openai/triton) >= 3.0
- [Mamba-ssm](https://github.com/state-spaces/mamba) (**Manual** installation from .whl files)
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
| 2025 | CVPR    | GSPN        | [Parallel Sequence Modeling via Generalized Spatial Propagation Network](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Parallel_Sequence_Modeling_via_Generalized_Spatial_Propagation_Network_CVPR_2025_paper.html) | [official](https://github.com/NVlabs/GSPN) |
| 2025 | CVPR    | GSPN        | [GroupMamba: Parameter-Efficient and Accurate Group Visual State Space Model](https://openaccess.thecvf.com/content/CVPR2025/html/Shaker_GroupMamba_Efficient_Group-Based_Visual_State_Space_Model_CVPR_2025_paper.html) | [official](https://github.com/Amshaker/GroupMamba) |


## Acknowledgments
We would like to express our deepest respect to [Songlin](https://github.com/sustcsonglin) and other maintainers of the [fla](https://github.com/fla-org/flash-linear-attention) library.