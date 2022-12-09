# CosineSampler

#### We implemented the 2D, and 3D customized CUDA kernel of the triple backward grid sampler that supports cosine, linear, and smoothstep kernel [(Thomas Müller)](https://nvlabs.github.io/instant-ngp/) and third-order gradients $u_{xxc}, u_{yyc}$ with second-order gradients [(Tymoteusz Bleja)](https://github.com/tymoteuszb/smooth-sampler.git). As a result, the runtime and the memory requirement were significantly reduced. It is used in https://github.com/NamGyuKang/PIXEL

## Installation

The code is tested with Python3 environment (3.8, 3.9) and PyTorch (1.11, 11.2) with CUDA (>=11.3).

```bash
pip install git+https://github.com/NamGyuKang/CosineSampler.git
```

## Usage

You can choose the kernel (cosine, linear, smoothstep), and the offset (True, False).
The offset is used in [PIXEL (Physics-Informed Cell Representation)](https://github.com/NamGyuKang/PIXEL), and if you set the offset False, and linear kernel,
it is the same with Pytorch grid_sample and our CosineSampler support triple backpropagation of kernel.

## Compare CUDA with Pytorch
Second-order PDE (Helmholtz equation)
<img width="100%" src="https://user-images.githubusercontent.com/70684829/206615560-77b358ff-59d7-4ef8-af99-48312514c1ab.png"/>

# Citation
If you use this code in your research, please consider citing:

```
@article{kang2023pixel,
title={PIXEL: Physics-Informed Cell Representations for Fast and Accurate PDE Solvers},
author={Kang, Namgyu and Lee, Byeonghyeon and Hong, Youngjoon and Yun, Seok-Bae and Park, Eunbyung},
journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
year={2023}}
                    
```

