# CosineSampler

## Installation

On Python3 environment with Pytorch >=1.11 CUDA installation run:

```bash
pip install git+https://github.com/NamGyuKang/CosineSampler.git
```

## Usage

You can choose the kernel (cosine, linear, smoothstep), and the offset (True, False).
The offset is used in [PIXEL](https://github.com/NamGyuKang/PIXEL), and if you set the offset False, and linear kernel,
it is the same with Pytorch grid_sample and our CosineSampler support triple backpropagation of kernel.

## Compare CUDA with Pytorch

<img width="{90%}" src="{https://user-images.githubusercontent.com/70684829/206614851-86fc3382-f30c-4305-acea-b4b087cd3a5b.png}"/>

# Citation
If you use this code in your research, please consider citing:

```
@article{kang2023pixel,
title={PIXEL: Physics-Informed Cell Representations for Fast and Accurate PDE Solvers},
author={Kang, Namgyu and Lee, Byeonghyeon and Hong, Youngjoon and Yun, Seok-Bae and Park, Eunbyung},
journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
year={2023}}
                    
```

