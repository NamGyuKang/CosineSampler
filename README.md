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
