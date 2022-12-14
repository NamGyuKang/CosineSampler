import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name                = 'CosineSampler',
    version             = '0.0.1',
    description         = 'Triple backward custom CUDA kernel for interpolation',
    author              = 'Namgyu Kang',
    author_email        = 'kangnamgyu27@gmail.com',
    url                 = 'https://github.com/NamGyuKang/CosineSampler',
    ext_modules=[CUDAExtension('cosine_sampler_2d._cosine_2d', ['cosine_sampler_2d/csrc/cosine_sampler_2d_kernel.cu', 'cosine_sampler_2d/csrc/cosine_sampler_2d.cpp']), 
                    CUDAExtension('cosine_sampler_3d._cosine_3d', ['cosine_sampler_3d/csrc/cosine_sampler_3d_kernel.cu', 'cosine_sampler_3d/csrc/cosine_sampler_3d.cpp'])],
    packages=['cosine_sampler_2d', 'cosine_sampler_3d'],
    cmdclass={'build_ext': BuildExtension},
    keywords            = ['triple backward interpolation'],
    python_requires     = '>=3',
    classifiers         = [
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: C++",
		"Programming Language :: CUDA",
		"Programming Language :: Python :: 3 :: Only",    
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
)
