# import torch
# from setuptools import setup
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# setup(
#     name                = 'CosineSampler',
#     version             = '0.0.1',
#     description         = 'Triple backward custom CUDA kernel for interpolation',
#     author              = 'Namgyu Kang',
#     author_email        = 'kangnamgyu27@gmail.com',
#     url                 = 'https://github.com/NamGyuKang/CosineSampler',
#     ext_modules=[CUDAExtension('cosine_sampler_2d._cosine_2d', ['cosine_sampler_2d/csrc/cosine_sampler_2d_kernel.cu', 'cosine_sampler_2d/csrc/cosine_sampler_2d.cpp']), 
#                     CUDAExtension('cosine_sampler_3d._cosine_3d', ['cosine_sampler_3d/csrc/cosine_sampler_3d_kernel.cu', 'cosine_sampler_3d/csrc/cosine_sampler_3d.cpp'])],
#     packages=['cosine_sampler_2d', 'cosine_sampler_3d'],
#     cmdclass={'build_ext': BuildExtension},
#     keywords            = ['triple backward interpolation'],
#     python_requires     = '>=3',
#     classifiers         = [
#         "Environment :: GPU :: NVIDIA CUDA",
#         "Topic :: Utilities",
#         "License :: OSI Approved :: BSD License",
#         "Programming Language :: C++",
# 		"Programming Language :: CUDA",
# 		"Programming Language :: Python :: 3 :: Only",    
#         "Topic :: Scientific/Engineering :: Artificial Intelligence",
#         ],
# )

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

extra_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "-Xcompiler", "-fPIC",
    ],
}

setup(
    name="CosineSampler",
    version="0.0.1",
    description="Triple backward custom CUDA kernel for interpolation",
    author="Namgyu Kang",
    author_email="kangnamgyu27@gmail.com",
    license="BSD License",
    url="https://github.com/NamGyuKang/CosineSampler",
    packages=find_packages(exclude=("test*",)),
    ext_modules=[
        CUDAExtension(
            "cosine_sampler_2d._cosine_2d",
            [
                "cosine_sampler_2d/csrc/cosine_sampler_2d_kernel.cu",
                "cosine_sampler_2d/csrc/cosine_sampler_2d.cpp",
            ],
            extra_compile_args=extra_compile_args,
        ),
        CUDAExtension(
            "cosine_sampler_3d._cosine_3d",
            [
                "cosine_sampler_3d/csrc/cosine_sampler_3d_kernel.cu",
                "cosine_sampler_3d/csrc/cosine_sampler_3d.cpp",
            ],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    keywords = ["triple backward interpolation", "CUDA", "interpolation", "torch", "grid sampler", "cosine kernel"],
    python_requires=">=3",
    classifiers=[
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: C++",
		"Programming Language :: CUDA",
		"Programming Language :: Python :: 3 :: Only",    
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,
)