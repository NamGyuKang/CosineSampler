from setuptools import setup, find_packages
from torch.utils import cpp_extension

setup(
    name                = 'CosineSampler',
    version             = '0.0.1',
    description         = 'Triple backward custom CUDA kernel for interpolation',
    author              = 'Namgyu Kang',
    author_email        = 'kangnamgyu27@gmail.com',
    url                 = 'https://github.com/NamGyuKang/CosineSampler',
    download_url        = 'https://github.com/NamGyuKang/CosineSampler.git',
    ext_modules=[cpp_extension.CUDAExtension('CosineSampler.cosine_sampler_2d', ['/hdd/kng/CosineSampler/CosineSampler_2d/cosine_sampler_2d_kernel.cu', '/hdd/kng/CosineSampler/CosineSampler_2d/cosine_sampler_2d.cpp']),
                    cpp_extension.CUDAExtension('CosineSampler.cosine_sampler_3d', ['/hdd/kng/CosineSampler/CosineSampler_3d/cosine_sampler_3d_kernel.cu', '/hdd/kng/CosineSampler/CosineSampler_3d/cosine_sampler_3d.cpp'])],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    keywords            = ['triple backward interpolation'],
    python_requires     = '>=3',
    classifiers         = [
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Utilities",
        "Programming Language :: C++",
		"Programming Language :: CUDA",
		"Programming Language :: Python :: 3 :: Only",    
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
)
