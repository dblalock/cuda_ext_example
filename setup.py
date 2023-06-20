# Largely adapted from https://github.com/NVIDIA/apex/blob/master/setup.py

import os
import subprocess
import warnings
from packaging.version import parse, Version

from setuptools import setup, find_packages

PACKAGE_NAME = 'cuda_ext_example'

is_torch_installed = False
try:
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
    is_torch_installed = True
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("No module named 'torch'. Torch is required to install this repo.") from e


extra_deps = {
    'dev': [
        'pytest',
        'toml',
        'yapf',
        'isort',
        'yamllint',
    ]
}


def package_files(prefix: str, directory: str, extension: str):
    # from https://stackoverflow.com/a/36693250
    paths = []
    for (path, _, filenames) in os.walk(os.path.join(prefix, directory)):
        for filename in filenames:
            if filename.endswith(extension):
                paths.append(os.path.relpath(os.path.join(path, filename), prefix))
    return paths


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def append_nvcc_threads(nvcc_extra_args):
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version >= Version('11.2'):
        return nvcc_extra_args + ['--threads', '4']
    return nvcc_extra_args


def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
    raw_output, bare_metal_version = get_cuda_bare_metal_version(cuda_dir)
    torch_binary_version = parse(torch.version.cuda)

    print("\nCompiling cuda extensions with")
    print(raw_output + "from " + cuda_dir + "/bin\n")

    if (bare_metal_version != torch_binary_version):
        raise RuntimeError(
            "Cuda extensions are being compiled with a version of Cuda that does "
            "not match the version used to compile Pytorch binaries.  "
            "Pytorch binaries were compiled with Cuda {}.\n".format(torch.version.cuda)
            + "In some cases, a minor-version mismatch will not cause later errors:  "
            "https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  "
            "You can try commenting out this check (at your own risk)."
        )


# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

cmdclass = {}
ext_modules = []

# Only install CUDA extensions if available
if 'cu' in torch.__version__:
    check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)

    # Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
    # See https://github.com/pytorch/pytorch/pull/70650
    generator_flag = []
    torch_dir = torch.__path__[0]
    if os.path.exists(os.path.join(torch_dir, 'include', 'ATen', 'CUDAGeneratorImpl.h')):
        generator_flag = ['-DOLD_GENERATOR_PATH']

    # Check, if CUDA11 is installed for compute capability 8.0
    cc_flag = []
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_70,code=sm_70")
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_80,code=sm_80")
    if bare_metal_version >= Version("11.1"):
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_86,code=sm_86")
    if bare_metal_version >= Version("11.8"):
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_90,code=sm_90")

    ext_modules.append(
        CUDAExtension(
            name='_my_cuda_kernels',
            sources=[
                'csrc/example_op/add_cuda.cu',
                'csrc/example_op/add.cpp',
            ],
            extra_compile_args={
                'cxx': ['-O3'] + generator_flag,
                'nvcc':
                    append_nvcc_threads([
                        '-O3',
                        # uncomment these if you hit errors resulting from
                        # PyTorch and CUDA independently implementing slightly
                        # different [b]f16 support
                        # '-U__CUDA_NO_HALF_OPERATORS__',
                        # '-U__CUDA_NO_HALF_CONVERSIONS__',
                        # '-U__CUDA_NO_BFLOAT16_OPERATORS__',
                        # '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                        # '-U__CUDA_NO_BFLOAT162_OPERATORS__',
                        # '-U__CUDA_NO_BFLOAT162_CONVERSIONS__',
                        '--expt-relaxed-constexpr',
                        '--expt-extended-lambda',
                        '--use_fast_math',
                    ] + generator_flag + cc_flag),
            },
            include_dirs=[os.path.join(this_dir, 'csrc', 'example_op')],
        ))
    cmdclass = {'build_ext': BuildExtension}
else:
    warnings.warn('Warning: No CUDA devices; cuda code will not be compiled.')

setup(
    name='cuda_ext_example',
    version='0.0.1',
    author='dblalock',
    author_email='davis@mosaicml.com',
    description="simple example project that builds a PyTorch CUDA extension",
    url='https://github.com/dblalock/cuda_ext_example',
    packages=find_packages(exclude=['tests*']),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    extras_require=extra_deps,
    python_requires='>=3.7',
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
