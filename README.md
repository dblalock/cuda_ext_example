Minimal example of writing a custom CUDA function and calling it on PyTorch tensors. Includes a super simple version of addition and a slightly more complex one that uses [vectorized memory accesses](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/) (although these don't seem to help for this particular op and dtype on an A100).

To clean build and run everything, do:
```bash
pip uninstall -y cuda_ext_example; pip install -e . --no-build-isolation && pytest -s --tb=short tests/ && python scripts/benchmark.py
```
The --no-build-isolation helped me get this to build when I had CUDA 11.8 and torch 2.0.1 installed in the docker image `mosaicml/llm-foundry:2.0.1_cu118-latest`; in general, though, omitting this arg is better since it makes your build more reproducible.

Building any C++ file that includes `<torch/extension.h>` takes at least a full minute, and having pip resolve all the dependencies makes `pip install -e .` take even longer. To just build the C++ code, you can do `make cpp`.

For an even faster C++ debug cycle, cd into `csrc` and build + run `main.cu`, which doesn't rely on `<torch/extension.h>`. This code doesn't get wrapped in Python; it's just there to help you interactively debug your code. On an A100, you can do this with:
```bash
nvcc -O3 main.cu -gencode arch=compute_80,code=sm_80 -o main.o && ./main.o
```

To build a wheel and push it to PyPI...you're gonna have an adventure. The problem is that:

- PyPI only allows [manylinux](https://github.com/pypa/manylinux) linux wheels
- manylinux wheels try to bundle up all the shared libraries into the wheel, except common libraries with well-known ABIs. Think GLIBC, GLIBCXX, and CXXABI.
- We probably don't want to bundle all the CUDA libraries into our wheel, if for no other reason than to avoid hitting PyPI size limits.
- Also, we can't use the recommended manylinux build image because we need all the CUDA and torch stuff to get our CUDA extensions to build. We could FROM this image and install the necessary stuff ourselves, but ain't nobody got time for that.

So here's what we're gonna do.

First, we'll build our wheel.
```bash
python setup.py bdist_wheel
```

Now we'll install some libraries we need to turn this wheel into a manylinux wheel.
```bash
pip install auditwheel patchelf
```
You can sanity check everything so far with:
```bash
$ auditwheel show dist/cuda_ext_example-0.0.1-cp310-cp310-linux_x86_64.whl
```
which should show something like
```
cuda_ext_example-0.0.1-cp310-cp310-linux_x86_64.whl is consistent with
the following platform tag: "linux_x86_64".

The wheel references external versioned symbols in these
system-provided shared libraries: libgcc_s.so.1 with versions
{'GCC_3.0'}, libcudart.so.11.0 with versions {'libcudart.so.11.0'},
libc.so.6 with versions {'GLIBC_2.2.5', 'GLIBC_2.3.4', 'GLIBC_2.14',
'GLIBC_2.4', 'GLIBC_2.3', 'GLIBC_2.17', 'GLIBC_2.3.2', 'GLIBC_2.3.3'},
libstdc++.so.6 with versions {'CXXABI_1.3.5', 'CXXABI_1.3.2',
'GLIBCXX_3.4.18', 'GLIBCXX_3.4', 'CXXABI_1.3', 'CXXABI_1.3.3',
'GLIBCXX_3.4.9'}, libdl.so.2 with versions {'GLIBC_2.2.5'}, librt.so.1
with versions {'GLIBC_2.2.5'}, libpthread.so.0 with versions
{'GLIBC_2.2.5'}

This constrains the platform tag to "manylinux_2_17_x86_64". In order
to achieve a more compatible tag, you would need to recompile a new
wheel from source on a system with earlier versions of these
libraries, such as a recent manylinux image.
```
So we *could* target a recent-ish manylinux arch, but we still have to actually convert our wheel to make this happen. `auditwheel repair` can help us with this, but it will complain because it can't find various torch libraries. We *could* help it find them by adding the right path to our `PATH`:
```bash
# don't run this command because it will make your wheel too big
for d in $(python -c 'from torch.utils.cpp_extension import library_paths; [print(p) for p in library_paths()]'); do export PATH="${PATH}:$d"; done
```
But to avoid our wheel ending up huge, we're just gonna ignore them and rely on the user having torch libraries in their environment:
```bash
auditwheel repair dist/cuda_ext_example-0.0.1-cp310-cp310-linux_x86_64.whl --plat manylinux2014_x86_64 --exclude libtorch_cpu.so --exclude libtorch_python.so --exclude libc10.so
```

Now we just need to upload the resulting wheel. You'll need to already have a PyPI username and password for this part. You'll also need to change the project name in setup.py and maybe pyproject.toml before doing this because you won't have permission to write to my PyPI project.
```bash
pip install twine
twine check wheelhouse/*  # optional; just helps catch errors
twine upload wheelhouse/*
```

You can check that everything worked by running the following code in a colab notebook (or other system with an NVIDIA GPU):
```python
import torch  # this line just makes us fail fast if torch isn't installed

!pip install cuda_ext_example
import cuda_ext_example
a = torch.arange(5, device='cuda')
b = torch.arange(5, device='cuda') + 10
print(cuda_ext_example.my_add(a, b))
print(cuda_ext_example.my_fast_add(a, b))
```
