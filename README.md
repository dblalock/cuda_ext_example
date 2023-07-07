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

