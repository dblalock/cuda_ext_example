
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "cpp/src/add.cuh"
#include "cpp/src/utils.hpp"

template <typename Dtype>
void AddGPU(at::Tensor in_a, at::Tensor in_b, at::Tensor out_c) {
    int N = in_a.numel();
    if (N != in_b.numel())
        throw std::invalid_argument(
            Formatter() << "Size mismatch A.numel(): " << in_a.numel()
                        << ", B.numel(): " << in_b.numel());

    out_c.resize_({N});

    AddGPUKernel<Dtype>(in_a.data_ptr<Dtype>(), in_b.data_ptr<Dtype>(),
                        out_c.data_ptr<Dtype>(), N,
                        at::cuda::getCurrentCUDAStream());
}

template void AddGPU<float>(at::Tensor in_a, at::Tensor in_b, at::Tensor out_c);
