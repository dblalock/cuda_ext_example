#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// using index_t = uint32_t;

// TODO inline func instead of macro
#define AS_DENSE_ND(TENSOR, NDIMS) (TENSOR).packed_accessor32<scalar_t, NDIMS, torch::RestrictPtrTraits>()
#define AS_DENSE_1D(TENSOR) AS_DENSE_ND(TENSOR, 1)
#define AS_DENSE_2D(TENSOR) AS_DENSE_ND(TENSOR, 2)
#define AS_DENSE_3D(TENSOR) AS_DENSE_ND(TENSOR, 3)

template<typename T> using Dense1d = torch::PackedTensorAccessor32<T, 1, torch::RestrictPtrTraits>;

namespace {

template <typename scalar_t>
__global__ void _add_kernel(const Dense1d<scalar_t> a,
                            const Dense1d<scalar_t> b,
                            Dense1d<scalar_t> c,
                            size_t N)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // grid stride loop to allow grid size smaller than numel; see:
    // https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride) {
        c[i] = a[i] + b[i];
    }
}

} // anon namespace

size_t div_round_up(size_t x, size_t y) {
    return (x + y - 1) / y;
}

void add_wrapper(const at::Tensor in_a,
                 const at::Tensor in_b,
                 at::Tensor out_c,
                 int block_size = 64)
{
    size_t N = in_a.numel();
    dim3 grid_shape = div_round_up(N, block_size);
    // for available dispatch macro options, see here:
    // https:// github.com/pytorch/pytorch/blob/a2988c9e6ac281c2bf88eefde7fdd8ead44a8b36/aten/src/ATen/Dispatch.h
    // you'd think AT_DISPATCH_ALL_TYPES would include fp16 and bf16, but it
    // doesn't. Also, this macro still doesn't include bool, although
    // that's probably for the best.
    // AT_DISPATCH_FLOATING_TYPES(
    AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        in_a.scalar_type(), "add_cuda", ([&] {
            _add_kernel<scalar_t><<<grid_shape, block_size>>>(
                AS_DENSE_1D(in_a), AS_DENSE_1D(in_b), AS_DENSE_1D(out_c), N);
    }));
}
