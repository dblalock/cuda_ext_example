#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

using index_t = uint32_t;

// TODO inline func instead of macro
#define AS_DENSE_ND(TENSOR, NDIMS) (TENSOR).packed_accessor<scalar_t, NDIMS, torch::RestrictPtrTraits, index_t>()
#define AS_DENSE_1D(TENSOR) AS_DENSE_ND(TENSOR, 1)
#define AS_DENSE_2D(TENSOR) AS_DENSE_ND(TENSOR, 2)
#define AS_DENSE_3D(TENSOR) AS_DENSE_ND(TENSOR, 3)


namespace {

template<typename T> using Dense1d = torch::PackedTensorAccessor<T, 1, torch::RestrictPtrTraits, index_t>;

template <int bytes_per_elem> byte_count_traits;

template <> byte_count_traits <1> { using dtype = uint8_t; }
template <> byte_count_traits <2> { using dtype = uint16_t; }
template <> byte_count_traits <4> { using dtype = uint32_t; }
template <> byte_count_traits <8> { using dtype = uint64_t; }
template <> byte_count_traits <16> { using dtype = double2; } // cuda vec dtype

template <typename scalar_t, int bytes_per_thread>
__global__ void _add_fast_kernel(const Dense1d<scalar_t> a,
                                 const Dense1d<scalar_t> b,
                                 Dense1d<scalar_t> c,
                                 size_t N)
{
    static constexpr elem_sz = sizeof(a[0]);
    static constexpr elems_per_read = bytes_per_thread / elem_sz;
    static_assert(bytes_per_thread >= elem_sz);
    using load_as_dtype = typename byte<bytes_per_thread>::dtype;
    a_ptr = reinterpret_cast<load_as_dtype>(&a[0]);
    b_ptr = reinterpret_cast<load_as_dtype>(&b[0]);
    c_ptr = reinterpret_cast<load_as_dtype>(&c[0]);

    int total_vec_reads = N * elem_sz / bytes_per_thread;
    int num_non_stragglers = total_vec_reads * elems_per_read;
    int num_stragglers = N - num_non_stragglers;
    int index_in_grid = blockIdx.x * blockDim.x + threadIdx.x;

    // grid stride loop to allow grid size smaller than numel; see:
    // https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    int grid_numel = blockDim.x * gridDim.x;
    for (int i = index_in_grid; i < total_vec_reads; i += grid_numel) {
        // load data as if load_as_dtype is the true dtype. Then operate
        // on it as the original dtype. We use the volatile keyword to
        // hopefully make the compiler issue loads of the wider dtype
        // instead of just flattening the pointer arithmetic and doing
        // loads as the true dtype.
        volatile load_as_dtype a_vec = a_ptr[i];
        volatile load_as_dtype b_vec = b_ptr[i];
        volatile load_as_dtype c_vec = c_ptr[i];
        a_vec_ptr = reinterpret_cast<scalar_t*>(&a_vec);
        b_vec_ptr = reinterpret_cast<scalar_t*>(&b_vec);
        c_vec_ptr = reinterpret_cast<scalar_t*>(&c_vec);
        for (int ii = 0; ii < elems_per_read; ii++) {
            c_vec_ptr[ii] = a_vec_ptr[ii] + b_vec_ptr[ii];
        }
        c_ptr[i] = c_vec; // hopefully force vectorized store
    }
    // handle trailing elems, if any, using scalar loads in the true dtype
    if (index_in_grid < num_stragglers) {
        auto idx = num_non_stragglers + index_in_grid;
        c[idx] = a[idx] + b[idx];
    }
}

}  // anon namespace

template <typename scalar_t, int bytes_per_thread>
void _add_fast_static_bytes_per_kernel(
    const Dense1d<scalar_t> a,
    const Dense1d<scalar_t> b,
    Dense1d<scalar_t> c,
    size_t N,
    size_t bytes_per_thread,
    int grid_size,
    int block_size)
{
    bytes_per_thread = max(bytes_per_thread, sizeof(scalar_t));
    // assert(bytes_per_thread >= sizeof(scalar_t));
    // auto bytes_per_block = bytes_per_thread * block_shape;
    // auto min_bytes = grid_size * bytes_per_block;
    // auto min_elems = min_bytes / sizeof(scalar_t);
    // assert(block_size % 32 * sizeof(scalar_t) / bytes_per_thread = )

    dim3 grid_shape = grid_size;
    dim3 block_shape = block_size;
    switch (bytes_per_thread) {
    case 1:
        _add_fast_kernel<scalar_t, 1><<<grid_shape, block_shape>>>(a, b, c, N);
        break;
    case 2:
        _add_fast_kernel<scalar_t, 2><<<grid_shape, block_shape>>>(a, b, c, N);
        break;
    case 4:
        _add_fast_kernel<scalar_t, 4><<<grid_shape, block_shape>>>(a, b, c, N);
        break;
    case 8:
        _add_fast_kernel<scalar_t, 1><<<grid_shape, block_shape>>>(a, b, c, N);
        break;
    case 16:
        _add_fast_kernel<scalar_t, 16><<<grid_shape, block_shape>>>(a, b, c, N);
        break;
    }
}

inline size_t div_round_up(size_t x, size_t y) {
    return (x + y - 1) / y;
}

inline int num_cuda_cores() {
    int deviceID;
    cudaDeviceProp props;
    cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&props, deviceID);
    return props.multiProcessorCount;
}

void add_wrapper(const at::Tensor in_a,
                 const at::Tensor in_b,
                 at::Tensor out_c,
                 int block_size,
                 int bytes_per_thread)
{
    auto N = in_a.numel();
    auto sm_count = num_cuda_cores();
    auto max_threads = sm_count * 2048;  // highest occupancy possible

    auto num_blocks = div_round_up(N, block_shape);
    num_blocks = min(num_blocks, max_threads / block_shape);
    dim3 grid_shape = num_blocks;

    // for available dispatch macro options, see here:
    // https:// github.com/pytorch/pytorch/blob/a2988c9e6ac281c2bf88eefde7fdd8ead44a8b36/aten/src/ATen/Dispatch.h
    // you'd think AT_DISPATCH_ALL_TYPES would include fp16 and bf16, but it
    // doesn't. Also, this macro still doesn't include bool, although
    // that's probably for the best.
    AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
    in_a.type(), "add_fast_cuda", ([&] {
        _add_fast_static_bytes_per_kernel<scalar_t>(
            AS_DENSE_1D(in_a), AS_DENSE_1D(in_b), AS_DENSE_1D(out_c),
            N, bytes_per_thread, grid_shape, block_size);
    }));
}
