
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

namespace {

__global__ void _add_fast_f32(const float* __restrict__ a_tensor,
                              const float* __restrict__ b_tensor,
                              float* __restrict__ c_tensor,
                              uint32_t N,
                              uint32_t total_vec_reads)
{
    using load_as_dtype = float4;
    static constexpr uint32_t elem_sz = sizeof(float);
    static constexpr uint32_t bytes_per_thread = sizeof(load_as_dtype);
    static constexpr uint32_t elems_per_read = bytes_per_thread / elem_sz;
    static_assert(elems_per_read == 1 || elems_per_read == 2 ||
                  elems_per_read == 4);

    auto a_ptr = reinterpret_cast<const load_as_dtype*>(&a_tensor[0]);
    auto b_ptr = reinterpret_cast<const load_as_dtype*>(&b_tensor[0]);
    auto c_ptr = reinterpret_cast<load_as_dtype*>(&c_tensor[0]);

    auto num_non_stragglers = total_vec_reads * elems_per_read;
    auto num_stragglers = N - num_non_stragglers;
    auto index_in_grid = blockIdx.x * blockDim.x + threadIdx.x;

    // grid stride loop to allow grid size smaller than numel; see:
    // https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    auto grid_numel = blockDim.x * gridDim.x;
    for (uint32_t i = index_in_grid; i < total_vec_reads; i += grid_numel) {
        const load_as_dtype a = __ldg(&a_ptr[i]);
        const load_as_dtype b = __ldg(&b_ptr[i]);
        load_as_dtype c = c_ptr[i];

        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        c_ptr[i] = c;
    }
    if (index_in_grid < num_stragglers) {
        auto idx = num_non_stragglers + index_in_grid;
        c_tensor[idx] = a_tensor[idx] + b_tensor[idx];
    }
}

inline size_t div_round_up(size_t x, size_t y) {
    return (x + y - 1) / y;
}

inline int num_sms() {
    int deviceID;
    cudaDeviceProp props;
    cudaGetDevice(&deviceID);
    cudaGetDeviceProperties(&props, deviceID);
    return props.multiProcessorCount;
}

// note that this requires you to do a grid-stride loop, since it doesn't
// guarantee a thread count proportional to the input size N
inline int grid_size_for_block_size(size_t block_size, size_t N, size_t numel_per_thread=1) {
    constexpr size_t kMaxBlocksPerSM = 32;
    auto numel = N / numel_per_thread;
    auto sm_count = num_sms();
    auto max_simultaneous_blocks = sm_count * kMaxBlocksPerSM;
    auto max_blocks_of_work = div_round_up(numel, block_size);
    return min(max_simultaneous_blocks, max_blocks_of_work);
}

}  // anon namespace


void add_fast_wrapper(const at::Tensor in_a, const at::Tensor in_b,
                      at::Tensor out_c, size_t block_size, int grid_size)
{
    constexpr int numel_per_thread = 4;  // since using float4
    auto N = in_a.numel();
    int num_blocks = grid_size;
    if (grid_size <= 0) {
        num_blocks = grid_size_for_block_size(block_size, N, numel_per_thread);
    }

    const auto& the_type = in_a.type();
    switch (in_a.scalar_type()) {
    case at::ScalarType::Float:
        auto total_vec_reads = N / numel_per_thread;
        _add_fast_f32<<<num_blocks, block_size>>>(
            in_a.data_ptr<float>(),
            in_b.data_ptr<float>(),
            out_c.data_ptr<float>(),
            N, total_vec_reads);
        break;
    }
}
