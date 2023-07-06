// This file is just here to help with debugging. You're gonna need it.
// run with:
//  nvcc -O3 main.cu -o main.o && ./main.o
// run on an A100 with fastmath optimizations (irrelevant for add) with:
//  nvcc -O3 --use_fast_math -gencode arch=compute_80,code=compute_80 main.cu -o main.o && ./main.o
// to inspect register usage on an A100:
//  nvcc --resource-usage --gpu-architecture=sm_80 --use_fast_math main.cu

#include <stdio.h>
#include <iostream>
#include <stdint.h>
#include <vector>

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

// note that this requires you to do a grid-stride loop, since it doesn't
// guarantee a thread count proportional to the input size N
inline int grid_size_for_block_size(size_t block_size, size_t N, size_t numel_per_thread=1) {
    auto numel = N / numel_per_thread;
    auto sm_count = num_cuda_cores();
    auto max_threads = sm_count * 2048;  // highest occupancy possible
    auto max_grid_size = div_round_up(max_threads, block_size);
    auto max_blocks_possible = div_round_up(numel, block_size);
    return min(max_blocks_possible, max_grid_size);
}

__global__ void _add_fast_f32(const float* __restrict__ a_tensor,
                              const float* __restrict__ b_tensor,
                              float* __restrict__ c_tensor,
                              uint32_t N) {
    using load_as_dtype = float4;
    static constexpr uint32_t elem_sz = sizeof(float);
    static constexpr uint32_t bytes_per_thread = sizeof(load_as_dtype);
    static constexpr uint32_t elems_per_read = bytes_per_thread / elem_sz;
    static_assert(elems_per_read == 1 || elems_per_read == 2 ||
                  elems_per_read == 4);

    auto a_ptr = reinterpret_cast<const load_as_dtype*>(&a_tensor[0]);
    auto b_ptr = reinterpret_cast<const load_as_dtype*>(&b_tensor[0]);
    auto c_ptr = reinterpret_cast<load_as_dtype*>(&c_tensor[0]);

    auto total_vec_reads = N / elems_per_read;
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

template <typename scalar_t>
__global__ void _add_simple(const scalar_t* __restrict__ a,
                            const scalar_t* __restrict__ b,
                            scalar_t* __restrict__ out,
                            uint32_t N)
{
    auto index = blockIdx.x * blockDim.x + threadIdx.x;

    // grid stride loop to allow grid size smaller than numel; see:
    // https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    auto stride = blockDim.x * gridDim.x;
    for (auto i = index; i < N; i += stride) {
        out[i] = a[i] + b[i];
    }
}

int main() {
    // constexpr size_t N = ((long)1) << 24; // works
    constexpr size_t N = ((long)1) << 28;

    // figure out block + grid sizes
    int block_size = 256;
    int numel_per_thread = 4;  // 4 for fast version
    auto grid_size = grid_size_for_block_size(block_size, N, numel_per_thread);
    std::cout << "grid_size:" << grid_size << " block_size: " << block_size << std::endl;

    using dtype = float;
    std::vector<dtype> a(N);
    std::vector<dtype> b(N);
    std::vector<dtype> out(N);
    for (size_t i = 0; i < N; i++) {
        int val = i % 10000;  // not just i to avoid fp error past 2^24
        a[i] = val;
        b[i] = 1 - val;
        out[i] = -777;
    }

    // copy to device
    dtype *a_cu, *b_cu, *out_cu;
    size_t nbytes = sizeof(dtype) * N;
    cudaMalloc((void**)&a_cu, nbytes);
    cudaMalloc((void**)&b_cu, nbytes);
    cudaMalloc((void**)&out_cu, nbytes);
    cudaMemcpy(a_cu, a.data(), nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(b_cu, b.data(), nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(out_cu, out.data(), nbytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    _add_fast_f32<<<grid_size, block_size>>>(a_cu, b_cu, out_cu, N);
    // _add_simple<float><<<div_round_up(N, block_size), block_size>>>(a_cu, b_cu, out_cu, N);
    cudaEventRecord(stop);
    cudaMemcpy(out.data(), out_cu, nbytes, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("milliseconds: %f", milliseconds);

    uint32_t num_errors = 0;
    int64_t first_err_index = -1;
    for (size_t i = 0; i < N; i++) {
        // printf("%.1f ", out[i]);
        if (out[i] != 1.) {
            if (num_errors == 0) {
                first_err_index = i;
            }
            num_errors++;
        //     printf("%d: %.0f\n", i, out[i]);
        }
    }
    printf("\nTotal errors: %u / %ld; first at %ld\n", num_errors, (long)N, first_err_index);
    cudaFree(a_cu);
    cudaFree(b_cu);
    cudaFree(out_cu);

    return 0;
}
