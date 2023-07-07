
#include <iostream> // for debugging
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// // TODO inline func instead of macro
// #define AS_DENSE_ND(TENSOR, NDIMS) ((TENSOR).packed_accessor32<scalar_t, NDIMS, torch::RestrictPtrTraits>())
// #define AS_DENSE_1D(TENSOR) AS_DENSE_ND(TENSOR, 1)
// #define AS_DENSE_2D(TENSOR) AS_DENSE_ND(TENSOR, 2)
// #define AS_DENSE_3D(TENSOR) AS_DENSE_ND(TENSOR, 3)


namespace {

// template<typename T> using Dense1d = torch::PackedTensorAccessor32<T, 1, torch::RestrictPtrTraits>;

// template <int bytes_per_elem> struct byte_count_traits {};

// template<> struct byte_count_traits<1> { using dtype = uint8_t; };
// template<> struct byte_count_traits<2> { using dtype = uint16_t; };
// template<> struct byte_count_traits<4> { using dtype = uint32_t; };
// template<> struct byte_count_traits<8> { using dtype = uint64_t; };
// template<> struct byte_count_traits<16> { using dtype = double2; }; // cuda vec dtype

// template<typename scalar_t, int bytes_per_thread>
// __global__ void _add_fast_kernel(const Dense1d<scalar_t> a,
//                                  const Dense1d<scalar_t> b,
//                                  Dense1d<scalar_t> c,
//                                  size_t N) {
//     using load_as_dtype = typename byte_count_traits<bytes_per_thread>::dtype;
//     static constexpr auto elem_sz = sizeof(a[0]);
//     static_assert(bytes_per_thread >= elem_sz);
//     static constexpr auto elems_per_read = bytes_per_thread / elem_sz;

//     auto a_ptr = reinterpret_cast<const load_as_dtype*>(&a[0]);
//     auto b_ptr = reinterpret_cast<const load_as_dtype*>(&b[0]);
//     auto c_ptr = reinterpret_cast<load_as_dtype*>(&c[0]);

//     int total_vec_reads = N * elem_sz / bytes_per_thread;
//     int num_non_stragglers = total_vec_reads * elems_per_read;
//     int num_stragglers = N - num_non_stragglers;
//     int index_in_grid = blockIdx.x * blockDim.x + threadIdx.x;

//     // grid stride loop to allow grid size smaller than numel; see:
//     // https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
//     int grid_numel = blockDim.x * gridDim.x;
//     for (int i = index_in_grid; i < total_vec_reads; i += grid_numel) {
//         // load data as if load_as_dtype is the true dtype. Then operate
//         // on it as the original dtype. We use the volatile keyword to
//         // hopefully make the compiler issue loads of the wider dtype
//         // instead of just flattening the pointer arithmetic and doing
//         // loads as the true dtype.
//         // const volatile load_as_dtype a_vec = a_ptr[i];
//         // const volatile load_as_dtype b_vec = b_ptr[i];
//         // volatile load_as_dtype c_vec = c_ptr[i];
//         const load_as_dtype a_vec = __ldg(&a_ptr[i]);
//         const load_as_dtype b_vec = __ldg(&b_ptr[i]);
//         load_as_dtype c_vec = __ldg(&c_ptr[i]);

//         // version that runs correctly, but super slowly
//         // auto a_vec_ptr = reinterpret_cast<const scalar_t*>(&a_vec);
//         // auto b_vec_ptr = reinterpret_cast<const scalar_t*>(&b_vec);
//         // auto c_vec_ptr = reinterpret_cast<scalar_t*>(&c_vec);
//         // for (int ii = 0; ii < elems_per_read; ii++) {
//         //     c_vec_ptr[ii] = a_vec_ptr[ii] + b_vec_ptr[ii];
//         // }
//         // // c_ptr[i] = *reinterpret_cast<load_as_dtype*>(&c_vec);  // cast to rm volatile
//         // c_ptr[i] = c_vec;

//         // version that hopefully runs fast?
//         scalar_t a_ar[elems_per_read];
//         scalar_t b_ar[elems_per_read];
//         scalar_t c_ar[elems_per_read];
//         std::memcpy(&a_ar, &a_vec, sizeof(load_as_dtype));
//         std::memcpy(&b_ar, &b_vec, sizeof(load_as_dtype));
//         std::memcpy(&c_ar, &c_vec, sizeof(load_as_dtype));

//         // auto a_vec_ar = reinterpret_cast<scalar_t[elems_per_read]>(&a_vec);
//         // auto b_vec_ar = reinterpret_cast<scalar_t[elems_per_read]>(&b_vec);
//         // auto c_vec_ar = reinterpret_cast<scalar_t[elems_per_read]>(&c_vec);
//         for (int ii = 0; ii < elems_per_read; ii++) {
//             c_ar[ii] = a_ar[ii] + b_ar[ii];
//         }
//         // c_ptr[i] = *reinterpret_cast<load_as_dtype*>(&c_vec);  // cast to rm
//         // volatile
//         std::memcpy(&c_ptr[i], &c_ar, sizeof(load_as_dtype));
//         // c_ptr[i] = c_vec_ar;
//     }
//     // handle trailing elems, if any, using scalar loads in the true dtype
//     if (index_in_grid < num_stragglers) {
//         auto idx = num_non_stragglers + index_in_grid;
//         c[idx] = a[idx] + b[idx];
//     }
// }

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

    // auto total_vec_reads = N / elems_per_read;
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

// __global__ void _add_fast_f32(const Dense1d<float> a_tensor,
//                               const Dense1d<float> b_tensor,
//                               Dense1d<float> c_tensor, size_t N) {
//     using load_as_dtype = float4;
//     static constexpr size_t elem_sz = sizeof(float);
//     static constexpr size_t bytes_per_thread = sizeof(load_as_dtype);
//     static constexpr size_t elems_per_read = bytes_per_thread / elem_sz;
//     static_assert(elems_per_read == 1 || elems_per_read == 2 ||
//                   elems_per_read == 4);

//     auto a_ptr = reinterpret_cast<const load_as_dtype*>(&a_tensor[0]);
//     auto b_ptr = reinterpret_cast<const load_as_dtype*>(&b_tensor[0]);
//     auto c_ptr = reinterpret_cast<load_as_dtype*>(&c_tensor[0]);

//     auto total_vec_reads = N / elems_per_read;
//     auto num_non_stragglers = total_vec_reads * elems_per_read;
//     auto num_stragglers = N - num_non_stragglers;
//     auto index_in_grid = blockIdx.x * blockDim.x + threadIdx.x;

//     // grid stride loop to allow grid size smaller than numel; see:
//     // https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
//     auto grid_numel = blockDim.x * gridDim.x;
//     for (int i = index_in_grid; i < total_vec_reads; i += grid_numel) {
//         const load_as_dtype a = __ldg(&a_ptr[i]);
//         const load_as_dtype b = __ldg(&b_ptr[i]);
//         load_as_dtype c = c_ptr[i];

//         c.x = a.x + b.x;
//         c.y = a.y + b.y;
//         c.z = a.z + b.z;
//         c.w = a.w + b.w;
//         c_ptr[i] = c;
//         // switch (elems_per_read) {
//         //     case 1:
//         //         c.x = a.x + b.x;
//         //     case 2:
//         //         c.y = a.y + b.y;
//         //     case 4:
//         //         // makes it not compile with float2 loads since
//         //         // `error: class "float2" has no member "z"`
//         //         c.z = a.z + b.z;
//         //         c.w = a.x + b.w;
//         // }
//     }
//     if (index_in_grid < num_stragglers) {
//         auto idx = num_non_stragglers + index_in_grid;
//         c_tensor[idx] = a_tensor[idx] + b_tensor[idx];
//     }
// }

}  // anon namespace

// template<typename scalar_t>
// void _add_fast_static_bytes_per_kernel(const Dense1d<scalar_t> a,
//                                        const Dense1d<scalar_t> b,
//                                        Dense1d<scalar_t> c,
//                                        size_t N,
//                                        size_t bytes_per_thread,
//                                        size_t grid_size,
//                                        size_t block_size) {
//     bytes_per_thread = max(bytes_per_thread, sizeof(scalar_t));
//     // assert(bytes_per_thread >= sizeof(scalar_t));
//     // auto bytes_per_block = bytes_per_thread * block_shape;
//     // auto min_bytes = grid_size * bytes_per_block;
//     // auto min_elems = min_bytes / sizeof(scalar_t);
//     // assert(block_size % 32 * sizeof(scalar_t) / bytes_per_thread = )

//     dim3 grid_shape = grid_size;
//     dim3 block_shape = block_size;
//     switch (bytes_per_thread) {
//     // case 1:
//     //     _add_fast_kernel<scalar_t, 1><<<grid_shape, block_shape>>>(a, b, c, N);
//     //     break;
//     // case 2:
//     //     _add_fast_kernel<scalar_t, 2><<<grid_shape, block_shape>>>(a, b, c, N);
//     //     break;
//     case 4:
//         _add_fast_kernel<scalar_t, 4><<<grid_shape, block_shape>>>(a, b, c, N);
//         break;
//     case 8:
//         _add_fast_kernel<scalar_t, 8><<<grid_shape, block_shape>>>(a, b, c, N);
//         break;
//     case 16:
//         _add_fast_kernel<scalar_t, 16><<<grid_shape, block_shape>>>(a, b, c, N);
//         break;
//     }
// }

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
    // auto max_threads = sm_count * 2048;  // highest occupancy possible
    auto max_simultaneous_blocks = sm_count * kMaxBlocksPerSM;
    auto max_blocks_of_work = div_round_up(numel, block_size);
    return min(max_simultaneous_blocks, max_blocks_of_work);
    // auto max_grid_size = div_round_up(max_threads, block_size);
    // return min(max_blocks_of_work, max_grid_size);
}

void add_fast_wrapper(const at::Tensor in_a, const at::Tensor in_b,
                      at::Tensor out_c, size_t block_size, int grid_size)
{
    constexpr int numel_per_thread = 4;  // 4 for fast version
    auto N = in_a.numel();
    int num_blocks = grid_size;
    if (grid_size <= 0) {
        auto num_blocks = grid_size_for_block_size(block_size, N, numel_per_thread);
        // num_blocks = div_round_up(N, block_size * numel_per_thread); // TODO is this faster?
    }

    // std::cout << "num blocks: " << num_blocks;
    // std::cout << " block size: " << block_size << std::endl;

    // auto sm_count = num_cuda_cores();
    // size_t max_threads = sm_count * 2048;  // highest occupancy possible

    // // TODO use bytes_per_thread

    // // block_size = min(32 * (N / 32), block_size);
    // size_t num_blocks = div_round_up(N, block_size);
    // num_blocks = min(num_blocks, max_threads / block_size);
    // // dim3 grid_shape = num_blocks;

    // for available dispatch macro options, see here:
    // https:// github.com/pytorch/pytorch/blob/a2988c9e6ac281c2bf88eefde7fdd8ead44a8b36/aten/src/ATen/Dispatch.h
    // you'd think AT_DISPATCH_ALL_TYPES would include fp16 and bf16, but it
    // doesn't. Also, this macro still doesn't include bool, although
    // that's probably for the best.
    // AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
    // AT_DISPATCH_FLOATING_TYPES(
    //     in_a.type(), "add_fast_cuda", ([&] {
    //         _add_fast_static_bytes_per_kernel<scalar_t>(
    //             AS_DENSE_1D(in_a), AS_DENSE_1D(in_b), AS_DENSE_1D(out_c),
    //             N, bytes_per_thread, num_blocks, block_size);
    // }));

    // this is the more manual version that lets you choose specific dtypes
    // note that the function call has to be wrapped in a lambda for the macro
    // to work; these macros aren't needed for the op to run; they just add
    // some logging + error handling, defines scalar_t, and maps torch
    // dtypes to cpp types.
        // in_a.type(), "add_fast_cuda",
    // AT_DISPATCH_CASE(at::ScalarType::Float, [&] {
    // AT_DISPATCH_SWITCH(in_a.type(), "add_fast_cuda",
    //     AT_DISPATCH_CASE_FLOATING_TYPES([&] {
    //         _add_fast_static_bytes_per_kernel<scalar_t>(
    //             AS_DENSE_1D(in_a), AS_DENSE_1D(in_b),
    //             AS_DENSE_1D(out_c), N, bytes_per_thread, num_blocks,
    //             block_size);
    //         };
    //     )
    // );
    const auto& the_type = in_a.type();
    switch (in_a.scalar_type()) {
    case at::ScalarType::Float:
        // using scalar_t = float; // for AS_DENSE macros
        // at::ScalarType _st = ::detail::scalar_type(the_type);
        auto total_vec_reads = N / numel_per_thread;
        _add_fast_f32<<<num_blocks, block_size>>>(
            in_a.data_ptr<float>(),
            in_b.data_ptr<float>(),
            out_c.data_ptr<float>(),
            N, total_vec_reads);
        // _add_fast_static_bytes_per_kernel<scalar_t>(
        //     AS_DENSE_1D(in_a), AS_DENSE_1D(in_b), AS_DENSE_1D(out_c), N,
        //     bytes_per_thread, num_blocks, block_size);
        break;
    }

// #define AT_DISPATCH_SWITCH(TYPE, NAME, ...)                                  \
//     [&] {                                                                    \
//         const auto& the_type = TYPE;                                         \
//         constexpr const char* at_dispatch_name = NAME;                       \
//         /* don't use TYPE again in case it is an expensive or side-effect op \
//          */                                                                  \
//         at::ScalarType _st = ::detail::scalar_type(the_type);                \
//         RECORD_KERNEL_FUNCTION_DTYPE(at_dispatch_name, _st);                 \
//         switch (_st) {                                                       \
//             __VA_ARGS__                                                      \
//             default:                                                         \
//                 AT_ERROR('"', at_dispatch_name, "\" not implemented for '",  \
//                          toString(_st), "'");                                \
//         }                                                                    \
//     }()
}
