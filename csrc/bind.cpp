
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

// ================================================================ add

// forward declare public function from cuda file
void add_wrapper(const at::Tensor in_a, const at::Tensor in_b, at::Tensor out_c,
                 int block_size);

void add(at::Tensor in_a, at::Tensor in_b, at::Tensor out_c,
         int block_size = 64) {
    CHECK_INPUT(in_a);
    CHECK_INPUT(in_b);
    CHECK_INPUT(out_c);
    add_wrapper(in_a, in_b, out_c, block_size);
}

// ================================================================ add_fast

// forward declare public function from cuda files
void add_fast_wrapper(const at::Tensor in_a, const at::Tensor in_b,
                      at::Tensor out_c, size_t block_size, int grid_size);

void add_fast(at::Tensor in_a, at::Tensor in_b, at::Tensor out_c,
              int block_size, int grid_size)
{
    CHECK_INPUT(in_a);
    CHECK_INPUT(in_b);
    CHECK_INPUT(out_c);
    add_fast_wrapper(in_a, in_b, out_c, block_size, grid_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add, "Elementwise addition op");
    m.def("add_fast", &add_fast, "Faster elementwise addition op");
}
