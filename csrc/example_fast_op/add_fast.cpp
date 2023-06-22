
#include <torch/extension.h>

// forward declare public function from cuda file
void add_fast_wrapper(const at::Tensor in_a,
                 const at::Tensor in_b,
                 at::Tensor out_c,
                 int block_size,
                 int bytes_per_thread);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void add_fast(at::Tensor in_a, at::Tensor in_b, at::Tensor out_c,
         int block_size = 64, int bytes_per_thread = 4)
{
    CHECK_INPUT(in_a);
    CHECK_INPUT(in_b);
    CHECK_INPUT(out_c);
    // TORCH_CHECK(block_size % (bytes_per_thread * 32 / sizeof(in_a.type().dtype()) == 0, "Block size must be large enough to accomodate 32 loads of size " #bytes_per_thread )
    add_fast_wrapper(in_a, in_b, out_c, block_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("add_fast", &add, "Elementwise addition op");
}
