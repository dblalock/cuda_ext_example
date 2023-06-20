
import pytest
import torch

import cuda_ext_example as ours

# @pytest.mark.parametrize('dtype', [torch.float32, torch.int8])
@pytest.mark.parametrize('dtype', [torch.float32])
def test_add(dtype: torch.dtype):
    shape = (5, 5)
    a = torch.randint(10, size=shape, dtype=dtype, device='cuda')
    b = torch.randint(10, size=shape, dtype=dtype, device='cuda')
    torch.testing.assert_close(a + b, ours.my_add(a, b))
