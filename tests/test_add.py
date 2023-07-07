
from typing import Tuple

import pytest
import torch

import cuda_ext_example as ours

# rm bfloat16 if you have an older GPU that doesn't support it
@pytest.mark.parametrize('dtype', [torch.float32, torch.int8, torch.bfloat16, torch.float16])
@pytest.mark.parametrize('shape', [(5, 5), (1, 2, 3, 25)])
def test_add(shape: Tuple[int], dtype: torch.dtype):
    shape = (5, 5)
    a = torch.randint(10, size=shape, dtype=dtype, device='cuda')
    b = torch.randint(10, size=shape, dtype=dtype, device='cuda')
    torch.testing.assert_close(a + b, ours.my_add(a, b))

@pytest.mark.parametrize('dtype', [torch.float32])
@pytest.mark.parametrize('shape', [(5, 5), (1, 2, 3, 25), [int(1e6), 37]])
# @pytest.mark.parametrize('shape', [(2, 96)])
# @pytest.mark.parametrize('shape', [(33,)])
def test_fast_add(shape: Tuple[int], dtype: torch.dtype):
    # shape = (5, 5)
    a = torch.randint(10, size=shape, dtype=dtype, device='cuda')
    b = torch.randint(10, size=shape, dtype=dtype, device='cuda')
    # a = torch.arange(shape, dtype=dtype, device='cuda')
    # b = torch.arange(shape, dtype=dtype, device='cuda')
    # c = a + b
    # c_hat = ours.my_fast_add(a, b)
    # print()
    # print("c:    ", c)
    # print("c_hat:", c_hat)
    # torch.testing.assert_close(c, c_hat)
    torch.testing.assert_close(a + b, ours.my_fast_add(a, b))

