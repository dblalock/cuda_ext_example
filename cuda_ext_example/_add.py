from typing import Optional

import torch

import _my_cuda_kernels as kernels


def my_add(a: torch.Tensor,
           b: torch.Tensor,
           out: Optional[torch.Tensor] = None,
           block_size: int = 64) -> torch.Tensor:
    assert a.shape == b.shape
    if out is None:
        out = torch.empty_like(a)
    assert a.shape == out.shape

    kernels.add(a.ravel(), b.ravel(), out.ravel(), block_size)
    return out.reshape(a.shape)


def my_fast_add(a: torch.Tensor,
                b: torch.Tensor,
                out: Optional[torch.Tensor] = None,
                block_size: int = 64,
                bytes_per_thread: int = 4) -> torch.Tensor:
    assert a.shape == b.shape
    if out is None:
        out = torch.empty_like(a)
    assert a.shape == out.shape

    kernels.add_fast(a.ravel(), b.ravel(), out.ravel(), block_size, bytes_per_thread)
    return out.reshape(a.shape)
