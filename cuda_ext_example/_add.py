from typing import Optional

import torch
import functools

import _my_cuda_kernels as kernels


def my_add(a: torch.Tensor,
           b: torch.Tensor,
           out: Optional[torch.Tensor] = None,
           block_size: int = 128) -> torch.Tensor:
    assert a.shape == b.shape
    if out is None:
        out = torch.empty_like(a)
    assert a.shape == out.shape

    kernels.add(a.ravel(), b.ravel(), out.ravel(), block_size)
    return out.reshape(a.shape)


# def _max_simultaneous_blocks() -> int:
def _sm_count() -> int:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.multi_processor_count


def _div_round_up(x: int, y: int) -> int:
    return (x + y - 1) // y


# cache the sm count and arithmtic since code is faster when we just
# call it with the right number of blocks. We normally use 32 blocks per
# SM since this is the minimal number that can run concurrently; this
# amortizes block creation + cleanup overhead
@functools.cache
def _blocks_to_use(numel: int, block_size: int, elems_per_thread) -> int:
    # these are just always true for A100s
    sm_count = _sm_count()
    max_simultaneous_threads = sm_count * 2048
    max_simultaneous_blocks = sm_count * 32
    max_blocks_schedulable = _div_round_up(max_simultaneous_threads, block_size)
    max_blocks_of_work = _div_round_up(numel, block_size * elems_per_thread)
    max_blocks = min(max_simultaneous_blocks, max_blocks_schedulable)
    max_blocks = min(max_blocks, max_blocks_of_work)
    return max_blocks * 2  # the x2 isn't needed; just like 1% faster


def my_fast_add(a: torch.Tensor,
                b: torch.Tensor,
                out: Optional[torch.Tensor] = None,
                block_size: int = 256,
                grid_size: int = -1) -> torch.Tensor:
    assert block_size >= 64
    assert a.shape == b.shape
    if out is None:
        out = torch.empty_like(a)
    assert a.shape == out.shape
    if grid_size < 1:
        grid_size = _blocks_to_use(a.numel(), block_size, 4)

    kernels.add_fast(a.ravel(), b.ravel(), out.ravel(), block_size, grid_size)
    return out.reshape(a.shape)
