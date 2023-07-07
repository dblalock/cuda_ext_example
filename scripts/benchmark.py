
import numpy as np
from typing import Callable, Tuple

import torch

import cuda_ext_example as ours


def time_elemwise_op(f: Callable,
                     shape: Tuple[int] = (1 << 28,),
                     dtype: torch.dtype = torch.float32,
                     num_iters_per_trial: int = 5,
                     num_trials: int = 3):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cuda'

    # Generate random tensors
    a = torch.randint(10, size=shape, device=device, dtype=dtype)
    b = torch.randint(10, size=shape, device=device, dtype=dtype)
    c = torch.empty(shape, device=device, dtype=dtype)

    # Warmup iters
    for _ in range(3):
        f(a, b, out=c)

    times = []
    for trial in range(num_trials):
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_time.record()
        for _ in range(num_iters_per_trial):
            f(a, b, out=c)
        end_time.record()
        torch.cuda.synchronize()

        ms = start_time.elapsed_time(end_time)
        times.append(ms / num_iters_per_trial)

    return np.array(times) / 1000. # convert to seconds


def main():
    # shape = (1 << 14, 1 << 14)
    shape = (1 << 28,)
    times_torch = time_elemwise_op(torch.add, shape=shape)
    times_simple = time_elemwise_op(ours.my_add, shape=shape)
    times_fast = time_elemwise_op(ours.my_fast_add, shape=shape)

    print("times_ms_torch: ", times_torch * 1000)
    print("times_ms_simple:", times_simple * 1000)
    print("times_ms_fast:  ", times_fast * 1000)

    # num_bytes = np.prod(shape) * 4 * 3  # 4 bytes per float, 3 arrays
    num_bytes = shape[0] * 4 * 3  # 4 bytes per float, 3 arrays
    num_tb = num_bytes / 1e12
    print("thruputs_torch (TB/s): ", num_tb / times_torch)
    print("thruputs_simple (TB/s):", num_tb / times_simple)
    print("thruputs_fast (TB/s):  ", num_tb / times_fast)



if __name__ == '__main__':
    # np.set_printoptions(precision=3, suppress=True)
    np.set_printoptions(formatter={'float': lambda x: f'{x:.4f}'})
    main()
