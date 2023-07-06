
import numpy as np
from typing import Callable, Tuple

import torch

import cuda_ext_example as ours


def time_elemwise_op(f: Callable,
                     shape: Tuple[int] = (1024, 1024),
                     dtype: torch.dtype = torch.float32,
                     num_iters_per_trial: int = 10,
                     num_trials: int = 3):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cuda'

    # Generate random tensors
    a = torch.randint(10, size=shape, device=device, dtype=dtype)
    b = torch.randint(10, size=shape, device=device, dtype=dtype)
    c = torch.empty(shape, device=device, dtype=dtype)

    for _ in range(3):
        f(a, b, out=c)

    # Warmup iters
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)

    times = []
    for trial in range(num_trials):
        torch.cuda.synchronize()
        start_time.record()
        for _ in range(num_iters_per_trial):
            f(a, b, out=c)
        end_time.record()
        torch.cuda.synchronize()

        t = start_time.elapsed_time(end_time)
        times.append(t / num_iters_per_trial)

    return np.array(times)


def main():
    times_torch = time_elemwise_op(torch.add)
    times_simple = time_elemwise_op(ours.my_add)
    times_fast = time_elemwise_op(ours.my_fast_add)

    print("times_torch: ", times_torch)
    print("times_simple:", times_torch)
    print("times_fast:  ", times_fast)



if __name__ == '__main__':
    # np.set_printoptions(precision=3, suppress=True)
    np.set_printoptions(formatter={'float': lambda x: f'{x:.4f}'})
    main()
