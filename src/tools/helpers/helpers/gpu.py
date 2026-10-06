"""Utilities for selecting a CUDA device for model inference."""

import torch


def select_cuda_device():
    """Return the GPU with the most free memory, or CPU if CUDA is unavailable.

    Memory is checked at call time. This does not reserve the memory, so it can
    change before a model finishes loading.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")

    device_count = torch.cuda.device_count()
    if device_count == 0:
        return torch.device("cpu")

    free_memory = [
        torch.cuda.mem_get_info(index)[0] for index in range(device_count)
    ]
    selected_index = max(range(device_count), key=free_memory.__getitem__)
    return torch.device(f"cuda:{selected_index}")


if __name__ == "__main__":
    print(f"Selected device: {select_cuda_device()}")
