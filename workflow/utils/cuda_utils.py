from typing import Optional

import torch


def get_total_vram_gb() -> Optional[float]:
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not available.')

    properties = torch.cuda.get_device_properties(0)
    total_vram_gb = properties.total_memory / (1024**3)
    print(f'[Info] Detected GPU: {properties.name}, Total VRAM: {total_vram_gb:.2f} GB')
    return total_vram_gb
