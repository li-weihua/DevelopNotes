from typing import Optional

import torch
import cuda_mma

def matmul_8x8x16_s8(A: torch.Tensor, B: torch.Tensor, C: Optional[torch.Tensor] = None) -> torch.Tensor:
    return cuda_mma.matmul_8x8x16_s8(A, B, C)
