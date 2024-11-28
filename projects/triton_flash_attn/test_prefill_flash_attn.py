from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from flash_attn_triton import _flash_attn_forward

torch.manual_seed(10)
torch.set_grad_enabled(False)

def gen_mask(total_len, text_len, chunk_size):
    mask = torch.ones([total_len, total_len], dtype=torch.int)

    for row in range(total_len):
        for col in range(total_len):
            is_mask_elem = False

            if row < text_len and col >= text_len:
                    is_mask_elem = True

            if row >= text_len and col >= text_len:
                row_a = row - text_len
                col_a = col - text_len

                if col_a >= (1 + row_a // chunk_size) * chunk_size:
                    is_mask_elem = True

            if is_mask_elem:
                mask[row, col] = 0
    return mask

def DoAttention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, mask=None):
    batch, seqlen, num_heads, head_dim  = q.shape

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    bias = torch.zeros(batch, num_heads, seqlen, seqlen, dtype=q.dtype, device=q.device)
    if mask is not None:
        bias.masked_fill_(mask.logical_not(), float("-inf"))

    #return F.scaled_dot_product_attention(q, k, v, attn_mask=bias).transpose(1,2)
    p = torch.matmul(q, k.transpose(-1, -2)) * scale
    if mask is not None:
        p += bias
    p = torch.nn.functional.softmax(p, dim=-1)
    out = torch.matmul(p, v)

    out = out.transpose(1, 2)
    return out


batch_size = 2
text_len = 142
chunk_size = 32
seq_len = 622
num_heads = 16
head_dim = 64

device = torch.device("cuda")
dtype = torch.bfloat16

q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
k = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
v = torch.rand(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)

q = q.to(dtype)
k = k.to(dtype)
v = v.to(dtype)

q0 = q.clone().contiguous()
k0 = k.clone().contiguous()
v0 = v.clone().contiguous()

q1 = q.clone().contiguous()
k1 = k.clone().contiguous()
v1 = v.clone().contiguous()

softmax_scale = 1.0 / sqrt(head_dim)

mask = gen_mask(seq_len, text_len, chunk_size).to(device)

y0 = DoAttention(q0, k0, v0, softmax_scale, mask)

local_bias = mask.logical_not().reshape(1, 1, seq_len, seq_len).to(q.dtype).to(q.device) * -10000
y1, _, _ = _flash_attn_forward(q, k, v, local_bias)


# check result
print(f"precison: {dtype}")
print(f"ref : {y0.shape}, {y0.min()}, {y0.max()}")
print(f"cuda: {y1.shape}, {y1.min()}, {y1.max()}")
print(f"diff: {(y1-y0).abs().max().item()}")
print()

# benchmark performance
start = torch.cuda.Event(enable_timing=True)
end1 = torch.cuda.Event(enable_timing=True)
end2 = torch.cuda.Event(enable_timing=True)

n = 10

for i in range(n):
    start.record()
    y0 = DoAttention(q0, k0, v0, softmax_scale, mask)
    end1.record()
    y1, _, _ = _flash_attn_forward(q, k, v, local_bias)
    end2.record()
    end2.synchronize()

    print(f"Iter(μs) {i}: {start.elapsed_time(end1)*1000:.1f}, {end1.elapsed_time(end2)*1000:.1f}")
