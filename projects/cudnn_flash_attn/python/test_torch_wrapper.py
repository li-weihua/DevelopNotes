import math
import torch
import torch.nn.functional as F

torch.classes.load_library('../build/libfusedattn_torch.so')

torch.manual_seed(10)
torch.set_grad_enabled(False)

def gen_ladder_mask(total_len, text_len, chunk_size):
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

    return mask.bool()


text_len = 142
chunk_size = 32
max_s = 2048  # max sequence length
b = 2  # batch size
h = 16  # query number of heads
s = 622  # real sequence length
d = 64  # embedding dimension per head

dtype = torch.bfloat16
device = torch.device('cuda')

assert (s - text_len) % chunk_size == 0

prefill_mask = gen_ladder_mask(s, text_len, chunk_size).cuda()

attn_scale = 1.0 / math.sqrt(d)

dims = (b, s, h, d)
cache_dims = (b, max_s, h, d)
cache_strides = (max_s * h * d, h * d, d, 1)

q = torch.randn(b, s, h, d, dtype=dtype, device=device)
o = torch.empty(b, s, h, d, dtype=dtype, device=device)

k_cache_gpu = torch.randn(b * max_s * h * d, dtype=dtype, device=device).as_strided(cache_dims, cache_strides)
v_cache_gpu = torch.randn(b * max_s * h * d, dtype=dtype, device=device).as_strided(cache_dims, cache_strides)
k_cache_gpu[:, s:] = 0
v_cache_gpu[:, s:] = 0

k = k_cache_gpu[:, 0:s].as_strided(dims, cache_strides)
v = v_cache_gpu[:, 0:s].as_strided(dims, cache_strides)

print(f"strides information:")
print(f"q strides: {q.stride()}")
print(f"k strides: {k.stride()}")
print(f"v strides: {v.stride()}")
print()

#k = torch.randn(b, s, h, d, dtype=dtype, device=device)
#v = torch.randn(b, s, h, d, dtype=dtype, device=device)

# bias_gpu is ladder mask!
bias = torch.zeros(1, 1, s, s, dtype=q.dtype, device=q.device)
bias.masked_fill_(~prefill_mask, float("-inf")).to(q.dtype)

# reference
q_ref = q.clone().transpose(1, 2).contiguous()
k_ref = k.clone().transpose(1, 2).contiguous()
v_ref = v.clone().transpose(1, 2).contiguous()

is_fp16 = False
if q.dtype == torch.float16:
    is_fp16 = True

# test causal
for is_causal in [False, True]:
    o_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, scale=attn_scale, is_causal=is_causal).transpose(1,2)
    cudnn_attn = torch.classes.cudnn_flash_attn.CudnnFlashAttention(q.size(0), q.size(2), q.size(3), q.size(1), k.size(1),
                                                                    q.stride(0), q.stride(1), q.stride(2),
                                                                    k.stride(0), k.stride(1), k.stride(2),
                                                                    v.stride(0), v.stride(1), v.stride(2),
                                                                    q.stride(0), q.stride(1), q.stride(2),
                                                                    attn_scale, is_fp16, is_causal, False)
    o_gpu = cudnn_attn.forward(q, k, v, bias)

    print(f"causal: {is_causal}")
    print(f"torch : {o_ref.shape}, {o_ref.min().item()}, {o_ref.max().item()}")
    print(f"cudnn : {o_gpu.shape}, {o_gpu.min().item()}, {o_gpu.max().item()}")
    print(f"diff  : {(o_gpu - o_ref).abs().max().item()}")
    print()

# test ladder mask
o_ref = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, attn_mask=bias, scale=attn_scale).transpose(1,2)
cudnn_attn = torch.classes.cudnn_flash_attn.CudnnFlashAttention(q.size(0), q.size(2), q.size(3), q.size(1), k.size(1),
                                                                q.stride(0), q.stride(1), q.stride(2),
                                                                k.stride(0), k.stride(1), k.stride(2),
                                                                v.stride(0), v.stride(1), v.stride(2),
                                                                q.stride(0), q.stride(1), q.stride(2),
                                                                attn_scale, is_fp16, False, True)
o_gpu = cudnn_attn.forward(q, k, v, bias)

print(f"ladder mask")
print(f"torch : {o_ref.shape}, {o_ref.min().item()}, {o_ref.max().item()}")
print(f"cudnn : {o_gpu.shape}, {o_gpu.min().item()}, {o_gpu.max().item()}")
print(f"diff  : {(o_gpu - o_ref).abs().max().item()}")
print()

# benchmark
start = torch.cuda.Event(enable_timing=True)
end1 = torch.cuda.Event(enable_timing=True)
end2 = torch.cuda.Event(enable_timing=True)

n = 10

for i in range(n):
    start.record()
    F.scaled_dot_product_attention(q_ref, k_ref, v_ref, attn_mask=bias, scale=attn_scale)
    end1.record()
    cudnn_attn.forward(q, k, v, bias)
    end2.record()
    end2.synchronize()
    print(f"Iter(μs) {i}: {start.elapsed_time(end1)*1000:.1f}, {end1.elapsed_time(end2)*1000:.1f}")
