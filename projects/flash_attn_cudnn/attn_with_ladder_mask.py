import cudnn
import torch
import math

torch.set_grad_enabled(False)
torch.manual_seed(42)
handle = cudnn.create_handle()

assert torch.cuda.is_available()
assert (
    torch.cuda.get_device_capability()[0] >= 8
), "SDPA operation is only supported on SM80 architecture (Ampere) or above"
assert (
    cudnn.backend_version() >= 8903
), "SDPA operation is only supported cuDNN version 8.9.3 or above"


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
max_s = 2048     # max sequence length
b = 2            # batch size
h = 16           # query number of heads
s = 622          # real sequence length
d = 64           # embedding dimension per head

assert (s - text_len) % chunk_size == 0

prefill_mask = gen_ladder_mask(s, text_len, chunk_size).cuda()

attn_scale = 1.0 / math.sqrt(d)

# The tensors will have non-interleaved
# BSHD (batch, sequence_length, num_head, dims_per_head) physical tensor layout
# BHSD (batch, num_head, sequence_length, dims_per_head) logical tensor layout
dims = (b, h, s, d)
strides = (s * h * d, d, h * d, 1)

q_gpu = torch.randn(b * s * h * d).half().cuda().as_strided(dims, strides)
o_gpu = torch.empty(b * s * h * d).half().cuda().as_strided(dims, strides)

cache_dims = (b, h, max_s, d)
cache_strides = (max_s* h * d, d, h * d, 1)

k_cache_gpu = torch.randn(b * max_s * h * d).half().cuda().as_strided(cache_dims, cache_strides)
v_cache_gpu = torch.randn(b * max_s * h * d).half().cuda().as_strided(cache_dims, cache_strides)

k_cache_gpu[:,:,s:,:] = 0
v_cache_gpu[:,:,s:,:] = 0

k_gpu = k_cache_gpu[:, :, 0:s, :].as_strided(dims, strides)
v_gpu = v_cache_gpu[:, :, 0:s, :].as_strided(dims, strides)

# bias_gpu is ladder mask!
bias_gpu = torch.zeros(b, h, s, s, dtype=q_gpu.dtype, device=q_gpu.device)
bias_gpu.masked_fill_(~prefill_mask, float("-inf")).to(q_gpu.dtype)

graph = cudnn.pygraph(
    io_data_type=cudnn.data_type.HALF,
    intermediate_data_type=cudnn.data_type.FLOAT,
    compute_data_type=cudnn.data_type.FLOAT,
)

q = graph.tensor_like(q_gpu)
k = graph.tensor_like(k_gpu)
v = graph.tensor_like(v_gpu)
bias = graph.tensor_like(bias_gpu)

# the second return for the stats tensor is used for training only.
# causal mask is enabled
o, _ = graph.sdpa(
    name="sdpa",
    q=q,
    k=k,
    v=v,
    bias=bias,
    is_inference=True,
    attn_scale=attn_scale,
)

o.set_output(True).set_dim(dims).set_stride(strides)

# build graph
graph.validate()
graph.build_operation_graph()
graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
graph.check_support()
graph.build_plans()

# execute graph
variant_pack = {
    q: q_gpu,
    k: k_gpu,
    v: v_gpu,
    bias: bias_gpu,
    o: o_gpu,
}

workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
graph.execute(variant_pack, workspace)
torch.cuda.synchronize()

# check results
q_ref = q_gpu.detach()
k_ref = k_gpu.detach()
v_ref = v_gpu.detach()
bias_ref = bias_gpu.detach()

o_ref = torch.nn.functional.scaled_dot_product_attention(
    q_ref, k_ref, v_ref, attn_mask=bias_ref, scale=attn_scale
)
torch.testing.assert_close(o_ref, o_gpu, atol=5e-3, rtol=3e-3)

print(f"torch: {o_ref.shape}, {o_ref.min()}, {o_ref.max()}")
print(f"cudnn: {o_gpu.shape}, {o_gpu.min()}, {o_gpu.max()}")
print(f"diff : {(o_gpu - o_ref).abs().max()}")

# benchmark
start = torch.cuda.Event(enable_timing=True)
end1 = torch.cuda.Event(enable_timing=True)
end2 = torch.cuda.Event(enable_timing=True)

n = 10

for i in range(n):
    start.record()
    o_ref = torch.nn.functional.scaled_dot_product_attention(q_ref, k_ref, v_ref, attn_mask=bias_ref, scale=attn_scale)
    end1.record()
    graph.execute(variant_pack, workspace)
    end2.record()
    end2.synchronize()

    print(f"Iter(μs) {i}: {start.elapsed_time(end1)*1000:.1f}, {end1.elapsed_time(end2)*1000:.1f}")

