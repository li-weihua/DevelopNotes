#include <memory>
#include <ratio>
#include "helpers.h"
#include "fusedattn.h"

namespace fe = cudnn_frontend;

int main(int argc, char* argv[]) {
  if (cudnnGetVersion() < 8903) {
    std::cerr << "Test requires cudnn 8.9.3 or above" << std::endl;
    return -1;
  }

  int64_t batch = 2;      // batch size
  int64_t num_head = 16;  // number of heads
  int64_t head_dim = 64;  // head dim
  int64_t seq_q = 622;    // q tensor
  int64_t seq_kv = 622;   // k and v tensor is padded to this seq length

  // q,k,v,o strides, real memory order: (batch, sequence, num_head, head_dim)
  int64_t ostrides[3] = {seq_q * num_head * head_dim, num_head * head_dim, head_dim};
  int64_t qstrides[3] = {seq_q * num_head * head_dim, num_head * head_dim, head_dim};
  int64_t kstrides[3] = {seq_kv * num_head * head_dim, num_head * head_dim,
                         head_dim};
  int64_t vstrides[3] = {seq_kv * num_head * head_dim, num_head * head_dim,
                         head_dim};

  float attn_scale = 1.0f / sqrt(head_dim);

  cudnnHandle_t handle;
  CUDNN_CHECK(cudnnCreate(&handle));

  bool is_fp16 = false;
  bool is_causal = false;
  bool has_bias = false;

  auto start = std::chrono::high_resolution_clock::now();

  auto fused_attn = std::make_unique<CudnnFlashAttention>(
      handle, batch, num_head, head_dim, seq_q, seq_kv, qstrides[0], qstrides[1],
      qstrides[2], kstrides[0], kstrides[1], kstrides[2], vstrides[0], vstrides[1],
      vstrides[2], ostrides[0], ostrides[1], ostrides[2], attn_scale, is_fp16,
      is_causal, has_bias);

  REQUIRE(fused_attn->GetStatus());

  auto end = std::chrono::high_resolution_clock::now();
  float time = std::chrono::duration<float, std::milli>(end - start).count();
  std::cout << "Build graph time: " << time << " ms" << std::endl;

  int64_t workspace_size = fused_attn->GetBufferSize();
  Surface<int8_t> workspace(workspace_size, false);

  // Build variant pack
  Surface<half> q_tensor(batch * seq_q * num_head * head_dim, false);
  Surface<half> k_tensor(batch * seq_kv * num_head * head_dim, false);
  Surface<half> v_tensor(batch * seq_kv * num_head * head_dim, false);
  Surface<half> o_tensor(batch * seq_q * num_head * head_dim, false);
  Surface<half> bias_tensor(1 * 1 * seq_q * seq_kv, false);

  for (int i = 0; i < 10; ++i) {
    start = std::chrono::high_resolution_clock::now();
    REQUIRE(fused_attn->DoForward(o_tensor.devPtr, q_tensor.devPtr, k_tensor.devPtr,
                                  v_tensor.devPtr, bias_tensor.devPtr,
                                  workspace.devPtr));
    CUDA_CHECK(cudaDeviceSynchronize());

    end = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration<float, std::micro>(end - start).count();
    std::cout << "Execute graph time: " << time << " μs" << std::endl;
  }

  cudnnDestroy(handle);

  return 0;
}
