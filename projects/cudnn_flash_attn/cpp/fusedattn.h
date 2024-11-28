#pragma once

#include <memory>

#include <cudnn_frontend.h>

// Tensors in forward pass
#define Q_UID 1
#define K_UID 2
#define V_UID 3
#define O_UID 4
#define STATS_UID 5
#define BIAS_UID 6

// NOTE: This wrapper only supports that q,k,v has same number of heads,
//       while cudnn supports group query attention!
// when is_fp16 = true, dtype is half
// when is_fp16 = false, dtype is bfloat16
class CudnnFlashAttention {
 public:
  CudnnFlashAttention(cudnnHandle_t handle, const int64_t batch,
                      const int64_t num_head, const int64_t head_dim,
                      const int64_t seq_q, const int64_t seq_kv,
                      const int64_t stride_qb,  // q.stride(0): batch stride
                      const int64_t stride_qs,  // q.stride(1): seq stride
                      const int64_t stride_qh,  // q.stride(2): head stride
                      const int64_t stride_kb,  // k.stride(0): batch stride
                      const int64_t stride_ks,  // k.stride(1): seq stride
                      const int64_t stride_kh,  // k.stride(2): head stride
                      const int64_t stride_vb,  // v.stride(0): batch stride
                      const int64_t stride_vs,  // v.stride(1): seq stride
                      const int64_t stride_vh,  // v.stride(2): head stride
                      const int64_t stride_ob,  // o.stride(0): batch stride
                      const int64_t stride_os,  // o.stride(1): seq stride
                      const int64_t stride_oh,  // o.stride(2): head stride
                      const float attn_scale, bool is_fp16, bool is_causal,
                      bool has_bias);

  // when false, it means input's parameters are wrong!
  bool GetStatus();

  int64_t GetBufferSize();

  bool DoForward(void* out, void* q, void* k, void* v, void* attn_bias,
                 void* workspace);

 private:
  std::shared_ptr<cudnn_frontend::graph::Graph> graph_;
  cudnnHandle_t handle_;
  bool has_bias_;
  bool status_ok_ = false;
};
