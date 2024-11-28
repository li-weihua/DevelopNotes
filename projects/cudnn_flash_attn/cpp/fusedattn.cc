#include "fusedattn.h"

#include <cudnn.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cassert>
#include <chrono>
#include <memory>

namespace fe = cudnn_frontend;

CudnnFlashAttention::CudnnFlashAttention(
    cudnnHandle_t handle, const int64_t batch, const int64_t num_head,
    const int64_t head_dim, const int64_t seq_q, const int64_t seq_kv,
    const int64_t stride_qb, const int64_t stride_qs, const int64_t stride_qh,
    const int64_t stride_kb, const int64_t stride_ks, const int64_t stride_kh,
    const int64_t stride_vb, const int64_t stride_vs, const int64_t stride_vh,
    const int64_t stride_ob, const int64_t stride_os, const int64_t stride_oh,
    const float attn_scale, bool is_fp16, bool is_causal, bool has_bias)
    : handle_(handle), has_bias_(has_bias) {
  graph_ = std::make_shared<fe::graph::Graph>();

  if (is_fp16) {
    graph_->set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
  } else {
    graph_->set_io_data_type(fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
  }

  auto Q = graph_->tensor(fe::graph::Tensor_attributes()
                              .set_name("Q")
                              .set_uid(Q_UID)
                              .set_dim({batch, num_head, seq_q, head_dim})
                              .set_stride({stride_qb, stride_qh, stride_qs, 1}));

  // TODO: support max_s_kv
  auto K = graph_->tensor(fe::graph::Tensor_attributes()
                              .set_name("K")
                              .set_uid(K_UID)
                              .set_dim({batch, num_head, seq_kv, head_dim})
                              .set_stride({stride_kb, stride_kh, stride_ks, 1}));

  auto V = graph_->tensor(fe::graph::Tensor_attributes()
                              .set_name("V")
                              .set_uid(V_UID)
                              .set_dim({batch, num_head, seq_kv, head_dim})
                              .set_stride({stride_vb, stride_vh, stride_vs, 1}));

  auto sdpa_options = fe::graph::SDPA_attributes()
                          .set_name("flash_attention")
                          .set_is_inference(true)
                          .set_alibi_mask(false)
                          .set_causal_mask(is_causal)
                          .set_attn_scale(attn_scale);

  if (has_bias_) {
    auto bias =
        graph_->tensor(fe::graph::Tensor_attributes()
                           .set_name("bias")
                           .set_uid(BIAS_UID)
                           .set_dim({1, 1, seq_q, seq_kv})
                           .set_stride({seq_q * seq_kv, seq_q * seq_kv, seq_kv, 1}));
    sdpa_options.set_bias(bias);
  }

  auto [O, Stats] = graph_->sdpa(Q, K, V, sdpa_options);

  O->set_output(true)
      .set_dim({batch, num_head, seq_q, head_dim})
      .set_stride({stride_ob, stride_oh, stride_os, 1})
      .set_uid(O_UID);

  // inference
  assert(Stats == nullptr);

  status_ok_ = graph_->build(handle_, {fe::HeurMode_t::A}).is_good();
}

bool CudnnFlashAttention::GetStatus() { return status_ok_; }

int64_t CudnnFlashAttention::GetBufferSize() {
  int64_t workspace_size = 0;

  status_ok_ &= graph_->get_workspace_size(workspace_size).is_good();

  if (status_ok_)
    return workspace_size;
  else
    return -1;
}

bool CudnnFlashAttention::DoForward(void* out, void* q, void* k, void* v,
                                    void* attn_bias, void* workspace) {
  std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
      {Q_UID, q}, {K_UID, k}, {V_UID, v}, {O_UID, out}};

  if (has_bias_) variant_pack[BIAS_UID] = attn_bias;

  status_ok_ &= graph_->execute(handle_, variant_pack, workspace).is_good();

  return status_ok_;
}
