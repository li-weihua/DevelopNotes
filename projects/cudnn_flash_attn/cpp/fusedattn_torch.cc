#include "fusedattn.h"

#include "ATen/cuda/CUDAContext.h"
#include "ATen/cudnn/Handle.h"
#include "torch/torch.h"
#include "torch/library.h"

namespace fe = cudnn_frontend;

class TorchCudnnFlashAttention : public torch::CustomClassHolder {
 public:
  TorchCudnnFlashAttention(int64_t batch, int64_t num_head, int64_t head_dim,
                           int64_t seq_q, int64_t seq_kv, int64_t stride_qb,
                           int64_t stride_qs, int64_t stride_qh, int64_t stride_kb,
                           int64_t stride_ks, int64_t stride_kh, int64_t stride_vb,
                           int64_t stride_vs, int64_t stride_vh, int64_t stride_ob,
                           int64_t stride_os, int64_t stride_oh, double attn_scale,
                           bool is_fp16, bool is_causal, bool has_bias) {
    auto handle = at::native::getCudnnHandle();

    cudnn_attn_ = std::make_unique<CudnnFlashAttention>(
        handle, batch, num_head, head_dim, seq_q, seq_kv, stride_qb, stride_qs,
        stride_qh, stride_kb, stride_ks, stride_kh, stride_vb, stride_vs, stride_vh,
        stride_ob, stride_os, stride_oh, static_cast<float>(attn_scale), is_fp16,
        is_causal, has_bias);

    TORCH_CHECK(cudnn_attn_->GetStatus(), "Input paramtes wrong!");

    int64_t workspace_size = cudnn_attn_->GetBufferSize();
    auto option = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    workspace_ = torch::empty({workspace_size}, option);
  }

  torch::Tensor DoForward(const torch::Tensor& q, const torch::Tensor& k,
                          const torch::Tensor& v,
                          const c10::optional<torch::Tensor>& bias) {
    // q shape: [batch, seq_q,  num_head, head_dim]
    // k shape: [batch, seq_kv, num_head, head_dim]
    // v shape: [batch, seq_kv, num_head, head_dim]
    TORCH_CHECK(q.stride(-1) == 1,
                "Input tensor q must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1,
                "Input tensor k must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1,
                "Input tensor v must have contiguous last dimension");

    // TODO: add some check

    auto out = torch::empty_like(q);

    TORCH_CHECK(cudnn_attn_->DoForward(out.data_ptr(), q.data_ptr(), k.data_ptr(),
                                       v.data_ptr(), bias.value().data_ptr(),
                                       workspace_.data_ptr()),
                "Forward Failed!");

    return out;
  }

 private:
  std::unique_ptr<CudnnFlashAttention> cudnn_attn_;

  torch::Tensor workspace_;
};

// op register
TORCH_LIBRARY(cudnn_flash_attn, m) {
  m.class_<TorchCudnnFlashAttention>("CudnnFlashAttention")
      .def(torch::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                       int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
                       int64_t, int64_t, int64_t, double, bool, bool, bool>())
      .def("forward", &TorchCudnnFlashAttention::DoForward);
}
