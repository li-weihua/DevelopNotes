**********************
deepseek模型学习笔记
**********************

.. figure:: /_static/images/deepseek_arch.png

Multi-Head Latent Attention (MLA)
====================================

.. note::

   约定所有计算用行向量，即 :math:`y = x * W`

Q的计算公式如下：

.. math::
    \begin{align}
    \mathbf{c}_{t}^{Q} &= \mathbf{h}_{t} W^{DQ}, \\
    [\mathbf{q}_{t, 1}^{C};\mathbf{q}_{t, 2}^{C};...;\mathbf{q}_{t, n_{h}}^{C}] = \mathbf{q}_{t}^{C} &= \mathbf{c}_{t}^{Q} W^{UQ}, \\
    [\mathbf{q}_{t, 1}^{R};\mathbf{q}_{t, 2}^{R};...;\mathbf{q}_{t, n_{h}}^{R}] = \mathbf{q}_{t}^{R} &= \operatorname{RoPE}(\mathbf{c}_{t}^{Q} {W^{QR}}), \\
    \mathbf{q}_{t, i} &= [\mathbf{q}_{t, i}^{C}; \mathbf{q}_{t, i}^{R}],
    \end{align}

where :math:`\mathbf{c}_{t}^{Q} \in \mathbb{R}^{d_c^{\prime}}` is the compressed latent vector for queries.
:math:`d_c^{\prime} (\ll d_h n_h)` denotes the query compression dimension;
:math:`W^{DQ} \in \mathbb{R}^{d \times d_c^{\prime}}, W^{UQ} \in \mathbb{R}^{d_c^{\prime} \times d_h n_h}` are the down-projection and up-projection matrices for queries, respectively;
and :math:`W^{QR} \in \mathbb{R}^{d_c^{\prime} \times d_h^R n_h}` is the matrix to produce the decoupled queries that carry RoPE.


KV的计算公式如下：

.. math::
    \begin{align}
    \boxed{\color{blue} \mathbf{c}_{t}^{KV}} &= \mathbf{h}_{t} W^{DKV}, \\
    [\mathbf{k}_{t, 1}^{C};\mathbf{k}_{t, 2}^{C};...;\mathbf{k}_{t, n_{h}}^{C}] = \mathbf{k}_{t}^{C} &= \mathbf{c}_{t}^{KV} W^{UK}, \\
    \boxed{\color{blue}\mathbf{k}_{t}^{R}} &= \operatorname{RoPE}(\mathbf{h}_{t} {W^{KR}}), \\
    \mathbf{k}_{t, i} &= [\mathbf{k}_{t, i}^{C}; \mathbf{k}_{t}^{R}], \\
    [\mathbf{v}_{t, 1}^{C};\mathbf{v}_{t, 2}^{C};...;\mathbf{v}_{t, n_{h}}^{C}] = \mathbf{v}_{t}^{C} &= \mathbf{c}_{t}^{KV} W^{UV},
    \end{align}


where :math:`\mathbf{c}_{t}^{KV} \in \mathbb{R}^{d_c}` is the compressed latent vector for keys and values;
:math:`d_c (\ll d_h n_h)` indicates the KV compression dimension;
:math:`W^{DKV} \in \mathbb{R}^{d \times d_c}` denotes the down-projection matrix;
:math:`W^{UK},W^{UV} \in \mathbb{R}^{d_c \times d_h n_h}` are the up-projection matrices for keys and values, respectively;
:math:`W^{KR} \in \mathbb{R}^{d \times d_h^R}` is the matrix used to produce the decoupled key that carries Rotary Positional Embedding (RoPE);
:math:`\operatorname{RoPE}(\cdot)` denotes the operation that applies RoPE matrices;
Note that for MLA, only the blue-boxed vectors (:math:`\color{blue} \mathbf{c}_{t}^{KV}` and :math:`\color{blue}\mathbf{k}_{t}^{R}`) need to be cached during generation,
which results in significantly reduced KV cache while maintaining performance comparable to standard Multi-Head Attention (MHA).

Ultimately, the attention queries (:math:`\mathbf{q}_{t, i}`), keys (:math:`\mathbf{k}_{j, i}`), and values (:math:`\mathbf{v}_{j, i}^{C}`) are combined to yield the final attention output :math:`\mathbf{u}_{t}`:

.. math::
    \begin{align}
        \mathbf{o}_{t, i} &= \sum_{j=1}^{t} \operatorname{Softmax}_j(\frac{\mathbf{q}_{t, i} \mathbf{k}^T_{j, i}}{\sqrt{d_{h} + d_{h}^{R}}}) \mathbf{v}_{j, i}^{C}, \\
        \mathbf{u}_{t} &= [\mathbf{o}_{t, 1};\mathbf{o}_{t, 2};...;\mathbf{o}_{t, n_{h}}] W^{O},
    \end{align}

where :math:`W^{O} \in \mathbb{R}^{d_h n_h \times d}` denotes the output projection matrix.


实际参数大小如下：

* d = hidden_size = 7168
* :math:`d_c` = kv_lora_rank = 512
* :math:`d_c^{\prime}` = q_lora_rank = 1536
* :math:`n_h` = num_heads = 128
* :math:`d_h` = qk_nope_head_dim = 128
* :math:`d_h^R` = qk_rope_head_dim = 64

:math:`W^{UQ}` 和 :math:`W^{QR}` 可以合并起来，q_head_dim = qk_nope_head_dim + qk_rope_head_dim  = 192.
:math:`W^{DKV}` 和 :math:`W^{KR}` 可以合并起来，kv_lora_rank + qk_rope_head_dim  = 576.


矩阵吸收absorb
------------------

首先考虑如下计算：

.. math::
    Y = X A B, \; C = A B

其中 :math:`X\in \mathbb{R}^{m\times d}` 是输入hidden states, :math:`A \in \mathbb{R}^{d \times d_c}` 和 :math:`B \in \mathbb{R}^{d_c \times n}` 是权重矩阵,
:math:`C\in \mathbb{R}^{d \times n}` 是合并后的等效权重矩阵， 直接计算的flops为：

.. math::
    2 m d d_c + 2 m n d_c = 2 m d_c (d + n)

合并权重后计算的flops为： :math:`2 m d n`

当 :math:`d_c` 相对较小时，通常导致 :math:`\boxed{d n \gt d_c (d + n)}`，所以不一定要合并两个权重矩阵！


先不考虑RoPE部分，只考虑从 :math:`\mathbf{c}^Q` 和 :math:`\mathbf{c}^{KV}` 计算 :math:`\mathbf{q} \mathbf{k}^T`

.. math::
    \begin{align}
    q k^T &= \mathbf{c}^{Q} W^{UQ} (\mathbf{c}^{KV} W^{UK})^T \\
              &= \boxed{\mathbf{c}^{Q} W^{UQ} (W^{UK})^T} (\mathbf{c}^{KV})^T, \\
              &= \boxed{q^{nope} (W^{UK})^T} (\mathbf{c}^{KV})^T, \\
    \end{align}
