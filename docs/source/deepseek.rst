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
=================

首先考虑如下计算：

.. math::
    Y = X A B, \; C = A B

其中 :math:`X\in \mathbb{R}^{m\times d}` 是输入hidden states, :math:`A \in \mathbb{R}^{d \times d_c}` 和 :math:`B \in \mathbb{R}^{d_c \times n}` 是权重矩阵,
:math:`C\in \mathbb{R}^{d \times n}` 是absorb后的等效权重矩阵， 直接计算的flops为：

.. math::
    2 m d d_c + 2 m n d_c = 2 m d_c (d + n)

合并权重后计算的flops为： :math:`2 m d n`

当 :math:`d_c` 相对较小时，通常导致 :math:`\boxed{d n \gt d_c (d + n)}`，所以不一定要合并两个权重矩阵！


先不考虑RoPE部分，只考虑从 :math:`\mathbf{c}^Q` 和 :math:`\mathbf{c}^{KV}` 计算 :math:`\mathbf{q}_i \mathbf{k}_i^T` (i表示i-th head)

.. math::
    \begin{align*}
    q_i k_i^T &= \boxed{\mathbf{c}^{Q} W^{UQ}_i} \; \boxed{(\mathbf{c}^{KV} W^{UK}_i)^T}, \\
              &= \boxed{\mathbf{c}^{Q} W^{UQ}_i (W^{UK}_i)^T} (\mathbf{c}^{KV})^T, & \\
              &= \boxed{q_i (W^{UK}_i)^T} (\mathbf{c}^{KV})^T,  & \boxed{\textrm{Absorb}} \\
              &= q_i \boxed{(\mathbf{c}^{KV} W^{UK}_i)^T},  & \boxed{\textrm{Normal}} \\
    \end{align*}


为什么计算的时候不把 :math:`W^{UQ}_i  (W^{UK}_i)^T` 合并起来？
------------------------------------------------------------

可以简单的计算出来对于单个token，单个head所需要的flops分别为： :math:`2 d_h (d_c^{\prime} + d_c) = 524288` ,
:math:`2 d_c^{\prime} d_c = 1572864 = 3 * 524288` ,
合并后计算量反而是原来的3倍！


为什么prefill阶段明确计算出k和v，而decode阶段不需要？
----------------------------------------------------

假定输入shape如下：

.. math::
    \begin{align*}
    \mathbf{q} &: (b, n_h, s_q, d_h) \\
    \mathbf{c}^{KV} &: (b, 1, s_{kv}, d_c) \\
    W^{UK} &: (d_c, n_h d_h) \\
    \end{align*}

可以计算出 ``Normal`` 和 ``Absorb`` 计算出的flops分别如下：

.. math::
    \begin{align*}
    T_{\textrm{Normal}} &= 2 b s_{kv} d_c d_h n_h + 2 b n_h s_q s_{kv} d_h = 2 b n_h d_h (d_c s_{kv} + s_q s_{kv}), \\
    T_{\textrm{Absorb}} &= 2 b s_q d_c d_h n_h + 2 b n_h s_q s_{kv} d_c = 2 b n_h d_c (d_h s_q + s_q s_{kv}), \\
    \end{align*}

**Prefill** 阶段 :math:`s_q = s_{kv} = s`，

.. math::
    \frac{T_{\textrm{Normal}}}{T_{\textrm{Absorb}}} = \frac{ 2 b n_h d_h (d_c + s) s}{2 b n_h d_c (d_h + s) s} \approx \frac{d_h}{d_c} = \frac{1}{4}


**Decode** 阶段 :math:`s_q = 1, s_{kv} = s`，

.. math::
    \frac{T_{\textrm{Normal}}}{T_{\textrm{Absorb}}} = \frac{ 2 b n_h d_h (d_c s + s)}{2 b n_h d_c (d_h + s)} = \frac{d_h (d_c + 1) s}{d_c (d_h + s)}
    \approx d_h

从计算量上看，**Prefill** 阶段 ``Normal`` 的计算量比较小，且由于 **Prefill** 阶段是 ``计算瓶颈``，所以采用公式(15)或者(12)计算，即 **显式的计算出q和k**。

而 **Decode** 阶段 ``Absorb`` 方式的计算量小，且瓶颈是 ``显存带宽``，矩阵运算是 :math:`(b, n_h, 1, d_c) \times (b, 1, s, d_c)`，假定为bfloat16精度，读取的memory为

.. math::
    M_{\textrm{MLA}} = 2 b n_h d_c + 2 b s d_c = 2 b d_c (n_h + s).

而标准的MHA :math:`(b, n_h, 1, d_h)\times (b, n_h, s, d_h)` 的内存读取量为：

.. math::
    M_{\textrm{MHA}} = 2 b n_h d_h + 2 b n_h s d_h = 2 b d_h n_h (1 + s).

内存读取比例为：

.. math::
    \frac{M_{\textrm{MLA}}}{M_{\textrm{MHA}}} = \frac{2 b d_c (n_h + s)}{2 b d_h n_h (1 + s)} = \frac{128 + s}{ 32 (1 + s)} \approx \frac{1}{32}.

所以 **Decode** 阶段采用了 ``Absorb`` 方式计算，并可以复用MQA (Multi-Query Attention) 的实现。
