*************************
deep learning operators
*************************


LSTM
=======

The multi-layer LSTM can be implemented as:

.. literalinclude:: ../../python/multi_layer_lstm.py
    :language: python


方差计算
===========

方差定义（two-pass method）：

.. math::
	\begin{aligned}
    \bar{x}_n &= \frac{1}{n} \sum_{i=1}^{n} x_i \\
    \sigma^2_n &= \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x}_n)^2
	\end{aligned}

方差计算简化方式 (naive method, use one-pass)：

.. math::
	\begin{aligned}
    \sigma^2_n &= \frac{1}{n} \sum_{i=1}^{n} x_i^2 - \bar{x}_n^2
	\end{aligned}

Welford计算方差
------------------

Welford计算方差是用one-pass method，但误差远小于naive method。

均值的递推关系：

.. math::
    :label: welford_mean_recursive

    \bar{x}_n = \frac{(n-1)\bar{x}_{n-1} + x_n }{n} = \bar{x}_{n-1} + \frac{x_n - \bar{x}_{n-1}}{n}

方差的递推关系：

.. math::
	\begin{aligned}
    & n \sigma_n^2 - (n-1) \sigma_{n-1}^2 \\
    &= \sum_{i=1}^{n} (x_i - \bar{x}_n)^2 - \sum_{i=1}^{n-1} (x_i - \bar{x}_{n-1})^2 \\
    &= (x_n - \bar{x}_n)^2 + \sum_{i=1}^{n-1} \left( (x_i - \bar{x}_n)^2 - (x_i - \bar{x}_{n-1})^2 \right) \\
    &= (x_n - \bar{x}_n)^2 + \sum_{i=1}^{n-1} \left( (x_i - \bar{x}_n)^2 - (x_i - \bar{x}_{n-1})^2 \right) \\
    &= (x_n - \bar{x}_n)^2 + \sum_{i=1}^{n-1} (2 x_i - \bar{x}_n - \bar{x}_{n-1}) (\bar{x}_{n-1} - \bar{x}_n) \\
    &= (x_n - \bar{x}_n)^2 + (\bar{x}_n - x_n) (\bar{x}_{n-1} - \bar{x}_n) \\
    &= (x_n - \bar{x}_n) (x_n - \bar{x}_{n-1})
	\end{aligned}

定义 :math:`M_n` (文献中通常定义成 :math:`M_{2,n}` ) 如下：

.. math::
    M_{n} = \sum_{i=1}^n (x_i - \bar{x}_n)^2

可以得到Welford算法如下：

.. math::
    :label: welford_var_recursive

	\begin{aligned}
    M_{n} &= M_{n-1} + (x_n - \bar{x}_n) (x_n - \bar{x}_{n-1}) \\
    \sigma_n^2 &= M_{n} / n
	\end{aligned}

当 :math:`x_n` 偏离均值比较多的时候，:math:`x_n - \bar{x}_n` 比较小，:math:`x_n - \bar{x}_{n-1}` 比较大， 线性偏差。

python实现示例如下：

.. literalinclude:: ../../python/welford.py
    :language: python

c++实现示例
-------------

demo只考虑样本数是4的倍数。

c++实现示例 `WelfordCpp`_， 支持x86 sse和arm neon指令。

.. _WelfordCpp: https://github.com/li-weihua/DevelopNotes/tree/main/projects/variance


除法计算和Newton-Raphson iteration
-------------------------------------

均值递推关系 :eq:`welford_mean_recursive` 需要计算除法 ``1/n``，但除法计算的延迟比较高。

`Newton's method (Newton-Raphson method) <https://en.wikipedia.org/wiki/Newton%27s_method>`_:

.. math::

    x_1 = x_0 - \frac{f(x_0)}{f^{\prime}(x_0)}


`Newton–Raphson division`_ is a fast method to calculate the reciprocal of a number :math:`a`.
We can define :math:`f(x) = 1/x - a` and thus :math:`f^{\prime}(x) = -1/x^2`.

Then Newton's iteration is:

.. math::
    \begin{aligned}
    x_{n+1} &= x_n - \frac{f(x_n)}{f^{\prime}(x_n)} \\
            &= x_n - \frac{\frac{1}{x_n} - a}{-\frac{1}{x_n^2}} \\
            &= x_n (2 - a x_n)
    \end{aligned}

为什么不选去其他函数，比如 :math:`f(x) = a x - 1` ，主要是收敛性和收敛速度等决定的。

ARM neon实现示例如下：

.. code:: c++

    float32x4_t fast_reciprocal(float32x4_t a) {
        float32x4_t recip = vrecpeq_f32(a);

        // Newton-Raphson iteration two times
        recip = vmulq_f32(recip, vrecpsq_f32(recip, a));
        recip = vmulq_f32(recip, vrecpsq_f32(recip, a));

        return recip;
    }

``vrecpsq_f32`` 就是计算 :math:`2.0 - a * x` 。

.. _Newton–Raphson division: https://en.wikipedia.org/wiki/Newton%27s_method#Multiplicative_inverses_of_numbers_and_power_series

.. todo::

    cuda layernorm


相关链接
------------

:参考1: https://mp.weixin.qq.com/s/t0x782mDkMo-ZBVEbK8gPg
:参考2: https://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/
