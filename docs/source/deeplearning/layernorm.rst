方差和layernorm
================

方差计算
--------

方差定义：

.. math::
	\begin{aligned}
    \bar{x}_n &= \frac{1}{n} \sum_{i=1}^{n} x_i \\
    \sigma^2_n &= \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x}_n)^2
	\end{aligned}

均值的递推关系：

.. math::
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
	\begin{aligned}
    M_{n} &= M_{n-1} + (x_n - \bar{x}_n) (x_n - \bar{x}_{n-1}) \\
    \sigma_n^2 &= M_{n} / n
	\end{aligned}

当 :math:`x_n` 偏离均值比较多的时候，:math:`x_n - \bar{x}_n` 比较小，:math:`x_n - \bar{x}_{n-1}` 比较大， 线性偏差。

python实现示例如下：

.. literalinclude:: ../../../python/welford.py
    :language: python


相关链接
---------

:参考1: https://mp.weixin.qq.com/s/t0x782mDkMo-ZBVEbK8gPg
:参考2: https://jonisalonen.com/2013/deriving-welfords-method-for-computing-variance/
