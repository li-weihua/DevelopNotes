Backpropagation的推导
=====================

约定
^^^^

.. math:: 
  z^{l+1}_j =\sum_k w^l_{jk} a^l_k + b^l_j, \quad a^l_j=\sigma(z^l_j) 
  :label: explicitform

其中，:math:`z^l_j` 表示未激活前第 :math:`l` 层、第 :math:`j` 个神经元的值，:math:`w^l_{jk}` 为连接第 :math:`l` 层第 :math:`j` 个神经元和第 :math:`l+1` 层第k个神经元的权重，
:math:`a^l_k` 表示激活后的第 :math:`l` 层第 :math:`k` 个神经元的值，
:math:`b^l_j` 为偏移量bias，:math:`\sigma` 为激活函数。

注意和书籍 http://neuralnetworksanddeeplearning.com/chap2.html 中公式(23)的约定由区别，我们把weight和bias和神经元的值放到同一层中。

公式 :eq:`explicitform` 写成矩阵形式为：

.. math::
	z^{l+1}=w^l a^l + b^l, \quad a^l=\sigma(z^l)


公式推导
^^^^^^^^

我们约定C为损失函数（loss function），并记：

.. math:: 
	\delta^l = \frac{\partial C}{\partial z^l}

约定 **Hadamard product** 或者elementwise相乘为（重复指标不求和）：

.. math::
	u\odot v = u_i * v_i


根据公式 :eq:`explicitform` 可以直接得出对偏移量 :math:`b` 的偏导数（梯度）：

.. math::
	\frac{\partial C}{\partial b^l_j} = \sum_i \frac{\partial C}{\partial z^{l+1}_i} \frac{\partial z^{l+1}_i}{\partial b^l_j} = \frac{\partial C}{z^{l+1}_j} = \delta^{l+1}_j

上式写成矩阵形式为：

.. math::
	\frac{\partial C}{\partial b^l} = \delta^{l+1}

对权重 :math:`w` 的求导为：

.. math::
	\frac{\partial C}{\partial w^l_{jk}} = \sum_i \frac{\partial C}{\partial z^{l+1}_i} \frac{\partial z^{l+1}_i}{\partial w^l_{jk}} = \frac{\partial C}{\partial z^{l+1}_j} a^l_k = \delta^{l+1}_j a^l_k

上式写成矩阵形式为：

.. math::
	\frac{\partial C}{\partial w^l} = \delta^{l+1} (a^l)^T

:math:`l` 层 :math:`\delta^l` 和 :math:`l+1` 层的 :math:`\delta^{l+1}` 的关系为：

.. math::
	\frac{\partial C}{\partial z^l_j} = \sum_{i,k} \frac{\partial C}{\partial z^{l+1}_i} \frac{\partial z^{l+1}_i}{\partial a^l_k} \frac{\partial a^l_k}{\partial z^l_j} = \sum_i \delta^{l+1}_i w^l_{ij} \sigma^{'}(z^l_j)

上式写成矩阵形式为：

.. math::
	\delta^l = (w^l)^T \delta^{l+1}\odot\sigma^{'}(z^l)

可以看出：

.. math::
	\nabla_a C = (w^l)^T \delta^{l+1}

BP算法总结
^^^^^^^^^^

BP算法可以概括为以下四个关系式：

.. math::	
	  \begin{aligned}
		\delta^l &= \frac{\partial C}{\partial z^l} = \nabla_z C   \\
		\frac{\partial C}{\partial w^l} &= \delta^{l+1} (a^l)^T    \\
		\frac{\partial C}{\partial b^l} &= \delta^{l+1}            \\
		\delta^l &= (w^l)^T \delta^{l+1}\odot\sigma^{'}(z^l)    
	  \end{aligned}	

可以看出，可以从 :math:`\delta^{l+1}` 的推导出对第 :math:`l` 层的权重和偏移量的偏导，以及第 :math:`l` 层的未激活前的神经元的偏导。
