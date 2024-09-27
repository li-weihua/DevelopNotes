convolution arithmetic
======================

:reference:
  - https://github.com/vdumoulin/conv_arithmetic
  - https://arxiv.org/abs/1603.07285

1. convolution
--------------
Set input data size :math:`i`, convolution kernel size :math:`k`, stride size :math:`s`, and zero padding size :math:`p`.
Then the output size :math:`o` is:

.. math::
  o = \left\lfloor{\frac{i + 2p - k}{s}}\right\rfloor + 1 \,.
  :label: conv

The floor function :math:`{\lfloor}\,{\rfloor}` can found at https://en.wikipedia.org/wiki/Floor_and_ceiling_functions.

2. pooling
----------
According to :eq:`conv`, pooling output size is:

.. math::
  o = \left\lfloor{\frac{i-k}{s}}\right\rfloor + 1 \,.
  :label: pooling

3. tansposed convolution
------------------------

:explanation:
  The convolution operation can be rewritten to matrix multiplication.


4. dilated convolution
-----------------------
The dilation "rate" is controlled by an additional hyperparameter :math:`d`. A kernel
of size k dilated by a factor d has an effective size:

.. math::
  \hat{k} = k + (k-1)(d-1) \,.

Combined with :eq:`conv` the output size is:

.. math::
  o = \left\lfloor{\frac{i + 2p - k - (k-1)(d-1)}{s}}\right\rfloor + 1 \,.
  :label: dilatedconv


