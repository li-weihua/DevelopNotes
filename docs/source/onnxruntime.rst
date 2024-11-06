*************
onnnxruntime
*************

Pre-build packages
====================

PC or server
-------------
https://github.com/microsoft/onnxruntime/releases

Android
---------
https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-android

iOS
-----
https://onnxruntime.ai/docs/install/#install-on-ios

``Podfile`` add

.. code::

  use_frameworks!

  pod 'onnxruntime-c'


Another ref: https://github.com/CocoaPods/Specs/tree/master/Specs/3/a/a/onnxruntime-c

e.g. 1.20.0: https://download.onnxruntime.ai/pod-archive-onnxruntime-c-1.20.0.zip


MLAS
=======

MLAS is a core library of onnxruntime.

Pre-pack weight matrix
------------------------

约定权重矩阵为B，大小为：K * N，
packed-B的大小：

.. math::
    \begin{aligned}
    \textrm{AlignedN} &= (N + 16- 1) / 16 \times 16 \\
    K &= 256 \times K_1 + K_2, \; K_2 \in [0, 256)
    \end{aligned}

内存排布:

#. 沿N方向排布16个数
#. 沿K方向排布256个数，最后不够的不补零
#. goto步骤1，直到AlignedN排布完，最后不够16的补零
#. 如果K方向还有剩余，goto步骤1，直到把K循环完
