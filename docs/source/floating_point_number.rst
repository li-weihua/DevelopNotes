*****************************
floating point number
*****************************

Microscaling Formats
=========================

.. list-table:: OCP 8-bit Floating Point Format
   :widths: 25 40 40
   :header-rows: 1

   * -
     - E4M3
     - E5M2
   * - Exponent bias
     - 7
     - 15
   * - Infinities
     - N/A
     - S 1111 00\ :sub:`2`
   * - NaN
     - S 1111 111\ :sub:`2`
     - S 11111 {01, 10, 11}\ :sub:`2`
   * - Zeros
     - S 0000 000\ :sub:`2`
     - S 00000 00\ :sub:`2`
   * - Max normal
     - S 1111 110\ :sub:`2` = ± 2\ :sup:`8` × 1.75 = ± 448
     - S 11110 11\ :sub:`2` = ± 2\ :sup:`15` × 1.75 = ± 57,344
   * - Min normal
     - S 0001 000\ :sub:`2` = ± 2\ :sup:`−6`
     - S 00001 00\ :sub:`2` = ± 2\ :sup:`−14`
   * - Max subnormal
     - S 0000 111\ :sub:`2` = ± 2\ :sup:`−6` × 0.875
     - S 00000 11\ :sub:`2` = ± 2\ :sup:`−14` × 0.75
   * - Min subnormal
     - S 0000 001\ :sub:`2` = ± 2\ :sup:`−9`
     - S 00000 01\ :sub:`2` = ± 2\ :sup:`−16`


.. list-table:: FP6 Format
   :widths: 25 40 40
   :header-rows: 1

   * -
     - E2M3
     - E3M2
   * - Exponent bias
     - 1
     - 3
   * - Infinities
     - N/A
     - N/A
   * - NaN
     - N/A
     - N/A
   * - Zeros
     - S 00 000\ :sub:`2`
     - S 000 00\ :sub:`2`
   * - Max normal
     - S 11 111\ :sub:`2` = ± 2\ :sup:`2` × 1.875 = ± 7.5
     - S 111 11\ :sub:`2` = ± 2\ :sup:`4` × 1.75 = ± 28
   * - Min normal
     - S 01 000\ :sub:`2` = ± 2\ :sup:`0` = ± 1
     - S 001 00\ :sub:`2` = ± 2\ :sup:`−2` = ± 0.25
   * - Max subnormal
     - S 00 111\ :sub:`2` = ± 2\ :sup:`0` × 0.875 = ± 0.875
     - S 000 11\ :sub:`2` = ± 2\ :sup:`−2` × 0.75 = ± 0.1875
   * - Min subnormal
     - S 00 001\ :sub:`2` = ± 2\ :sup:`−3` = ± 0.125
     - S 000 01\ :sub:`2` = ± 2\ :sup:`−4` = ± 0.0625


.. list-table:: FP4 Format
   :widths: 25 40
   :header-rows: 1

   * -
     - E2M1
   * - Exponent bias
     - 1
   * - Infinities
     - N/A
   * - NaN
     - N/A
   * - Zeros
     - S 00 0\ :sub:`2`
   * - Max normal
     - S 11 1\ :sub:`2` = ± 2\ :sup:`2` × 1.5 = ± 6
   * - Min normal
     - S 01 0\ :sub:`2` = ± 2\ :sup:`0` × 1.0 = ± 1
   * - Subnormal
     - S 00 1\ :sub:`2` = ± 2\ :sup:`0` × 0.5 = ± 0.5




IEEE 浮点数
==============

浮点数最新标准为IEEE 754-2019

浮点数格式如下：

+---------+---------------------+--------------------------------+
| S(sign) | E (biased exponent) | T (trailing significand field) |
+---------+---------------------+--------------------------------+
|  1 bit  |      w bits         |  t bits, t = p -1              |
+---------+---------------------+--------------------------------+

具有如下关系：

.. math::

  \begin{aligned}
       e & = E - bias \\
    e_{max} & = bias = 2^{w-1} - 1 \\
    e_{min} & = 1 - e_{max}
  \end{aligned}


关于biased E的说明:

1. normal number: [1 , :math:`2^w - 2`]，
   值为 :math:`(-1)^s \times 2^{E-bias} \times (1+ 2^{1-p} \times T)`
2. 0, 当T=0表示 :math:`\pm 0`; 当T!=0 表示 subnormal number,
   值为 :math:`(-1)^s \times 2^{e_{min}} \times (0+ 2^{1-p} \times T)`
3. :math:`2^w − 1` (二进制全部为1), 当T=0, 表示 :math:`\pm \infty`; 当T != 0, 表示 NaN.


ieee 规定的16, 32, 64, 128比特的浮点数格式列表
------------------------------------------------


+-----------+----------+----------+----------+-----------+
|  参数     | binary16 | binary32 | binary64 | binary128 |
+===========+==========+==========+==========+===========+
| 指数位数  |    5     |     8    |    11    |    15     |
+-----------+----------+----------+----------+-----------+
| emax/bias |   15     |    127   |   1023   |   16383   |
+-----------+----------+----------+----------+-----------+
| 小数位数  |   10     |    23    |    52    |    112    |
+-----------+----------+----------+----------+-----------+
