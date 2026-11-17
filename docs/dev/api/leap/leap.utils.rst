leap.utils module
=================

``poly`` function
*****************

The basic usage of the ``poly`` function is quite simple. We take some vector of length ``n``,
say ``n=4``:

.. math::

   x = \begin{bmatrix}
   1 \\
   2 \\
   3 \\
   4
   \end{bmatrix}


and ask to return a matrix where each column is the input vector raised to the corresponding power
up to the specified degree, say ``degree=3``:

.. math::

   \text{poly}(x, \text{degree}=3) = 
   \begin{bmatrix}
   1^1 & 1^2 & 1^3 \\
   2^1 & 2^2 & 2^3 \\
   3^1 & 3^2 & 3^3 \\
   4^1 & 4^2 & 4^3 \\
   \end{bmatrix} =
   \begin{bmatrix}
   1 & 1 & 1 \\
   2 & 4 & 8 \\
   3 & 9 & 27 \\
   4 & 16 & 64
   \end{bmatrix}


.. code-block:: python

   poly([1, 2, 3, 4], degree=3)
   >>> array([[1., 1., 1.],
   ...        [2., 4., 8.],
   ...        [3., 9., 27.],
   ...        [4., 16., 64.]])


Orthogonalization
------------------

If we want an orthogonal matrix, we use the ``Gram-Schmidt`` process. Let's say we have a vector:

.. math::

   \vec{x} = \begin{bmatrix}
   -3 \\
   0 \\
   4 \\
   3
   \end{bmatrix}
     


We subtract the mean of the vector, :math:`\bar{x} = 1`, from each element to center the vector
around zero:

.. math::

   \vec{x} - \bar{x} = \begin{bmatrix}
   -3 \\
   0 \\
   4 \\
   3
   \end{bmatrix} - 1
   = \begin{bmatrix}
   -4 \\
   -1 \\
   3 \\
   2
   \end{bmatrix}


Let us consider the Vandermonde matrix:

.. math::

   X = \begin{bmatrix}
   1 & -4 & 16 & -64 \\
   1 & -1 & 1 & -1 \\
   1 & 3 & 9 & 27 \\
   1 & 2 & 4 & 8
   \end{bmatrix} =
   \begin{bmatrix}
   \vec{x}_1 & \vec{x}_2 & \vec{x}_3 & \vec{x}_4
   \end{bmatrix}

To obtain an orthonormal matrix :math:`Q`, we apply the Gram-Schmidt process to the columns of
:math:`X`:

.. math::


   q_i = \dfrac{x_i - \sum_{j=1}^{i-1} <x_i, q_j> q_j}
      {\sqrt{
      <x_i - \sum_{j=1}^{i-1} <x_i, q_j> q_j,
      x_i - \sum_{j=1}^{i-1}<x_i, q_j> q_j>}}


So we have:

.. math::

   q_1 &= \dfrac{\vec{x}_1}{\sqrt{\vec{x}_1 \cdot \vec{x}_1}} 
   = \dfrac{1}{2} \begin{bmatrix}
      1 \\
      1 \\
      1 \\
      1
   \end{bmatrix}


  
.. info:: Math: :math:`q_1`
  :collapsible:

  .. math::

      q_1 &= \dfrac{\vec{x}_1}{\sqrt{\vec{x}_1 \cdot \vec{x}_1}} 
      = \begin{bmatrix}
      1 \\
      1 \\
      1 \\
      1
      \end{bmatrix} \dfrac{1}{\left(\begin{bmatrix}
      1 & 1 & 1 & 1
      \end{bmatrix} \cdot \begin{bmatrix}
      1 \\
      1 \\
      1 \\
      1
      \end{bmatrix}\right)^{1/2}} 
      = \begin{bmatrix}
      1 \\
      1 \\
      1 \\
      1
      \end{bmatrix} \dfrac{1}{\sqrt{1^2 + 1^2 + 1^2 + 1^2}}
      = \dfrac{1}{2} \begin{bmatrix}
      1 \\
      1 \\
      1 \\
      1
      \end{bmatrix}


.. math::

   q_2 &= \dfrac{\vec{x}_2 -  (\vec{x}_2 \cdot \vec{q}_1) \vec{q}_1}
         {\sqrt{
         (\vec{x}_2 - (\vec{x}_2 \cdot \vec{q}_1) \vec{q}_1) \cdot
         (x_2 - (\vec{x}_2 \cdot \vec{q}_1) \vec{q}_1)}} \\ 
   &= \dfrac{1}{\sqrt{30}} \begin{bmatrix}
      -4 \\
      -1 \\
      3 \\
      2
   \end{bmatrix}


  
.. info:: Math: :math:`q_2`
  :collapsible:

  First let us calculate :math:`\vec{x}_2 \cdot \vec{q}_1`:

  .. math::

      \vec{x}_2 \cdot \vec{q}_1 = \begin{bmatrix}
      -4 & -1 & 3 & 2
      \end{bmatrix} \cdot \dfrac{1}{2}
      \begin{bmatrix}
      1 \\
      1 \\
      1 \\
      1
      \end{bmatrix}
      = \dfrac{1}{2} (-4*1 - 1*1 + 3*1 + 2*1)
      = 0

   
  Next, we calculate :math:`\vec{x}_2 - (\vec{x}_2 \cdot \vec{q}_1) \vec{q}_1`:

  .. math::

      \vec{x}_2 - (\vec{x}_2 \cdot \vec{q}_1) \vec{q}_1
      = \begin{bmatrix}
      -4 \\
      -1 \\
      3 \\
      2
      \end{bmatrix} - 0
      = \begin{bmatrix}
      -4 \\
      -1 \\
      3 \\
      2
      \end{bmatrix}

  Finally, we calculate :math:`q_2` 

  .. math::

      q_2 &= \dfrac{1}{\sqrt{(-4)^2+ (-1)^2+ 3^2 + 2^2}}\begin{bmatrix}
      -4 \\
      -1 \\
      3 \\
      2
      \end{bmatrix} \\
      &= \dfrac{1}{\sqrt{30}}\begin{bmatrix}
      -4 \\
      -1 \\
      3 \\
      2
      \end{bmatrix}


.. math::

   q_3 = \dfrac{\vec{x}_3 - \sum_{j=1}^{2} (\vec{x}_3 \cdot \vec{q}_j) \vec{q}_j}
      {\sqrt{
      (\vec{x}_3 - \sum_{j=1}^{2} (\vec{x}_3 \cdot \vec{q}_j) q_j) \cdot
      (\vec{x}_3 - \sum_{j=1}^{2} (\vec{x}_3 \cdot \vec{q}_j) q_j)}}
   = \dfrac{1}{2\sqrt{11}} \begin{bmatrix}
      3 \\
      -5 \\
      3 \\
      -1
      \end{bmatrix}



.. info:: Math: :math:`q_3`
  :collapsible:

  First let us calculate :math:`\vec{x}_3 \cdot \vec{q}_1`:

  .. math::

      \vec{x}_3 \cdot \vec{q}_1 = \begin{bmatrix}
      16 & 1 & 9 & 4
      \end{bmatrix} \cdot \dfrac{1}{2} \begin{bmatrix}
      1 \\
      1 \\
      1 \\
      1
      \end{bmatrix}
      = \dfrac{1}{2} (16* 1 + 1* 1 + 9*1 + 4*1)
      = \dfrac{1}{2} (16 + 1 + 9 + 4)
      = 15

  Next, let us calculate :math:`\vec{x}_3 \cdot \vec{q}_2`:

  .. math::

      \vec{x}_3 \cdot \vec{q}_2 = \begin{bmatrix}
      16 & 1 & 9 & 4
      \end{bmatrix} \cdot \dfrac{1}{\sqrt{30}} \begin{bmatrix}
      -4 \\
      -1 \\
      3 \\
      2
      \end{bmatrix}
      = \dfrac{-16*4 - 1*1 + 9*3 + 4*2}{\sqrt{30}}
      = \dfrac{-64 - 1 + 27 + 8}{\sqrt{30}}
      = \dfrac{-30}{\sqrt{30}}
      = -\sqrt{30}

  Then we have:

  .. math::

      \vec{x}_3 - \sum_{j=1}^{2} (\vec{x}_3 \cdot \vec{q}_j) \vec{q}_j
      &= \vec{x}_3 - (\vec{x}_3 \cdot \vec{q}_1) \vec{q}_1 - (\vec{x}_3 \cdot \vec{q}_2) \vec{q}_2 \\
      &= \begin{bmatrix}
      16 \\
      1 \\
      9 \\
      4
      \end{bmatrix} - 
      \dfrac{15}{2} \begin{bmatrix}
      1 \\
      1 \\
      1 \\
      1
      \end{bmatrix} + 
      \dfrac{\sqrt{30}}{\sqrt{30}} \begin{bmatrix}
      -4 \\
      -1 \\
      3 \\
      2
      \end{bmatrix} \\
      &= \dfrac{1}{2} \begin{bmatrix}
      32 \\
      2 \\
      18 \\
      8
      \end{bmatrix}
      - \dfrac{1}{2} \begin{bmatrix}
      15 \\
      15 \\
      15 \\
      15
      \end{bmatrix} +
      \dfrac{1}{2}\begin{bmatrix}
      -8 \\
      -2 \\
      6 \\
      4
      \end{bmatrix} \\
      &= \dfrac{3}{2} \begin{bmatrix}
      3 \\
      -5\\
      3 \\
      -1
      \end{bmatrix}

  Then we have:

  .. math::

      &\dfrac{\vec{x}_3 - \sum_{j=1}^{2} (\vec{x}_3 \cdot \vec{q}_j) \vec{q}_j}
      {\sqrt{
      (\vec{x}_3 - \sum_{j=1}^{2} (\vec{x}_3 \cdot \vec{q}_j) q_j) \cdot
      (\vec{x}_3 - \sum_{j=1}^{2} (\vec{x}_3 \cdot \vec{q}_j) q_j)}} \\
      &= \dfrac{3}{2}\begin{bmatrix}
      3 \\
      -5 \\
      3 \\
      -1
      \end{bmatrix} * \dfrac{1}{\sqrt{
      \dfrac{3^2}{2^2} \begin{bmatrix}
      3 & -5 & 3 & -1
      \end{bmatrix} \cdot \begin{bmatrix}
      3 \\
      -5 \\
      3 \\
      -1  
      \end{bmatrix} }} \\
      &= \dfrac{3}{2}\begin{bmatrix}
      3 \\
      -5 \\
      3 \\
      -1
      \end{bmatrix} * \dfrac{2}{3\sqrt{
      3^2 + (-5)^2 + 3^2 + (-1)^2}} \\
      &= \begin{bmatrix}
      3 \\
      -5 \\
      3 \\
      -1
      \end{bmatrix} * \dfrac{1}{\sqrt{44}} \\
      &= \dfrac{1}{2\sqrt{11}} \begin{bmatrix}
      3 \\
      -5 \\
      3 \\
      -1
      \end{bmatrix}



.. math::

   q_4 = \dfrac{\vec{x}_4 - \sum_{j=1}^{3} (\vec{x}_4 \cdot \vec{q}_j) \vec{q}_j}
      {\sqrt{
      (\vec{x}_4 - \sum_{j=1}^{3} (\vec{x}_4 \cdot \vec{q}_j) q_j) \cdot
      (\vec{x}_4 - \sum_{j=1}^{3} (\vec{x}_4 \cdot \vec{q}_j) q_j)}}
   = \dfrac{1}{\sqrt{330}}  \begin{bmatrix}
      -2 \\
      7 \\
      9 \\
      -14
      \end{bmatrix} 


.. info:: Math: :math:`q_4`
  :collapsible:

  First let us calculate :math:`\vec{x}_4 \cdot \vec{q}_1`:

  .. math::

      \vec{x}_4 \cdot \vec{q}_1 = \begin{bmatrix}
      -64 & -1 & 27 & 8
      \end{bmatrix} \cdot \dfrac{1}{2} \begin{bmatrix}
      1 \\
      1 \\
      1 \\
      1
      \end{bmatrix}
      = \dfrac{1}{2} (-64* (1) - 1* (1) + 27*1 + 8*1)
      = \dfrac{1}{2} (-64 - 1 + 27 + 8)
      = -15

  Next, let us calculate :math:`\vec{x}_4 \cdot \vec{q}_2`:

  .. math::

      \vec{x}_4 \cdot \vec{q}_2 = \begin{bmatrix}
      -64 & -1 & 27 & 8
      \end{bmatrix} \cdot \dfrac{1}{\sqrt{30}} \begin{bmatrix}
      -4 \\
      -1 \\
      3 \\
      2
      \end{bmatrix}
      = \dfrac{-64*(-4) - 1*(-1) + 27*3 + 8*2}{\sqrt{30}}
      = \dfrac{256 + 1 + 81 + 16}{\sqrt{30}}
      = \dfrac{354}{\sqrt{30}}

  Finally, let us calculate :math:`\vec{x}_4 \cdot \vec{q}_3`:

  .. math::

      \vec{x}_4 \cdot \vec{q}_3 = \begin{bmatrix}
      -64 & -1 & 27 & 8
      \end{bmatrix} \cdot \dfrac{1}{2\sqrt{11}} \begin{bmatrix}
      3 \\
      -5 \\
      3 \\
      -1
      \end{bmatrix}
      = \dfrac{-64*3 - 1*(-5) + 27*(3) + 8*(-1)}{2 \sqrt{11}}
      = \dfrac{-192 + 5 + 81 -8}{2 \sqrt{11}}
      = \dfrac{-57}{\sqrt{11}}

  Then we have:

  .. math::

      \vec{x}_4 - \sum_{j=1}^{3} (\vec{x}_4 \cdot \vec{q}_j) \vec{q}_j
      &= \vec{x}_4 - (\vec{x}_4 \cdot \vec{q}_1) \vec{q}_1 - (\vec{x}_4 \cdot \vec{q}_2) \vec{q}_2 
      - (\vec{x}_4 \cdot \vec{q}_3) \vec{q}_3\\
      &= \begin{bmatrix}
      -64 \\
      -1 \\
      27 \\
      8
      \end{bmatrix} + \dfrac{15}{2} \begin{bmatrix}
      1 \\
      1 \\
      1 \\
      1
      \end{bmatrix} -
      \dfrac{354}{\sqrt{30}} \dfrac{1}{\sqrt{30}} \begin{bmatrix}
      -4 \\
      -1 \\
      3 \\
      2
      \end{bmatrix} +\dfrac{57}{\sqrt{11}} \dfrac{1}{2\sqrt{11}} \begin{bmatrix}
      3 \\
      -5 \\
      3 \\
      -1
      \end{bmatrix} \\
      &= \begin{bmatrix}
      -64 \\
      -1 \\
      27 \\
      8
      \end{bmatrix} + \dfrac{15}{2} \begin{bmatrix}
      1 \\
      1 \\
      1 \\
      1
      \end{bmatrix} -
      \dfrac{59}{5} \begin{bmatrix}
      -4 \\
      -1 \\
      3 \\
      2
      \end{bmatrix} +\dfrac{57}{22} \begin{bmatrix}
      3 \\
      -5 \\
      3 \\
      -1
      \end{bmatrix} \\
      &=  \dfrac{1}{110}\begin{bmatrix}
      -64*110 + 15 * 55 + 4 * 22 * 59 + 57*3 * 5 \\
      -110 + 15 * 55 + 22 * 59 - 57*5 * 5 \\
      27 * 110 + 15 * 55 - 3 * 22 * 59 + 57*3 * 5 \\
      8 * 110 + 15 * 55 - 2 * 22 * 59 - 57 * 5 \\
      \end{bmatrix} \\
      &=  \dfrac{42}{55}\begin{bmatrix}
      -2 \\
      7 \\
      9 \\
      -14 \\
      \end{bmatrix} 

  Then we have:

  .. math::

      &\dfrac{\vec{x}_4 - \sum_{j=1}^{3} (\vec{x}_4 \cdot \vec{q}_j) \vec{q}_j}
      {\sqrt{
      (\vec{x}_4 - \sum_{j=1}^{3} (\vec{x}_4 \cdot \vec{q}_j) q_j) \cdot
      (\vec{x}_4 - \sum_{j=1}^{3} (\vec{x}_4 \cdot \vec{q}_j) q_j)}} \\
      &= \dfrac{42}{55}\begin{bmatrix}
      -2 \\
      7 \\
      9 \\
      -14
      \end{bmatrix} * \dfrac{1}{\sqrt{
      \dfrac{42^2}{55^2} \begin{bmatrix}
      -2 & 7 & 9 & -14
      \end{bmatrix} \cdot \begin{bmatrix}
      -2 \\
      7 \\
      9 \\
      -14
      \end{bmatrix} }} \\
      &= \dfrac{42}{55} \begin{bmatrix}
      -2 \\
      7 \\
      9 \\
      -14
      \end{bmatrix} * \dfrac{55}{42\sqrt{
      (-2)^2 + 7^2 + 9^2 + (-14)^2}} \\
      &= \begin{bmatrix}
      -2 \\
      7 \\
      9 \\
      -14
      \end{bmatrix}  * \dfrac{1}{\sqrt{
      4 + 49 + 81 + 196}} \\
      &= \dfrac{1}{\sqrt{330}}  \begin{bmatrix}
      -2 \\
      7 \\
      9 \\
      -14
      \end{bmatrix} 

So we have our orthonormal matrix :math:`Q`:

.. math::

    Q = \begin{bmatrix}
    \vec{q}_1 & \vec{q}_2 & \vec{q}_3 & \vec{q}_4
    \end{bmatrix} 
    = \begin{bmatrix}
    \dfrac{1}{2} & \dfrac{-4}{\sqrt{30}} & \dfrac{3}{2 \sqrt{11}} & \dfrac{-2}{\sqrt{330}} \\
    \dfrac{1}{2} & \dfrac{-1}{\sqrt{30}} & \dfrac{-5}{2 \sqrt{11}} & \dfrac{7}{\sqrt{330}} \\
    \dfrac{1}{2} & \dfrac{3}{\sqrt{30}} & \dfrac{3}{2 \sqrt{11}} & \dfrac{9}{\sqrt{330}} \\
    \dfrac{1}{2} & \dfrac{2}{\sqrt{30}} & \dfrac{-1}{2 \sqrt{11}} & \dfrac{-14}{\sqrt{330}}
    \end{bmatrix}


To obtain :math:`R`, we have:

.. math::

    X = QR \implies R = Q^{-1} X = Q^T X 


.. math::

   Q^T &= \begin{bmatrix}
    \dfrac{1}{2}          & \dfrac{1}{2}          & \dfrac{1}{2}          & \dfrac{1}{2} \\
    \dfrac{-4}{\sqrt{30}}  & \dfrac{-1}{\sqrt{30}}  & \dfrac{3}{\sqrt{30}}  & \dfrac{2}{\sqrt{30}} \\
    \dfrac{3}{2 \sqrt{11}} & \dfrac{-5}{2 \sqrt{11}}         & \dfrac{3}{2 \sqrt{11}} &  \dfrac{-1}{2 \sqrt{11}}\\
    \dfrac{-2}{\sqrt{330}} & \dfrac{7}{\sqrt{330}} & \dfrac{9}{\sqrt{330}}  & \dfrac{-14}{\sqrt{330}}
    \end{bmatrix}

.. math::

   R = Q^T X 
   = \begin{bmatrix}
      2          & 0           & 15             & -15 \\
      0 & \dfrac{30}{\sqrt{30}}   & \dfrac{-30}{\sqrt{30}}    & \dfrac{354}{\sqrt{30}} \\
      0 & 0 & \dfrac{33}{\sqrt{11}}  & \dfrac{-57}{\sqrt{11}} \\
      0 & 0 & 0 & \dfrac{252}{\sqrt{330}}
      \end{bmatrix} \\

.. info:: Math: :math:`R=Q^T X`
  :collapsible:

  .. math::

      R &= Q^T X \\
      &= \begin{bmatrix}
         \dfrac{1}{2}           & \dfrac{1}{2}            & \dfrac{1}{2}           & \dfrac{1}{2} \\
         \dfrac{-4}{\sqrt{30}}  & \dfrac{-1}{\sqrt{30}}   & \dfrac{3}{\sqrt{30}}   & \dfrac{2}{\sqrt{30}} \\
         \dfrac{3}{2 \sqrt{11}} & \dfrac{-5}{2 \sqrt{11}} & \dfrac{3}{2 \sqrt{11}} &  \dfrac{-1}{2 \sqrt{11}}\\
         \dfrac{-2}{\sqrt{330}} & \dfrac{7}{\sqrt{330}}   & \dfrac{9}{\sqrt{330}}  & \dfrac{-14}{\sqrt{330}}
         \end{bmatrix} \begin{bmatrix}
      1 & -4 & 16 & -64 \\
      1 & -1 & 1 & -1 \\
      1 & 3 & 9 & 27 \\
      1 & 2 & 4 & 8
      \end{bmatrix} \\
      &= \begin{bmatrix}
      \dfrac{1}{2}(1 + 1 + 1 + 1)          & \dfrac{1}{2}(-4 -1  + 3 + 2)             & \dfrac{1}{2}(16 + 1 + 9 + 4)              & \dfrac{1}{2}(-64 -1 + 27 + 8) \\
      \dfrac{1}{\sqrt{30}}(-4 -1  + 3 + 2) & \dfrac{1}{\sqrt{30}}(16 +1 + 9 + 4)      & \dfrac{1}{\sqrt{30}}(-64 -1 + 27 + 8)     & \dfrac{1}{\sqrt{30}}(256 + 1 + 81 + 16) \\
      \dfrac{1}{2 \sqrt{11}}(3 - 5 +3 - 1) & \dfrac{1}{2 \sqrt{11}}(-12 + 5 + 9 - 2) & \dfrac{1}{2 \sqrt{11}}(48 - 5 + 27 - 4)   & \dfrac{1}{2 \sqrt{11}}(-192 + 5 +81 - 8) \\
      \dfrac{1}{\sqrt{330}}(-2+7+9-14)     & \dfrac{1}{\sqrt{330}}(8 -7 +27 -28)  & \dfrac{1}{\sqrt{330}}(-32 + 7 + 81 - 56) & \dfrac{1}{\sqrt{330}}(128 -7 +243 -112)
      \end{bmatrix} \\
      &= \begin{bmatrix}
      2          & 0           & 15             & -15 \\
      0 & \dfrac{30}{\sqrt{30}}   & \dfrac{-30}{\sqrt{30}}    & \dfrac{354}{\sqrt{30}} \\
      0 & 0 & \dfrac{33}{\sqrt{11}}  & \dfrac{-57}{\sqrt{11}} \\
      0 & 0 & 0 & \dfrac{252}{\sqrt{330}}
      \end{bmatrix}


Now we have computed the :math:`QR` decomposition of :math:`X`. Now, the original vector
:math:`\vec{x}` represents :math:`n` data points. For example, suppose we have 4 data points,
and the value we are looking at here is ``age``. If we are training a GLM, we may want to get
predictions on 10 new data points - but how does that work if we have a 4-dimensional
:math:`QR` decomposition?

Let's rewrite these matrices in terms of polynomials:

.. math::

    Z_{i\kappa} &= P_{\kappa - 1}(x_i) \\
    P_{\kappa}(x_i) &= (x_i - \bar{x})^{\kappa} - \sum_{j=0}^{\kappa-1} c_{\kappa j} P_j(x_i) \\
    c_{\kappa j} &= \dfrac{\langle (x - \bar{x})^{\kappa}, P_j(x) \rangle}{\langle P_j(x), P_j(x) \rangle} \\
    &= \dfrac{R_{j+1, \kappa+1}}{R_{j+1, j+1}}


So we have :math:`Z` as a matrix of orthogonal polynomials evaluated at the 4 data points:


.. math::

   Z = \begin{bmatrix}
    P_0(x_1) & P_1(x_1) & P_2(x_1) & P_3(x_1) \\
    P_0(x_2) & P_1(x_2) & P_2(x_2) & P_3(x_2) \\
    P_0(x_3) & P_1(x_3) & P_2(x_3) & P_3(x_3) \\
    P_0(x_4) & P_1(x_4) & P_2(x_4) & P_3(x_4)
    \end{bmatrix}


but we would like to evaluate these polynomials at new data points as well. For that, we will need
the following theorem:

.. theorem::

     Let :math:`P_n(x)` be a polynomial of degree :math:`n` in :math:`x`:

     .. math::

        P_n(x) = a_0 + a_1 x + a_2 x^2 + \dots + a_n x^n

     such that :math:`a_n \neq 0`, and let :math:`P_n(x)` be an orthogonal polynomial, i.e.

     .. math::

        \langle P_n(x), P_m(x) \rangle = 0, \quad \text{for } n \neq m

     Then the set of polynomials :math:`\{P_k(x)\}` satisfy the 3-term recurrence relation:

     .. math::

        P_{k+1}(x) = (x - \alpha_k) P_k(x) - \beta_k P_{k-1}(x), \quad k \ge 1

     with initial conditions :math:`P_0(x) = 1` and :math:`P_{-1}(x) = 0`.


So, if we can find :math:`\alpha` and :math:`\beta`, we can calculate these polynomials for
any new data points.
Next, we take just the diagonal entries of :math:`R` to form the matrix :math:`D`:


.. math::

    D = \begin{bmatrix}
      2          & 0           & 0             & 0 \\
      0 & \dfrac{30}{\sqrt{30}}   & 0   & 0 \\
      0 & 0 & \dfrac{33}{\sqrt{11}}  & 0 \\
      0 & 0 & 0 & \dfrac{252}{\sqrt{330}}
      \end{bmatrix} 


Then, we multiply :math:`Q` by :math:`D` to form the matrix :math:`P`:

.. math::

    P = Q D
    &= \begin{bmatrix}
    \dfrac{1}{2} & \dfrac{-4}{\sqrt{30}} & \dfrac{3}{2 \sqrt{11}} & \dfrac{-2}{\sqrt{330}} \\
    \dfrac{1}{2} & \dfrac{-1}{\sqrt{30}} & \dfrac{-5}{2 \sqrt{11}} & \dfrac{7}{\sqrt{330}} \\
    \dfrac{1}{2} & \dfrac{3}{\sqrt{30}} & \dfrac{3}{2 \sqrt{11}} & \dfrac{9}{\sqrt{330}} \\
    \dfrac{1}{2} & \dfrac{2}{\sqrt{30}} & \dfrac{-1}{2 \sqrt{11}} & \dfrac{-14}{\sqrt{330}}
    \end{bmatrix} \begin{bmatrix}
      2          & 0           & 0             & 0 \\
      0 & \dfrac{30}{\sqrt{30}}   & 0   & 0 \\
      0 & 0 & \dfrac{33}{\sqrt{11}}  & 0 \\
      0 & 0 & 0 & \dfrac{252}{\sqrt{330}}
      \end{bmatrix} \\
    &= \begin{bmatrix}
    1 & -4 & \dfrac{9}{2} &  \dfrac{-84}{55} \\
    1 & -1 & \dfrac{-15}{2}& \dfrac{294}{55} \\
    1 & 3 & \dfrac{9}{2}& \dfrac{378}{55}\\
    1 & 2 & \dfrac{-3}{2} & \dfrac{-588}{55}
    \end{bmatrix} 


If we think of :math:`P` as a matrix of column vectors:

.. math::

    P = \begin{bmatrix}
    \mathbf{p}_1 & \mathbf{p}_2 & \mathbf{p}_3 & \mathbf{p}_4
    \end{bmatrix}

   
Then we can normalize these column vectors by dividing each column by its norm:

.. math::

    \eta := \begin{bmatrix}
    |\mathbf{p}_1|^2 & |\mathbf{p}_2|^2 & |\mathbf{p}_3|^2 & |\mathbf{p}_4|^2
    \end{bmatrix}
    = \begin{bmatrix}
    4 & 30 & 99 & \dfrac{10584}{55}
    \end{bmatrix}


.. math::

    Z = \begin{bmatrix}
    \dfrac{\mathbf{p}_1}{|\mathbf{p}_1|} & \dfrac{\mathbf{p}_2}{|\mathbf{p}_2|} 
    & \dfrac{\mathbf{p}_3}{|\mathbf{p}_3|} & \dfrac{\mathbf{p}_4}{|\mathbf{p}_4|}
    \end{bmatrix}
    = \begin{bmatrix}
    \dfrac{1}{2} & -\dfrac{4}{\sqrt{30}} & \dfrac{9}{2\sqrt{99}} & \dfrac{-21}{\sqrt{55*2646}}  \\
    \dfrac{1}{2} & -\dfrac{1}{\sqrt{30}} & \dfrac{-15}{2\sqrt{99}} & \dfrac{294 }{\sqrt{55*10584}} \\
    \dfrac{1}{2} & \dfrac{3}{\sqrt{30}} & \dfrac{9}{2 \sqrt{99}} & \dfrac{378}{\sqrt{55*10584}}\\
    \dfrac{1}{2} & \dfrac{2}{\sqrt{30}} & \dfrac{-3}{2\sqrt{99}} &  \dfrac{-147}{\sqrt{55*2646}}
    \end{bmatrix}

We obtain :math:`\alpha` by:


.. math::

   S = [p_{ij}^2] = \begin{bmatrix}
    1 & 16 & \dfrac{81}{4} &  \dfrac{84^2}{55^2} \\
    1 & 1 & \dfrac{225}{4}& \dfrac{294^2}{55^2} \\
    1 & 9 & \dfrac{81}{4}& \dfrac{378^2}{55^2}\\
    1 & 4 & \dfrac{9}{4} & \dfrac{588^2}{55^2}
   \end{bmatrix} 


.. math::

   S^T \vec{x} = \begin{bmatrix}
   1 & 1 & 1 & 1 \\
   16 & 1 & 9 & 4 \\
   \dfrac{81}{4} & \dfrac{225}{4} & \dfrac{81}{4} & \dfrac{9}{4} \\
   \dfrac{84^2}{55^2} & \dfrac{294^2}{55^2} & \dfrac{378^2}{55^2} & \dfrac{588^2}{55^2} 
   \end{bmatrix} \begin{bmatrix}
   -4 \\
   -1 \\
   3 \\
   2
   \end{bmatrix}
   = \begin{bmatrix}
   0 \\
   -30 \\
   -72 \\
   \dfrac{201096}{605}
   \end{bmatrix}


.. info:: Math: :math:`S^T \vec{x}`
  :collapsible:

  .. math::

   S^T \vec{x} &= \begin{bmatrix}
   1 & 1 & 1 & 1 \\
   16 & 1 & 9 & 4 \\
   \dfrac{81}{4} & \dfrac{225}{4} & \dfrac{81}{4} & \dfrac{9}{4} \\
   \dfrac{84^2}{55^2} & \dfrac{294^2}{55^2} & \dfrac{378^2}{55^2} & \dfrac{588^2}{55^2} 
   \end{bmatrix} \begin{bmatrix}
   -4 \\
   -1 \\
   3 \\
   2
   \end{bmatrix} \\
   &= \begin{bmatrix}
   0 \\
   -30 \\
   \dfrac{-4*81 - 225 + 3*81 + 2*9}{4} \\
   -\dfrac{4* 84^2 - 294^2 + 3*378^2 + 2*588^2}{55^2}
   \end{bmatrix} \\
   &= \begin{bmatrix}
   0 \\
   -30 \\
   \dfrac{-288}{4} \\
   \dfrac{201096 * 5}{5*5*11*11}
   \end{bmatrix} \\
   &= \begin{bmatrix}
   0 \\
   -30 \\
   -72 \\
   \dfrac{201096}{605}
   \end{bmatrix}


.. math::

   \left[\dfrac{[S^T \vec{x}]_{i}}{\eta_i}\right] = \begin{bmatrix}
   \dfrac{0}{4}\\
   -\dfrac{30}{30} \\
   -\dfrac{72}{99} \\
   \dfrac{201096}{55*11}\cdot\dfrac{55}{10584}
   \end{bmatrix}
   = \begin{bmatrix}
      0 \\
      -1 \\
      -\dfrac{8}{11} \\
      \dfrac{19}{11}
   \end{bmatrix}


.. math::

   \left[\dfrac{[S^T \vec{x}]_{i}}{\eta_i}\right] + \bar{x}
   = \begin{bmatrix}
      0 \\
      -1 \\
      -\dfrac{8}{11} \\
      \dfrac{19}{11}
   \end{bmatrix} + \begin{bmatrix}
      1 \\
      1 \\
      1 \\
      1
   \end{bmatrix}
   = \begin{bmatrix}
      1 \\
      0 \\
      \dfrac{3}{11} \\
      \dfrac{30}{11}
   \end{bmatrix}


Finally, we take the first 3 elements of the resulting vector to obtain :math:`\alpha`:

.. math::

   \alpha = \begin{bmatrix}
      1 \\
      0 \\
      \dfrac{3}{11}
   \end{bmatrix}


So, now suppose we have a new vector, :math:`\vec{y}`, which has :math:`6` elements.
We have:

.. math::

   P_0(y_k) &= 1 \\
   P_1(y_k) &= y_k - \alpha_0 \quad k \in \{1, 2, 3, 4, 5, 6\} \\
   P_{\mu}(y_k) &= (y_k - \alpha_{\mu-1}) P_{\mu-1}(y_k) - \beta_{\mu-1} P_{\mu-2}(y_k) \quad k \in \{1, 2, 3, 4, 5, 6\}, \quad \mu \ge 2

.. automodule:: leap.utils
   :members:
   :undoc-members:
   :show-inheritance:
