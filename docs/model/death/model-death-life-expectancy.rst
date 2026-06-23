============================================
Appendix: Life Expectancy
============================================

Life expectancy is computed from a projected life table using the following standard
actuarial definitions. It is used during calibration to evaluate how well a candidate
:math:`\beta_{\text{sex}}` reproduces Statistics Canada's targets.

We first define some variables:

.. list-table::
   :widths: 15 15 80
   :header-rows: 1

   * - Variable
     - Type
     - Definition
   * - :math:`x`
     - :code:`int`
     - age in years
   * - :math:`l(x)`
     - :code:`int`
     - the number of people alive up to age :math:`x`
   * - :math:`q(x)`
     - :code:`float`
     - the probability of death between ages :math:`[x, x + \Delta x)`
   * - :math:`d(x)`
     - :code:`int`
     - the number of deaths between ages :math:`[x, x + \Delta x)`
   * - :math:`a(x)`
     - :code:`float`
     - the average fraction of the interval :math:`[x, x + \Delta x)` lived by those who die in the interval
   * - :math:`L(x)`
     - :code:`int`
     - the area under the survival curve between ages :math:`[x, x + \Delta x)`
   * - :math:`T(x)`
     - :code:`int`
     - the total number of person-years lived after age :math:`x`
   * - :math:`E(x)`
     - :code:`float`
     - the number of years left to live at age :math:`x`
   * - :math:`E(0)`
     - :code:`float`
     - the number of years left to live at age :math:`0`, i.e. ``life expectancy``


Number of People Alive
======================

First, we set the number of people alive at age :math:`0`:

.. math::

  l(0) := 100000

Next, we calculate the number of people alive up to age :math:`x`:

.. math::

  l(x) = l(x-\Delta x) \cdot (1 - q(x-\Delta x))

Total Deaths
=============

The number of deaths :math:`d(x)` between ages :math:`[x, x + \Delta x)`, is given by the number of
people alive at age :math:`x` multiplied by the probability of death between
ages :math:`[x, x + \Delta x)`:

.. math::

  d(x) = l(x) * q(x)

Survival Curve
=====================

Formally, the area under the survival curve between ages :math:`[x, x + \Delta x)` is given by:

.. math::

  L(x) = \int_{x}^{x + \Delta x} l(\chi) d\chi

However, we can approximate this using the midpoint formula for numerical integration:

.. math::

  L(x) = \Delta x \cdot l(x + \Delta x) + a(x) \cdot d(x) \cdot \Delta x

Assuming that deaths are uniform across the interval :math:`[x, x + \Delta x)`, we have
:math:`a(x) = 0.5` for all :math:`x`. Thus, we can simplify the above equation to:

.. math::

  \begin{align}
  L(x) &= (l(x + \Delta x) + 0.5 \cdot d(x)) \Delta x \\
  &= (l(x) \cdot (1 - q(x)) + 0.5 \cdot d(x))\Delta x  \\
  &= (l(x) - l(x) \cdot q(x) + 0.5 \cdot d(x))\Delta x  \\
  &= (l(x) - d(x) + 0.5 \cdot d(x)) \Delta x \\
  &= (l(x) - 0.5 \cdot d(x))\Delta x
  \end{align}

Because infant mortality is highest in the first few days of life, if :math:`\Delta x < 7 \text{days}`,
we set :math:`a(x_0) = 0.1`.

For the final age group, since everyone dies, :math:`l(x_f + \Delta x) = 0`, and :math:`q(x_f) = 1`.
Thus, we have:

.. math::

  \begin{align}
  L(x_f) &= (l(x_f + \Delta x) + a(x_f) \cdot d(x_f)) \Delta x \\
  &= (0 + a(x_f) \cdot d(x_f)) \Delta x \\
  &= a(x_f) \cdot d(x_f) \cdot \Delta x \\
  &= a(x_f) \cdot l(x_f) \cdot q(x_f) \cdot \Delta x \\
  &= a(x_f) \cdot l(x_f) \cdot \Delta x
  \end{align}

Time Lived After Age :math:`x`
==============================

:math:`T(x)`: calculate the total time lived after age :math:`x` by all people alive at age :math:`x`

.. math::

    T(x) = \sum_{n = 0}^{N} L(x + n \cdot \Delta x)

where :math:`x + N \cdot \Delta x = 110` is the maximum age in the life table.

Number of Years Left to Live
============================

The number of years left to live at age :math:`x` is given by :math:`E(x)`:

.. math::

    E(x) = \dfrac{T(x)}{l(x)}

Finally, to get the ``life expectancy``, we calculate the number of years left to live at
age ``0``, i.e. :math:`E(0)`:

.. math::

    \text{life expectancy} := E(0) = \dfrac{T(0)}{l(0)}
