============================================
Appendix: Life Expectancy
============================================

Life expectancy is computed from a projected life table using the following standard
actuarial definitions. It is used during calibration to evaluate how well a candidate
:math:`\beta_{\text{sex}}` reproduces Statistics Canada's targets.

We first define some variables:

.. list-table::
   :class: long-table
   :widths: 15 15 80
   :header-rows: 1

   * - Variable
     - Type
     - Definition
   * - :math:`x`
     - :code:`float`
     - age in years
   * - :math:`\Delta x`
     - :code:`float`
     - the width of the age interval in years
   * - :math:`\ell(x, \Delta x, t)`
     - :code:`int`
     - the number of people alive up to age :math:`x` at timepoint :math:`t`  
   * - :math:`q(x, \Delta x, t)`
     - :code:`float`
     - the probability of death between ages :math:`[x, x + \Delta x)` at timepoint :math:`t`
   * - :math:`d(x, \Delta x, t)`
     - :code:`int`
     - the number of deaths between ages :math:`[x, x + \Delta x)` at timepoint :math:`t`
   * - :math:`a(x, \Delta x, t)`
     - :code:`float`
     - the average fraction of the interval :math:`[x, x + \Delta x)` lived by those who die in the
       interval, for a given timepoint :math:`t` and :math:`\Delta x`
   * - :math:`L(x, \Delta x)`
     - :code:`int`
     - the area under the survival curve between ages :math:`[x, x + \Delta x)`
   * - :math:`T(x, t)`
     - :code:`int`
     - the total number of person-years lived after age :math:`x`
   * - :math:`E(x, t)`
     - :code:`float`
     - the number of years left to live at age :math:`x`
   * - :math:`E(x=0, t)`
     - :code:`float`
     - the number of years left to live at age :math:`0` for someone born at timepoint :math:`t`,
       i.e. ``life expectancy``


Number of People Alive
======================

First, we set the number of people alive at age :math:`0` at timepoint :math:`t`:

.. math::

  \ell(x=0, \Delta x, t) := 100000

Next, we calculate the number of people alive up to age :math:`x`:

.. math::

  \textcolor{magenta}{\underbrace{\ell(x, \Delta x, t)}_{\text{no. alive at age } x}}
  = \textcolor{orange}{\underbrace{\ell(x-\Delta x, \Delta x, t)}_{\text{no. alive at age } (x - \Delta x)}} 
  - \textcolor{green}{\underbrace{\ell(x-\Delta x, \Delta x, t) \cdot q(x-\Delta x, \Delta x, t)}_{\text{no. died between ages } [x - \Delta x, x)}}

Total Deaths
=============

The number of deaths :math:`d(x, \Delta x, t)` between ages :math:`[x, x + \Delta x)`, is given by
the number of people alive at age :math:`x` multiplied by the probability of death between
ages :math:`[x, x + \Delta x)`:

.. math::

  \textcolor{magenta}{\underbrace{d(x, \Delta x, t)}_{\text{no. died between ages } [x, x + \Delta x)}}
  = \textcolor{orange}{\underbrace{\ell(x, \Delta x, t)}_{\text{no. alive at age } x}} \cdot 
  \textcolor{green}{\underbrace{q(x, \Delta x, t)}_{\text{prob. of death between ages } [x, x + \Delta x)}}

Survival Curve
=====================

Formally, the area under the survival curve between ages :math:`[x, x + \Delta x)` is given by:

.. math::

  L(x, \Delta x, t) = \int_{x}^{x + \Delta x} \ell(\chi, \Delta \chi, t) d\chi

However, we can approximate this using the midpoint formula for numerical integration:

.. math::

  L(x, \Delta x, t) \approx \ell(x + \Delta x, \Delta x, t) \cdot \Delta x
    + a(x, \Delta x, t) \cdot d(x, \Delta x, t) \cdot \Delta x

We can simplify the above equation to:

.. math::

  \begin{align}
  L(x, \Delta x, t) &= (\ell(x, \Delta x, t) -  (1 - a(x, \Delta x, t)) \cdot d(x, \Delta x, t))\Delta x
  \end{align}

.. info:: Math: :math:`L(x, \Delta x, t)`
  :collapsible:

  .. math::

    \begin{align}
    L(x, \Delta x, t)
    &= (\ell(x + \Delta x, \Delta x, t) + a(x, \Delta x, t) \cdot d(x, \Delta x, t)) \Delta x \\
    &= (\ell(x, \Delta x, t) \cdot (1 - q(x, \Delta x, t)) + a(x, \Delta x, t) \cdot d(x, \Delta x, t))\Delta x  \\
    &= (\ell(x, \Delta x, t) - \textcolor{magenta}{\ell(x, \Delta x, t) \cdot q(x, \Delta x, t)} 
      + a(x, \Delta x, t) \cdot d(x, \Delta x, t))\Delta x  \\
    &= (\ell(x, \Delta x, t) - \textcolor{magenta}{d(x, \Delta x, t)} + a(x, \Delta x, t) \cdot d(x, \Delta x, t)) \Delta x \\
    &= (\ell(x, \Delta x, t) -  (1 - a(x, \Delta x, t)) \cdot d(x, \Delta x, t))\Delta x
    \end{align}


Assuming that deaths are uniform across the interval :math:`[x, x + \Delta x)`, we have
:math:`a(x) = 0.5` for all :math:`x`. Thus, we have:

.. math::

  \begin{align}
  L(x, \Delta x, t) = (\ell(x, \Delta x, t) - 0.5 d(x, \Delta x, t))\Delta x
  \end{align}

Because infant mortality is highest in the first few days of life, if :math:`\Delta x < 7 \text{days}`,
we set :math:`a(x_0) = 0.1`.

For the final age group, since everyone dies, :math:`l(x_f + \Delta x, \Delta x, t) = 0`, and
:math:`q(x_f, \Delta x, t) = 1`.
Thus, we have:

.. math::

  \begin{align}
  L(x_f, \Delta x, t)
  &= (\ell(x_f + \Delta x, \Delta x, t) + a(x_f, \Delta x , t) \cdot d(x_f, \Delta x, t)) \Delta x \\
  &= (0 + a(x_f, \Delta x, t) \cdot d(x_f, \Delta x, t)) \Delta x \\
  &= a(x_f, \Delta x, t) \cdot d(x_f, \Delta x, t) \cdot \Delta x \\
  &= a(x_f, \Delta x, t) \cdot \ell(x_f, \Delta x, t) \cdot q(x_f, \Delta x, t) \cdot \Delta x \\
  &= a(x_f, \Delta x, t) \cdot \ell(x_f, \Delta x, t) \cdot \Delta x
  \end{align}

Time Lived After Age :math:`x`
==============================

:math:`T(x, t)`: calculate the total time lived after age :math:`x` by all people alive at
age :math:`x`

.. math::

    T(x, t) = \sum_{n = 0}^{N} L(x + n \cdot \Delta x, \Delta x, t)

where :math:`x + N \cdot \Delta x = 110` is the maximum age in the life table, and :math:`t`
represents the current timepoint.

Number of Years Left to Live
============================

The number of years left to live at age :math:`x` is given by :math:`E(x, t)`:

.. math::

    E(x, t) = \dfrac{T(x, t)}{\ell(x, t)}

Finally, to get the ``life expectancy``, we calculate the number of years left to live at
age ``0``, i.e. :math:`E(0, t)`:

.. math::

    \text{life expectancy} := E(0, t) = \dfrac{T(0, t)}{\ell(0, t)}
