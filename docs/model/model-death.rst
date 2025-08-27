===========================
Mortality Model
===========================

Data
====

To obtain the mortality data for each year, we used one table from ``Statistics Canada``:

Past Data: 1996 - 2021
*************************

For past years, we used
`Table 13-10-00837-01 from StatCan <https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1310083701>`_.

The ``*.csv`` file can be downloaded from here:
`13100837-eng.zip <https://www150.statcan.gc.ca/n1/tbl/csv/13100837-eng.zip>`_
and is saved as:
`LEAP/leap/original_data/13100837.csv
<https://github.com/resplab/leap/blob/main/leap/original_data/13100837.csv>`_.

The relevant columns are:

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - ``REF_DATE``
     - :code:`int`
     - the calendar year
   * - ``AGE_GROUP``
     - :code:`str`
     - the age of the person in years
   * - ``GEO``
     - :code:`str`
     - the province or terriroty full name
   * - ``SEX``
     - :code:`str`
     - one of "Both sexes", "Females", or "Males"
   * - ``ELEMENT``
     - :code:`str`
     - describes what the variable of interest is; we want ``"Death probability between age x and x+1 (qx)"``
   * - ``VALUE``
     - :code:`int`
     - the probability of death between age ``x`` and ``x+1`` in that year, province, sex, and age group


Projected Data: 2021 - 2068
****************************

``Statistics Canada`` doesn't provide annual projections for death probabilities, but does
provide a projection for specific years (which we call calibration years):

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Region
     - Year
     - Projection Scenario
     - Mortality Scenario
   * - Canada
     - ``2028``
     - ``LG``
     - ``HM``
   * - Canada
     - ``2028``
     - ``M1``
     - ``MM``
   * - Canada
     - ``2028``
     - ``M2``
     - ``MM``
   * - Canada
     - ``2028``
     - ``M3``
     - ``MM``
   * - Canada
     - ``2028``
     - ``M4``
     - ``MM``
   * - Canada
     - ``2028``
     - ``M5``
     - ``MM``
   * - Canada
     - ``2028``
     - ``HG``
     - ``LM`` 
   * - Canada
     - ``2028``
     - ``SA``
     - ``HM``
   * - Canada
     - ``2028``
     - ``FA``
     - ``LM``
   * - Canada
     - ``2048``
     - ``LG``
     - ``HM``
   * - Canada
     - ``2048``
     - ``M1``
     - ``MM``
   * - Canada
     - ``2048``
     - ``M2``
     - ``MM``
   * - Canada
     - ``2028``
     - ``M3``
     - ``MM``
   * - Canada
     - ``2048``
     - ``M4``
     - ``MM``
   * - Canada
     - ``2048``
     - ``M5``
     - ``MM``
   * - Canada
     - ``2048``
     - ``HG``
     - ``LM`` 
   * - Canada
     - ``2048``
     - ``SA``
     - ``HM``
   * - Canada
     - ``2048``
     - ``FA``
     - ``LM``
   * - Canada
     - ``2073``
     - ``LG``
     - ``HM``
   * - Canada
     - ``2073``
     - ``M1``
     - ``MM``
   * - Canada
     - ``2073``
     - ``M2``
     - ``MM``
   * - Canada
     - ``2073``
     - ``M3``
     - ``MM``
   * - Canada
     - ``2073``
     - ``M4``
     - ``MM``
   * - Canada
     - ``2073``
     - ``M5``
     - ``MM``
   * - Canada
     - ``2073``
     - ``HG``
     - ``LM`` 
   * - Canada
     - ``2073``
     - ``SA``
     - ``HM``
   * - Canada
     - ``2073``
     - ``FA``
     - ``LM``
   * - Provinces / Territories
     - ``2028``
     - ``HG``, ``FA``
     - ``LM``
   * - Provinces / Territories
     - ``2028``
     - ``M1``, ``M2``, ``M3``, ``M4``, ``M5``, ``M6``
     - ``MM``
   * - Provinces / Territories
     - ``2028``
     - ``LG``, ``SA``
     - ``HM``
   * - Provinces / Territories
     - ``2033``
     - ``HG``, ``FA``
     - ``LM``
   * - Provinces / Territories
     - ``2033``
     - ``M1``, ``M2``, ``M3``, ``M4``, ``M5``, ``M6``
     - ``MM``
   * - Provinces / Territories
     - ``2033``
     - ``LG``, ``SA``
     - ``HM``
   * - Provinces / Territories
     - ``2038``
     - ``HG``, ``FA``
     - ``LM``
   * - Provinces / Territories
     - ``2038``
     - ``M1``, ``M2``, ``M3``, ``M4``, ``M5``, ``M6``
     - ``MM``
   * - Provinces / Territories
     - ``2038``
     - ``LG``, ``SA``
     - ``HM``
   * - Provinces / Territories
     - ``2043``
     - ``HG``, ``FA``
     - ``LM``
   * - Provinces / Territories
     - ``2043``
     - ``M1``, ``M2``, ``M3``, ``M4``, ``M5``, ``M6``
     - ``MM``
   * - Provinces / Territories
     - ``2043``
     - ``LG``, ``SA``
     - ``HM``
   * - Provinces / Territories
     - ``2048``
     - ``HG``, ``FA``
     - ``LM``
   * - Provinces / Territories
     - ``2048``
     - ``M1``, ``M2``, ``M3``, ``M4``, ``M5``, ``M6``
     - ``MM``
   * - Provinces / Territories
     - ``2048``
     - ``LG``, ``SA``
     - ``HM``
   * - Provinces / Territories
     - ``2053``
     - ``HG``, ``FA``
     - ``LM``
   * - Provinces / Territories
     - ``2053``
     - ``M1``, ``M2``, ``M3``, ``M4``, ``M5``, ``M6``
     - ``MM``
   * - Provinces / Territories
     - ``2053``
     - ``LG``, ``SA``
     - ``HM``
   * - Provinces / Territories
     - ``2058``
     - ``HG``, ``FA``
     - ``LM``
   * - Provinces / Territories
     - ``2058``
     - ``M1``, ``M2``, ``M3``, ``M4``, ``M5``, ``M6``
     - ``MM``
   * - Provinces / Territories
     - ``2058``
     - ``LG``, ``SA``
     - ``HM``
   * - Provinces / Territories
     - ``2063``
     - ``HG``, ``FA``
     - ``LM``
   * - Provinces / Territories
     - ``2063``
     - ``M1``, ``M2``, ``M3``, ``M4``, ``M5``, ``M6``
     - ``MM``
   * - Provinces / Territories
     - ``2063``
     - ``LG``, ``SA``
     - ``HM``
   * - Provinces / Territories
     - ``2068``
     - ``HG``, ``FA``
     - ``LM``
   * - Provinces / Territories
     - ``2068``
     - ``M1``, ``M2``, ``M3``, ``M4``, ``M5``, ``M6``
     - ``MM``
   * - Provinces / Territories
     - ``2068``
     - ``LG``, ``SA``
     - ``HM``
   * - Provinces / Territories
     - ``2073``
     - ``HG``, ``FA``
     - ``LM``
   * - Provinces / Territories
     - ``2073``
     - ``M1``, ``M2``, ``M3``, ``M4``, ``M5``, ``M6``
     - ``MM``
   * - Provinces / Territories
     - ``2073``
     - ``LG``, ``SA``
     - ``HM``



This data can be found in the ``Statistics Canada Population Projections Technical Report``:
`Table 3.1, Table 3.2, Table 5.2.1, Table 5.2.2, Table 5.2.3
<https://www150.statcan.gc.ca/n1/pub/91-620-x/91-620-x2025001-eng.htm>`_.


Model
========

We have mortality data for past years (1996 - 2020), and life expectancy projections for
specific future years; but we would like to have mortality data for all future years in our
model. ``Statistics Canada`` describes how they model mortality here:
`Methods for Constructing Life Tables for Canada, Provinces and Territories
<https://www150.statcan.gc.ca/n1/en/pub/84-538-x/84-538-x2021001-eng.pdf>`_.

In particular, the model they use is the ``Kannisto-Thatcher model``, described in this paper:
`On the use of Kannisto model for mortality trajectory modelling at very old ages
<https://ipc2025.popconf.org/uploads/252146>`_.

According to the Kannisto-Thatcher model, the instantaneous probability of death at age :math:`x`
is given by:

.. math::

    \mu(x) &= \dfrac{a e^{\beta x}}{1 + a e^{\beta x}} \\
    &= \lim_{\Delta x \to 0} \dfrac{P(\text{death between age $x$ and $x + \Delta x$} \mid \text{survived till $x$})}{\Delta x}

In mathematical terms, :math:`\mu(x)` is the ``hazard rate``. Let's break this down further.
Let :math:`F_X(x)` be the cumulative distribution function for age at death, :math:`X`:

.. math::

    F_X(x) :&= P(\text{age at death} \leq \text{given age}) \\ 
    &= P(X \leq x)

We want the conditional probability of death between age :math:`x` and :math:`x + \Delta x`,
given that the person has survived till age :math:`x`. This is given by:

.. math::

    P(x < X \leq x + \Delta x \mid X > x)

Recall that for a conditional probability:

.. math::
    
    P(A \mid B) = \dfrac{P(A \cap B)}{P(B)}

and so:

.. math::

    P(x < X \leq x + \Delta x \mid X > x) = \dfrac{P(x < X \leq x + \Delta x \bigcap X > x)}{P(X > x)}

Since :math:`F_X(x)` is the cumulative distribution function, by definition it must sum to 1:

.. math::

    P(X > x) = 1 - F_X(x)

Since :math:`X > x` if :math:`x < X \leq x + \Delta x`, we can rewrite the numerator as:

.. math::

    P(x < X \leq x + \Delta x \bigcap X > x) &= P(x < X \leq x + \Delta x) \\
    &= F_X(x + \Delta x) - F_X(x)

Putting it all together, we have:

.. math::

    P(x < X \leq x + \Delta x \mid X > x) = 
    \dfrac{F_X(x + \Delta x) - F_X(x)}{1 - F_X(x)}

Now, we want to find the instantaneous rate of death; the probability of death per unit time.
If we take the limit as :math:`\Delta x \to 0`, we will find the instantaneous probability of
death at age :math:`x`. To get the probability of death per unit time, we need to divide by
:math:`\Delta x`:

.. math::

    \mu(x) = \lim_{\Delta x \to 0} \dfrac{F_X(x + \Delta x) - F_X(x)}{\Delta x (1 - F_X(x))}

You will recognize the derivative of :math:`F_X(x)`:

.. math::

    \dfrac{d}{dx} F_X(x) = \lim_{\Delta x \to 0} \dfrac{F_X(x + \Delta x) - F_X(x)}{\Delta x}

and so:

.. math::

    \mu(x) = \dfrac{F_X'(x)}{1 - F_X(x)}

The data in the ``Statistics Canada`` mortality table is the probability of death between age
:math:`x` and :math:`x + 1`, which is denoted as :math:`q_x`. This is the same as the probability
:math:`P(x < X \leq x + \Delta x \mid X > x)`, with :math:`\Delta x = 1`. We would like to solve
for :math:`q_x`, using the ``Kannisto-Thatcher Equation`` for :math:`\mu(x)`. First, we can
write :math:`q_x` in terms of :math:`F_X(x)`:

.. math::

    q_x &= P(x < X \leq x + 1 \mid X > x) \\
    &= \dfrac{F_X(x + 1) - F_X(x)}{1 - F_X(x)}

Let us define :math:`S_X(x)`, the survival function, for convenience:

.. math::

    S_X(x) &:= 1 - F_X(x) \\
    &= P(X > x)

Then we have:

.. math::

    \dfrac{dS}{dx} = -F_X'(x)

and so :math:`\mu(x)` can be rewritten as:

.. math::

    \mu(x) = -\dfrac{dS}{dx}\dfrac{1}{S_X(x)}

Solving this first order separable linear differential equation, we have:

.. math::

    \int \dfrac{dS}{S_X} &= -\int \mu(x) dx \\
    \ln(S_X(x)) &= -\int \mu(x) dx + C \\
    &= -\int \dfrac{a e^{\beta x}}{1 + a e^{\beta x}} dx + C

Letting :math:`u(x) := 1 + a e^{\beta x}`, we have:

.. math::

    \ln(S_X(x)) &= - \dfrac{1}{\beta} \int \dfrac{du}{u} + C \\
    &= - \dfrac{1}{\beta} \ln(u(x)) + C \\
    S_X(x) &= e^C (1 + a e^{\beta x})^{-\frac{1}{\beta}} \\
    &= k (1 + a e^{\beta x})^{-\frac{1}{\beta}} \\
    1 - F_X(x) &= k (1 + a e^{\beta x})^{-\frac{1}{\beta}} \\
    F_X(x) &= 1 - k (1 + a e^{\beta x})^{-\frac{1}{\beta}}

Now, we can substitute this into the equation for :math:`q_x`:

.. math::

    q_x &= \dfrac{F_X(x + \Delta x) - F_X(x)}{1 - F_X(x)} \\
    &= \dfrac{
        1 - k (1 + a e^{\beta (x + \Delta x)})^{-\frac{1}{\beta}} - 
        1 + k (1 + a e^{\beta x})^{-\frac{1}{\beta}}
    }{k (1 + a e^{\beta x})^{-\frac{1}{\beta}}} \\
    &= \dfrac{
        - k (1 + a e^{\beta (x + \Delta x)})^{-\frac{1}{\beta}}
        + k (1 + a e^{\beta x})^{-\frac{1}{\beta}}
    }{k (1 + a e^{\beta x})^{-\frac{1}{\beta}}} \\
    &= 1 - \left(\dfrac{1 + a e^{\beta (x + \Delta x)}}{1 + a e^{\beta x}}\right)^{-\frac{1}{\beta}} \\
    &= 1 - \left(\dfrac{1 + a e^{\beta x}}{1 + a e^{\beta (x + \Delta x)}}\right)^{\frac{1}{\beta}}

If we take the ``logit`` of :math:`q_x`, we have:

.. math::

    \sigma^{-1}(q_x) &= \ln\left(\dfrac{q_x}{1 - q_x}\right) \\
    &= \ln\left(\dfrac{
        1 - \left(\dfrac{1 + a e^{\beta x}}{1 + a e^{\beta (x + \Delta x)}}\right)^{\frac{1}{\beta}}
    }{
        \left(\dfrac{1 + a e^{\beta x}}{1 + a e^{\beta (x + \Delta x)}}\right)^{\frac{1}{\beta}}
    }\right) \\
    &= \ln\left(
        \left(\dfrac{1 + a e^{\beta (x + \Delta x)}}{1 + a e^{\beta x}}\right)^{\frac{1}{\beta}} - 1
    \right) \\
    &= \ln
        \left(\dfrac{
            (1 + \alpha e^{\beta (x + \Delta x)})^{\frac{1}{\beta}} - 
            (1 + \alpha e^{\beta x})^{\frac{1}{\beta}}
        }{(1 + \alpha e^{\beta x})^{\frac{1}{\beta}}}\right)
     \\
    &= \ln\left(
            (1 + \alpha e^{\beta (x + \Delta x)})^{\frac{1}{\beta}} - 
            (1 + \alpha e^{\beta x})^{\frac{1}{\beta}}
        \right) -
        \dfrac{1}{\beta}\ln(1 + \alpha e^{\beta x})

Let us now look at :math:`\sigma^{-1}(q_x) - \sigma^{-1}(q_{x_0})`:

.. math::

    \sigma^{-1}(q_x) - \sigma^{-1}(q_{x_0}) &= 
        \ln\left(
            (1 + \alpha e^{\beta (x + \Delta x)})^{\frac{1}{\beta}} - 
            (1 + \alpha e^{\beta x})^{\frac{1}{\beta}}
        \right) -
        \dfrac{1}{\beta}\ln(1 + a e^{\beta x}) \\
        &- 
            \ln\left(
                (1 + \alpha e^{\beta (x_0 + \Delta x)})^{\frac{1}{\beta}} - 
                (1 + \alpha e^{\beta x_0})^{\frac{1}{\beta}}
            \right) +
            \dfrac{1}{\beta}\ln(1 + \alpha e^{\beta x_0})

Now, based on fitting the model to empirical data, typically we have
[Appendix D, Table 5, :cite:`kannisto`]:

1. :math:`\beta \approx \mathcal{O}(10^{-1})`
2. :math:`\alpha \approx \mathcal{O}(10^{-5})`


We can use the binomial approximation to simplify the above equation. Let us take:

.. math::

    (1 + \alpha e^{\beta x})^{\frac{1}{\beta}}

In order to use the binomial approximation, we must have:

.. math::

    \left|\alpha e^{\beta x}\right| &< 1 \\
    \left|\dfrac{\alpha e^{\beta x}}{\beta}\right| &\ll 1 \\
    
Since :math:`x` represents the age in years, we have :math:`x \in [0, 120]`. These conditions hold
for all ages. Using the binomial approximation, we have:

.. math::

    (1 + \alpha e^{\beta x})^{\frac{1}{\beta}} \approx 1 + \dfrac{\alpha e^{\beta x}}{\beta}

Going back to our equation for :math:`\sigma^{-1}(q_x) - \sigma^{-1}(q_{x_0})`, we have:

.. math::

    \sigma^{-1}(q_x) - \sigma^{-1}(q_{x_0}) &\approx
        \ln\left(
            1 + \dfrac{\alpha e^{\beta (x + \Delta x)}}{\beta} - 
            1 - \dfrac{\alpha e^{\beta x}}{\beta}
        \right) -
        \ln\left(1 + \dfrac{\alpha e^{\beta x}}{\beta}\right) \\
        &- 
            \ln\left(
               1 + \dfrac{\alpha e^{\beta (x_0 + \Delta x)}}{\beta} - 
               1 - \dfrac{\alpha e^{\beta x_0}}{\beta}
            \right) +
            \ln\left(1 + \dfrac{\alpha e^{\beta x_0}}{\beta}\right) \\
    &=  \ln\left(
            \dfrac{\alpha e^{\beta (x + \Delta x)}}{\beta} - 
            \dfrac{\alpha e^{\beta x}}{\beta}
        \right) -
        \ln\left(1 + \dfrac{\alpha e^{\beta x}}{\beta}\right) \\
        &- 
            \ln\left(
               \dfrac{\alpha e^{\beta (x_0 + \Delta x)}}{\beta} - 
               \dfrac{\alpha e^{\beta x_0}}{\beta}
            \right) +
            \ln\left(1 + \dfrac{\alpha e^{\beta x_0}}{\beta}\right) \\
    &= \textcolor{orange}{\cancel{\ln\left(\dfrac{\alpha}{\beta}\right)}} + \ln(e^{\beta x})+
        \textcolor{magenta}{\cancel{\ln\left(e^{\beta \Delta x} - 1\right)}} -
        \ln\left(1 + \dfrac{\alpha e^{\beta x}}{\beta}\right) \\
        &- 
            \textcolor{orange}{\cancel{\ln\left(\dfrac{\alpha}{\beta}\right)}} - \ln(e^{\beta x_0}) -
            \textcolor{magenta}{\cancel{\ln\left(e^{\beta \Delta x} - 1\right)}} +
            \ln\left(1 + \dfrac{\alpha e^{\beta x_0}}{\beta}\right) \\
    &= \ln(e^{\beta x})+
        \ln\left(1 + \dfrac{\alpha e^{\beta x_0}}{\beta}\right) -
        \ln\left(1 + \dfrac{\alpha e^{\beta x}}{\beta}\right) -
        \ln(e^{\beta x_0}) \\
    &= \beta (x - x_0) + 
    \ln\left(\dfrac{1 + \dfrac{\alpha e^{\beta x_0}}{\beta}}{1 + \dfrac{\alpha e^{\beta x}}{\beta}}\right)

The last term is much smaller than the first term, and so we can ignore it. Thus, we have:

.. math::

    \sigma^{-1}(q_x) \approx \sigma^{-1}(q_{x_0}) + \beta (x - x_0)

If :math:`x_0` is the age of the person in the starting year of the simulation, then
:math:`(\text{year} - \text{year}_0) = (x - x_0)`:

.. math::

    \sigma^{-1}(q_x(\text{sex}, \text{age})) = 
        \sigma^{-1}(q_{x_0}(\text{sex}, \text{age})) -
        \beta_{\text{sex}}(\text{year} - \text{year}_0)


The parameter :math:`\beta_{\text{sex}}` is unknown, and so we first need to calculate it.
To do so, we set :math:`\text{year} = \text{year}_C`, the calibration year, and use the ``Brent``
root-finding algorithm to optimize :math:`\beta_{\text{sex}}` such that the life expectancy in the
calibration year (which is known) matches the predicted life expectancy.

Once we have found :math:`\beta_{\text{sex}}`, we can use this formula to find the projected death
probabilities.