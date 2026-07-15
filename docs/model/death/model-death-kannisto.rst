============================================
Appendix: Kannisto-Thatcher Model
============================================

For the age groups 95 to 109 and the open age group of 110 years and over,
Statistics Canada models mortality using the ``Kannisto-Thatcher model``, described in:
`On the use of Kannisto model for mortality trajectory modelling at very old ages
<https://ipc2025.popconf.org/uploads/252146>`_.
See also: `Methods for Constructing Life Tables for Canada, Provinces and Territories
<https://www150.statcan.gc.ca/n1/en/pub/84-538-x/84-538-x2021001-eng.pdf>`_.

According to the Kannisto-Thatcher model, the instantaneous probability of death at age :math:`x`
is given by:

.. math::

    \mu(x, \Delta x, t) &= \dfrac{\alpha(t) e^{\beta(t) x}}{1 + \alpha(t) e^{\beta(t) x}} \\
    &= \lim_{\Delta x \to 0} 
    \dfrac{P(\text{death between age $x$ and $x + \Delta x$} \mid \text{survived till $x$})}{\Delta x}

In mathematical terms, :math:`\mu(x, \Delta x, t)` is the ``hazard rate``. Let's break this down further.
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

    \mu(x, \Delta x, t) = \lim_{\Delta x \to 0} \dfrac{F_X(x + \Delta x) - F_X(x)}{\Delta x (1 - F_X(x))}

You will recognize the derivative of :math:`F_X(x)`:

.. math::

    \dfrac{d}{dx} F_X(x) = \lim_{\Delta x \to 0} \dfrac{F_X(x + \Delta x) - F_X(x)}{\Delta x}

and so:

.. math::

    \mu(x, \Delta x, t) = \dfrac{F_X'(x)}{1 - F_X(x)}

The data in the ``Statistics Canada`` mortality table is the probability of death between age
:math:`x` and :math:`x + 1`, which is denoted as :math:`q_x`. This is the same as the probability
:math:`P(x < X \leq x + \Delta x \mid X > x)`, with :math:`\Delta x = 1`. We would like to solve
for :math:`q(x, \Delta x, t)`, using the ``Kannisto-Thatcher Equation`` for :math:`\mu(x)`. First,
we can write :math:`q(x, \Delta x, t)` in terms of :math:`F_X(x)`:

.. math::

    q(x, \Delta x, t) &= P(x < X \leq x + 1 \mid X > x) \\
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

    \mu(x, \Delta x, t) = -\dfrac{dS}{dx}\dfrac{1}{S_X(x)}


Solving this first order separable linear differential equation, we have:

.. math::

    F_X(x) = 1 - k (1 + \alpha(t) e^{\beta(t) x})^{-\frac{1}{\beta(t)}}

.. info:: Math: :math:`F_X(x)`
    :collapsible:

    .. math::

        \int \dfrac{dS}{S_X} &= -\int \mu(x, \Delta x, t) dx \\
        \ln(S_X(x)) &= -\int \mu(x, \Delta x, t) dx + C \\
        &= -\int \dfrac{\alpha(t) e^{\beta(t) x}}{1 + \alpha(t) e^{\beta(t) x}} dx + C

    Letting :math:`u(x, t) := 1 + \alpha(t) e^{\beta(t) x}`, we have:

    .. math::

        \ln(S_X(x)) &= - \dfrac{1}{\beta(t)} \int \dfrac{du}{u} + C \\
        &= - \dfrac{1}{\beta(t)} \ln(u(x, t)) + C \\
        S_X(x) &= e^C (1 + \alpha(t) e^{\beta(t) x})^{-\frac{1}{\beta(t)}} \\
        &= k (1 + \alpha(t) e^{\beta(t) x})^{-\frac{1}{\beta(t)}} \\
        1 - F_X(x) &= k (1 + \alpha(t) e^{\beta(t) x})^{-\frac{1}{\beta(t)}} \\
        F_X(x) &= 1 - k (1 + \alpha(t) e^{\beta(t) x})^{-\frac{1}{\beta(t)}}

Now, we can substitute this into the equation for :math:`q(x, \Delta x, t)`:

.. math::

    q(x, \Delta x, t) &= \dfrac{F_X(x + \Delta x) - F_X(x)}{1 - F_X(x)} \\
    &= 1 - \left(
        \dfrac{1 + \alpha(t) e^{\beta(t) x}}{1 + \alpha(t) e^{\beta(t) (x + \Delta x)}}
    \right)^{\frac{1}{\beta(t)}}


.. info:: Math: :math:`q(x, \Delta x, t)`
    :collapsible:

    .. math::

        q(x, \Delta x, t) &= \dfrac{F_X(x + \Delta x) - F_X(x)}{1 - F_X(x)} \\
        &= \dfrac{
            1 - k (1 + \alpha(t) e^{\beta(t) (x + \Delta x)})^{-\frac{1}{\beta(t)}} -
            1 + k (1 + \alpha(t) e^{\beta(t) x})^{-\frac{1}{\beta(t)}}
        }{k (1 + \alpha(t) e^{\beta(t) x})^{-\frac{1}{\beta(t)}}} \\
        &= \dfrac{
            - k (1 + \alpha(t) e^{\beta(t) (x + \Delta x)})^{-\frac{1}{\beta(t)}}
            + k (1 + \alpha(t) e^{\beta x})^{-\frac{1}{\beta(t)}}
        }{k (1 + \alpha(t) e^{\beta(t) x})^{-\frac{1}{\beta(t)}}} \\
        &= 1 - \left(\dfrac{1 + a e^{\beta(t) (x + \Delta x)}}{1 + \alpha(t) e^{\beta(t) x}}\right)^{-\frac{1}{\beta}} \\
        &= 1 - \left(
            \dfrac{1 + \alpha(t) e^{\beta(t) x}}{1 + \alpha(t) e^{\beta(t) (x + \Delta x)}}
        \right)^{\frac{1}{\beta(t)}}



Now, based on fitting the model to empirical data, typically we have
[Appendix D, Table 5, :cite:`kannisto`]:

1. :math:`\beta \approx \mathcal{O}(10^{-1})`
2. :math:`\alpha \approx \mathcal{O}(10^{-5})`




