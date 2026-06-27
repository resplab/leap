.. _statistical-background:

======================
Statistical Background
======================

This page provides background on the statistical models and methods used in LEAP.

.. contents:: Contents
   :local:
   :depth: 2

.. _generalized-linear-models:

Generalized Linear Models
==========================

A generalized linear model (GLM) is a generalization of ordinary linear regression that
allows for response variables with non-normal error distributions.

Background: Ordinary Linear Regression
-----------------------------------------

In ordinary linear regression, we assume that the response variable :math:`y` can be
described by the equation:

.. math::

    y_i = \beta_0 + \beta_1 x_{1,i} + \beta_2 x_{2,i} + \ldots + \beta_n x_{n,i}

where :math:`i` is the index of the sample, :math:`n` is the number of features,
:math:`\beta_0, \beta_1, \ldots, \beta_n` are the model coefficients, and
:math:`x_{1,i}, \ldots, x_{n,i}` are the features of the :math:`i`-th sample.

GLM: Linear Predictor
-----------------------

For a GLM, we assume that the relationship between different terms is linear, but the terms
themselves are not necessarily linear. The linear predictor :math:`\eta_i` can include
polynomial terms, interaction terms, or transformed variables. The key constraint is that the
:math:`\beta` parameters must enter linearly — exponents on :math:`\beta` themselves are not
permitted.

**Valid** examples of linear predictors:

.. math::

    \eta_i &= \beta_0 + \beta_{11} x_{1,i} + \beta_{12} x_{1,i}^2
              + \beta_{21} x_{2,i} + \beta_{22} x_{2,i}^2
              \quad \text{(polynomial)} \\[4pt]
    \eta_i &= \beta_0 + \beta_{12} x_{1,i} x_{2,i} + \beta_3 x_{3,i}
              \quad \text{(interaction)} \\[4pt]
    \eta_i &= \beta_0 + \beta_1 e^{x_{1,i}} + \beta_2 x_{2,i}
              \quad \text{(transformed covariate)}

**Invalid** — the exponent on a :math:`\beta` makes this non-linear:

.. math::

    \eta_i = \beta_0 + \beta_1 x_{1,i}^{\beta_5} + \beta_2 x_{2,i}

GLM: Random Component
-----------------------

The random component assumes that the observed data :math:`Y` comes from an exponential
family of distributions — such as Gaussian, Poisson, binomial, or gamma:

.. math::

    p(y_i;\, \theta_i) = a(\theta_i)\, b(y_i)\, e^{y_i Q(\theta_i)}

where :math:`\theta_i` is a function of the explanatory variables and :math:`Q(\theta)` is
the natural parameter.

.. _glm-link-function:

GLM: Link Function
--------------------

The link function :math:`g` connects the mean :math:`\mu_i = E(Y \mid X = x_i)` to the
linear predictor:

.. math::

    g(\mu_i) = \eta_i

The goal of a GLM is to estimate :math:`\mu`, not :math:`\eta` directly. The link function
that maps :math:`\mu` to :math:`Q(\theta)` is called the *canonical link function*.

Example 1: Identity Link (Ordinary Linear Regression)
---------------------------------------------------------

For ordinary linear regression, the link function is the identity:

.. math::

    g(\mu_i) = \mu_i = \eta_i

The random component is the Gaussian distribution:

.. math::

    p(y_i;\, \mu_i) = \frac{1}{\sigma\sqrt{2\pi}}\, e^{-\frac{(y_i - \mu_i)^2}{2\sigma^2}}

Example 2: Poisson Distribution with Log Link
-----------------------------------------------

When the response variable represents **counts of events** (non-negative integers), the
Poisson distribution is a natural choice. Incidence and prevalence rates — counts of
individuals diagnosed with or living with a condition in a given time interval — follow this
pattern.

The **log link** is the canonical link for the Poisson distribution. The mean :math:`\mu_i`
is always positive, but the linear predictor :math:`\eta_i` can be any real number. The log
link satisfies this constraint:

.. math::

    g(\mu_i) = \log(\mu_i) = \eta_i

The random component is the Poisson distribution:

.. math::

    P(Y = y) = \frac{(\mu_i)^y\, e^{-\mu_i}}{y!}

This is the distribution family used in LEAP's :ref:`occurrence Model 1 <occurrence-model-1>`
to predict population-level asthma incidence and prevalence rates.

.. _negative-binomial-glm:

Example 3: Negative Binomial Distribution with Log Link
----------------------------------------------------------

The Poisson distribution assumes that the mean and variance are equal:

.. math::

    \mu = \sigma^2

When the variance exceeds the mean — a common problem in count data known as
**overdispersion** — the ``Negative Binomial`` distribution is a better choice. It
introduces an extra parameter :math:`\theta` that controls the degree of overdispersion:

.. math::

    \sigma^2 = \mu + \frac{\mu^2}{\theta}

As :math:`\theta \to \infty`, the ``Negative Binomial`` distribution converges to the ``Poisson``
distribution.

The standard form of the ``Negative Binomial`` probability mass function is:

.. math::

    P(Y = k;\, r, p) := \binom{k+r-1}{k}(1-p)^k p^r

where :math:`k` is the number of failures before :math:`r` successes, and :math:`p` is the
probability of success. We reparametrize using :math:`\mu` and :math:`\theta` via:

.. math::

    p &= \frac{\mu}{\sigma^2} \\
    r &= \frac{\mu^2}{\sigma^2 - \mu} \\
    \sigma^2 &= \mu + \frac{\mu^2}{\theta}

Substituting :math:`\sigma^2` and simplifying:

.. math::

    p &= \frac{\theta}{\theta + \mu} \\
    r &= \theta

.. info:: Math: :math:`p` and :math:`r`
    :collapsible:

    .. math::

        p &= \dfrac{\mu}{\sigma^2} = \dfrac{\mu}{\mu + \dfrac{\mu^2}{\theta}} = \dfrac{\theta}{\theta + \mu} \\
        r &= \dfrac{\mu^2}{\sigma^2 - \mu} = \dfrac{\mu^2}{\mu + \dfrac{\mu^2}{\theta} - \mu} 
        = \dfrac{\mu^2}{\dfrac{\mu^2}{\theta}} 
        = \theta


Letting :math:`y = k`, the probability mass function in terms of :math:`\mu` and :math:`\theta` is:

.. math::

    P(Y = y;\, \mu, \theta) = \binom{y + \theta - 1}{y}
        \frac{\mu^y\, \theta^{\theta}}{(\theta + \mu)^{y+\theta}}

.. info:: Math: :math:`P(Y = y;\, \mu, \theta)`
    :collapsible:

    .. math::

        P(Y = y; \mu, \theta) &= \binom{y + \theta - 1}{y}
            \left(1-\dfrac{\theta}{\theta + \mu}\right)^y 
            \left(\dfrac{\theta}{\theta + \mu}\right)^{\theta} \\
        &= \binom{y + \theta - 1}{y}
            \left(\dfrac{\theta + \mu - \theta}{\theta + \mu}\right)^y 
            \left(\dfrac{\theta}{\theta + \mu}\right)^{\theta} \\
        &= \binom{y + \theta - 1}{y}
            \dfrac{\mu^y \theta^{\theta}}{(\theta + \mu)^{y+\theta}}


The **log link** is the natural choice, since the mean is always positive but the linear
predictor :math:`\eta^{(i)}` can be any real number:

.. math::

    g(\mu^{(i)}) = \log(\mu^{(i)}) = \eta^{(i)}

This is the distribution family used in LEAP's :ref:`antibiotic_exposure_model` to predict
per-capita antibiotic exposure rates.

.. _ordinal-regression:

Example 4: Ordinal Regression with Logit Link
-----------------------------------------------

Ordinal regression is used when the response variable is ordered but the intervals between
levels are arbitrary. Rather than modelling a single mean, the model predicts the
**cumulative probability** of being at or below each level :math:`k` using the logistic
(sigmoid) function as the link:

.. math::

  P(y \leq k) = \sigma(\theta_k + \eta)

where :math:`\theta_k` is the threshold parameter for level :math:`k`, :math:`\eta` is the
linear predictor, and :math:`\sigma(x) = \dfrac{1}{1 + e^{-x}}` is the logistic function.
The probability of being in exactly level :math:`k` follows from the cumulative probabilities:

.. math::

  P(y = k) = P(y \leq k) - P(y \leq k - 1)

A patient-specific random effect :math:`\beta_0^{(i)} \sim \mathcal{N}(0, \sigma^2)` can be
added to the linear predictor to account for within-subject correlation across repeated
measurements, giving:

.. math::

  P(y^{(i)} \leq k) = \sigma\!\left(\theta_k + \eta^{(i)} + \beta_0^{(i)}\right)

This is the model used in LEAP's :ref:`control-model` to predict asthma control level.

.. _contingency-tables:

Contingency Tables
==================

A contingency table (also called a cross-tabulation or crosstab) displays the joint frequency
distribution of two categorical variables. They are commonly used in statistics to examine
the relationship between two variables, and to calculate odds ratios and other measures of
association.

Two-by-Two Contingency Table
------------------------------

The simplest form is a 2×2 table, which has two rows and two columns. Consider two binary
variables — an exposure and an outcome — for a population of :math:`n` individuals:

.. raw:: html

  <table class="table">
      <thead>
      <tr>
          <th></th>
          <th>outcome +</th>
          <th>outcome -</th>
          <th>total</th>
      </tr>
      </thead>
      <tbody>
      <tr>
          <td>exposure +</td>
          <td><code class="notranslate">a</code></td>
          <td><code class="notranslate">b</code></td>
          <td><code class="notranslate">n₁</code></td>
      </tr>
      <tr>
          <td>exposure -</td>
          <td><code class="notranslate">c</code></td>
          <td><code class="notranslate">d</code></td>
          <td><code class="notranslate">n₀</code></td>
      </tr>
      <tr>
          <td>total</td>
          <td><code class="notranslate">n₂</code></td>
          <td></td>
          <td><code class="notranslate">n</code></td>
      </tr>
      </tbody>
  </table>

where the marginal totals are:

.. math::

    n_1 &= a + b \quad \text{(total with exposure)} \\
    n_0 &= c + d \quad \text{(total without exposure)} \\
    n_2 &= a + c \quad \text{(total with outcome)} \\
    n   &= a + b + c + d \quad \text{(total population)}

The **odds ratio** measures the strength of association between the exposure and the outcome:

.. math::

    \omega = \frac{a \cdot d}{b \cdot c}

An odds ratio of 1 indicates no association; values greater than 1 indicate that exposure is
associated with increased odds of the outcome.

Example: Smoking and Lung Cancer
----------------------------------

Suppose we observe ``n = 300`` patients and record whether they smoke and whether they have
lung cancer:

.. raw:: html

    <table class="table">
        <thead>
        <tr>
            <th></th>
            <th>lung cancer</th>
            <th>no lung cancer</th>
            <th>total</th>
        </tr>
        </thead>
        <tbody>
        <tr>
            <td>smoker</td>
            <td><code class="notranslate">a = 100</code></td>
            <td><code class="notranslate">b = 50</code></td>
            <td><code class="notranslate">n₁ = 150</code></td>
        </tr>
        <tr>
            <td>non-smoker</td>
            <td><code class="notranslate">c = 25</code></td>
            <td><code class="notranslate">d = 125</code></td>
            <td><code class="notranslate">n₀ = 150</code></td>
        </tr>
        <tr>
            <td>total</td>
            <td><code class="notranslate">n₂ = 125</code></td>
            <td></td>
            <td><code class="notranslate">n = 300</code></td>
        </tr>
        </tbody>
    </table>

The odds ratio for this table is:

.. math::

    \omega = \frac{100 \times 125}{50 \times 25} = 10

This means that smokers have 10 times the odds of developing lung cancer compared to
non-smokers in this population.

Usage in LEAP
--------------

Contingency tables are used in LEAP to optimize the age-dependent beta parameters for the
incidence equation in :ref:`occurrence-model-2`. See the
:ref:`optimizing-beta-parameters` section of the Asthma Occurrence Model for details.

Python Examples
================

For a hands-on walkthrough of GLMs using ``statsmodels`` and ``plotly`` — including
code you can run locally — see the notebook below.

.. toctree::
   :maxdepth: 1

   model-glm
