=================
Occurrence Model
=================

Occurrence Model 1: Crude Occurrence
=====================================

In the first model, we will use data collected from the ``BC Ministry of Health`` on the
incidence and prevalence of asthma in British Columbia. We will use this data to fit a 
``Generalized Linear Model (GLM)`` to predict the incidence and prevalence of asthma
given the age, sex, and year.

Datasets
*****************

The BC Ministry of Health Administrative Dataset contains asthma incidence and prevalence data
for the years ``2000-2019``, in 5-year age intervals. 

The data is formatted as follows:

.. raw:: html

  <table class="table">
    <thead>
      <tr>
          <th>Column</th>
          <th>Type</th>
          <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code class="notranslate">year</code></td>
        <td>
          <code class="notranslate">int</code>
        </td>
        <td>
          format <code>XXXX</code>, e.g <code>2000</code>, range <code>[2000, 2019]</code>
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">age</code></td>
        <td>
          <code class="notranslate">int</code>
        </td>
        <td>
          The midpoint of the age group, e.g. 3 for the age group 1-5 years
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">sex</code></td>
        <td>
          <code class="notranslate">int</code>
        </td>
        <td>
          1 = Female, 2 = Male
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">incidence</code></td>
        <td>
          <code class="notranslate">float</code>
        </td>
        <td>
          The incidence of asthma in BC for a given year, age group, and sex, per 100 people
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">prevalence</code></td>
        <td>
          <code class="notranslate">float</code>
        </td>
        <td>
          The prevalence of asthma in BC for a given year, age group, and sex, per 100 people
        </td>
      </tr>
    </tbody>
    </table>



Model: Generalized Linear Model - Poisson
****************************************************

Since our model projects into the future, we would like to be able to extend this data beyond
``2019``. Our model also makes predictions at 1-year age intervals, not 5-year age intervals.
To obtain these projections, we use a ``Generalized Linear Model (GLM)``. A ``GLM`` is a type of
regression analysis which is a generalized form of linear regression. See :doc:`model-glm` for more
information on ``GLMs``.

Probability Distribution
---------------------------------------

When fitting a ``GLM``, first you must choose a distribution for the ``response variable``. In our
case, the response variable is the asthma prevalence or incidence. Incidence and prevalence are
counts of the number of people diagnosed with asthma and the number of people with asthma,
respectively, in a given time interval (a year, in our case). Since these are counts, we need a
discrete probability distribution. The ``Poisson distribution`` is a good choice for our data.

.. math::

    P(Y = y) = p(y; \mu^{(i)}) = \dfrac{(\mu^{(i)})^{y} ~ e^{-\mu^{(i)}}}{y!}


Link Function
-----------------

We also need to choose a ``link function``. Recall that the link function :math:`g(\mu^{(i)})`
is used to relate the mean to the predicted value :math:`\eta^{(i)}`:

.. math::

    g(\mu^{(i)}) &= \eta^{(i)} \\
    \mu^{(i)} &= E(Y | X = x^{(i)})

How do we choose a link function? Well, we are free to choose any link function we like, but there
are some constraints. For example, in the Poisson distribution, the mean is always positive.
However, :math:`\eta^{(i)}` can be any real number. Therefore, we need a link function that maps
real numbers to positive numbers. The ``log link function`` is a good choice for this:

.. math::

    g(\mu^{(i)}) = \log(\mu^{(i)}) = \eta^{(i)}


Formula
-----------------

Now that we have our distribution and link function, we need to decide on a formula for
:math:`\eta^{(i)}`. We are permitted to use linear combinations of functions of the features
in our dataset.

Let's start with ``incidence``. We want a formula using ``age``, ``sex``, and ``year``.
Since asthma depends on factors such as pollution and antibiotic use, and these factors change
from year to year, it follows that asthma incidence should depend on the year. Antibiotic use
also depends on age, so we should include age in our formula. Finally, there is a sex difference
in asthma incidence, so we should include sex in our formula. 

TODO: Why was this formula chosen?


.. math::

    \eta^{(i)} = 
        \sum_{m=0}^1 \beta_{01m} t^{(i)} \cdot (s^{(i)})^m +
        \sum_{k=0}^{5} \sum_{m=0}^{1} \beta_{k0m} \cdot (a^{(i)})^k \cdot (s^{(i)})^m


where:

* :math:`\beta_{k\ell m}` is the coefficient for the feature :math:`(a^{(i)})^k \cdot (t^{(i)})^{\ell} \cdot (s^{(i)})^m`
* :math:`a^{(i)}` is the age
* :math:`t^{(i)}` is the year
* :math:`s^{(i)}` is the sex

There are :math:`2 + 6 * 2 = 14` coefficients in the incidence model.


Next we have the ``prevalence``. We again want a formula using ``age``, ``sex``, and ``year``.
Since asthma prevalence depends on the number of people who have asthma, and this number changes
from year to year, we should include year in our formula. Asthma prevalence also depends on age,
so we should include age in our formula. Finally, there is a sex difference
in asthma incidence and hence prevalence, so we should include sex in our formula.


.. math::

    \eta^{(i)} = \sum_{k=0}^{5} \sum_{\ell=0}^2 \sum_{m=0}^1 \beta_{k \ell m} 
        \cdot (a^{(i)})^k \cdot (t^{(i)})^{\ell} \cdot (s^{(i)})^m

There are :math:`6 * 3 * 2 = 36` coefficients in the prevalence model.


Occurrence Model 2: Risk Factors
=================================

Datasets
*****************

We use the predicted asthma incidence and prevalence from the first model, :math:`\eta`, as our
target asthma prevalence / incidence in this model. The data is formatted as follows:

.. raw:: html

  <table class="table">
    <thead>
      <tr>
          <th>Column</th>
          <th>Type</th>
          <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code class="notranslate">year</code></td>
        <td>
          <code class="notranslate">int</code>
        </td>
        <td>
          format <code>XXXX</code>, e.g <code>2000</code>, range <code>[2000, 2019]</code>
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">age</code></td>
        <td>
          <code class="notranslate">int</code>
        </td>
        <td>
          The age in years, a value in <code class="notranslate">[3, 100]</code>
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">sex</code></td>
        <td>
          <code class="notranslate">str</code>
        </td>
        <td>
          <code class="notranslate">"F"</code> = Female,
          <code class="notranslate">"M"</code> = Male
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">incidence</code></td>
        <td>
          <code class="notranslate">float</code>
        </td>
        <td>
          The predicted incidence of asthma in BC for a given year, age, and sex, per 100 people
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">prevalence</code></td>
        <td>
          <code class="notranslate">float</code>
        </td>
        <td>
          The predicted prevalence of asthma in BC for a given year, age, and sex, per 100 people
        </td>
      </tr>
    </tbody>
    </table>


Model: Risk Factors
****************************************************

We wanted to incorporate the effects of family history and antibiotic use on asthma incidence and
prevalence.

.. raw:: html

  <table class="table">
    <thead>
      <tr>
          <th>Risk Factor</th>
          <th>Values</th>
          <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Family History</td>
        <td>
          A value in <code class="notranslate">{0, 1}</code>
        </td>
        <td>
          1 = at least one parent has asthma, 0 = neither parent has asthma
        </td>
      </tr>
      <tr>
        <td>Antibiotic Dose</td>
        <td>
          An integer in <code class="notranslate">[0, 3]</code>
        </td>
        <td>
          This variable represents the number of courses of antibiotics taken during the first
          year of life. The maximum value is 3, since the likelihood of taking more than 3 courses
          of antibiotics in the first year of life is very low. The upper value of 3 indicates
          3 or more courses of antibiotics taken during the first year of life.
        </td>
      </tr>
    </tbody>
    </table>


Formula
---------------------------------------

Before we begin, let us define some terms. We have two risk factors we are interested in:
family history and antibiotic use. There are :math:`2 * 4 = 8` possible combinations of these two
risk factors:


.. raw:: html

  <table class="table">
    <thead>
      <tr>
          <th>&lambda;</th>
          <th>Family History</th>
          <th>Antibiotic Dose</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code class="notranslate">0</code></td>
        <td><code class="notranslate">0</code></td>
        <td><code class="notranslate">0</code></td>
      </tr>
      <tr>
        <td><code class="notranslate">1</code></td>
        <td><code class="notranslate">1</code></td>
        <td><code class="notranslate">0</code></td>
      </tr>
      <tr>
        <td><code class="notranslate">2</code></td>
        <td><code class="notranslate">0</code></td>
        <td><code class="notranslate">1</code></td>
      </tr>
      <tr>
        <td><code class="notranslate">3</code></td>
        <td><code class="notranslate">1</code></td>
        <td><code class="notranslate">1</code></td>
      </tr>
      <tr>
        <td><code class="notranslate">4</code></td>
        <td><code class="notranslate">0</code></td>
        <td><code class="notranslate">2</code></td>
      </tr>
      <tr>
        <td><code class="notranslate">5</code></td>
        <td><code class="notranslate">1</code></td>
        <td><code class="notranslate">2</code></td>
      </tr>
      <tr>
        <td><code class="notranslate">6</code></td>
        <td><code class="notranslate">0</code></td>
        <td><code class="notranslate">3</code></td>
      </tr>
      <tr>
        <td><code class="notranslate">7</code></td>
        <td><code class="notranslate">1</code></td>
        <td><code class="notranslate">3</code></td>
      </tr>
    </tbody>
    </table>


We can represent each combination as a vector of the form:

.. math::

  \begin{bmatrix}
    f_{\lambda} \\
    d_{\lambda}
  \end{bmatrix}

where :math:`f_{\lambda}` is the family history and :math:`d_{\lambda}` is the antibiotic dose.

We next define the odds ratio for a given risk factor as:

.. math::

  \omega(r=k) = \dfrac{P(A = 1 | r = k)}{P(A = 1 | r = 0)}

where :math:`A` is the asthma incidence or prevalence and :math:`r` is the risk factor.
To combine odds ratios, we have:

.. math::

  \omega_{\lambda} &= \omega(f = f_{\lambda}, d = d_{\lambda}) \\
  &= \dfrac{P(A = 1 | f = f_{\lambda}, d = d_{\lambda})}{P(A = 1 | f = 0, d = 0)} \\
  &= \dfrac{P(A = 1 | f = f_{\lambda})}{P(A = 1 | f = 0)} \cdot 
    \dfrac{P(A = 1 | d = d_{\lambda})}{P(A = 1 | d = 0)} \\
  &= \omega(f = f_{\lambda}) \cdot \omega(d = d_{\lambda})

Since these are multiplicative, the log of the odds ratios is additive:

.. math::

  \log(\omega_{\lambda}) = \log(\omega(f = f_{\lambda})) + 
    \log(\omega(d = d_{\lambda}))

We can now define our formula for the calibration model:

.. math::

  \zeta_{\lambda}^{(i)} = \sigma\left(\beta_{\eta} + \log(\omega_{\lambda}^{(i)}) + \alpha\right)

where:

* :math:`\beta_{\eta} = \sigma^{-1}(\eta^{(i)})` is determined by the output of the first model
* :math:`\eta^{(i)}`, defined above, is the predicted incidence or prevalence from the first model
* :math:`\sigma(x)` is the logistic function
* :math:`\alpha = \sum_{\lambda=1}^{n} p(\lambda) \cdot \beta_{\lambda}` is the
  correction / calibration term for either the incidence or prevalence
* :math:`\zeta^{(i)} = \sum_{\lambda=0}^{n} p(\lambda) \zeta_{\lambda}^{(i)}` is predicted
  asthma prevalence / incidence for the model. We want this to be as close as possible to
  :math:`\eta^{(i)}`
* :math:`\zeta_{\lambda}^{(i)}` is the predicted asthma incidence or prevalence from the model
  for the risk factor combination indexed by :math:`\lambda`
* :math:`p(\lambda)` is the probability of the risk factor combination indexed by :math:`\lambda`

Let's break this formula down:

Antibiotic Risk Factors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The antibiotic terms were fit by Lee et al. :cite:`lee2024`, using a random effects meta-regression
model:

.. math::

  \log(\omega(d_{\lambda})) =
    \begin{cases}
      \beta_{\text{abx_0}} + 
      \beta_{\text{abx_age}} \cdot \text{min}(a^{(i)}, 7) +
      \beta_{\text{abx_dose}} \cdot \text{min}(d^{(i)}, 3)
      && d^{(i)} > 0 \text{ and } a^{(i)} \leq 7 \\ \\
      0 && \text{otherwise}
    \end{cases}

where:

* :math:`\beta_{\text{abx_xxx}}` is a constant coefficient
* :math:`a^{(i)}` is the age
* :math:`d^{(i)}` is the number of courses of antibiotics taken during the first year of life

The beta coefficients were found to be:

* :math:`\beta_{\text{abx_0}} = 1.82581`
* :math:`\beta_{\text{abx_age}} = 0.2253`
* :math:`\beta_{\text{abx_dose}} = 0.0531475`

Family History Risk Factors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The family history terms were fit using the ``CHILD Study`` data, in the paper by Patrick et al.
:cite:`patrick2020`, using logistic regression:

.. math::

  \log(\omega(f_{\lambda})) = 
    \beta_{\text{hx_0}} \cdot f^{(i)} + 
    \beta_{\text{hx_age}} \cdot (\text{min}(a^{(i)}, 5) - 3) \cdot f^{(i)}

where:

* :math:`\beta_{\text{hx_xxx}}` is a constant coefficient
* :math:`a^{(i)}` is the age
* :math:`f^{(i)}` is the family history of asthma; 1 = at least one parent has asthma,
  0 = neither parent has asthma

The beta coefficients were found to be:

* :math:`\beta_{\text{hx_0}} = \log(1.13)`
* :math:`\beta_{\text{hx_age}} = \dfrac{\log(2.4) - \log(1.13)}{2}`


So, the only unknown term in our formula is the correction term :math:`\alpha`. To solve this,
we separate the formulae for incidence and prevalence. We will begin with prevalence.

Solving for the Correction Term: Prevalence
--------------------------------------------

.. math::

  \zeta_{\text{prev}} &= \sum_{\lambda=0}^{n} p(\lambda) \zeta_{\lambda} \\
  &= \sum_{\lambda=0}^{n} p(\lambda) \sigma(\beta_{\eta} + \log(\omega_{\lambda}) - \alpha) 


We want to find a correction term :math:`\alpha` such that the predicted asthma prevalence
:math:`\zeta` is as close as possible to the predicted asthma prevalence :math:`\eta`. To do this,
we use the ``Broyden-Fletcher-Goldfarb-Shanno (BFGS)`` algorithm to minimize the absolute
difference between :math:`\zeta` and :math:`\eta`.


Solving for the Correction Term: Incidence
--------------------------------------------

In our model, asthma incidence is defined as the number of new diagnoses between the previous year
and the current year, divided by the total population. To calibrate the incidence, we first
find the calibrated prevalence for the previous year:

.. math::

  \zeta_{\text{prev}}(t-1) &= \sum_{\lambda=0}^{n} p(\lambda, t-1) \zeta_{\text{prev}, \lambda}(t-1) \\
  &= \sum_{\lambda=0}^{n} p(\lambda, t-1) \sigma(\beta_{\eta} + \log(\omega_{\lambda}) - \alpha)

Now, what we want to find is the joint probability of each risk factor combination,
:math:`p(\lambda, A = 0 | t-1)`, for the population without asthma.

.. math::

  P(\lambda, A = 0) = P(A = 0 | \lambda) \cdot P(\lambda)

Now, we must have:

.. math::

  P(A = 0 | \lambda) = 1 - P(A = 1 | \lambda) = 1 - \zeta_{\text{prev}, \lambda}(t-1)

So, we can rewrite the joint probability as:

.. math::

  p(\lambda, A = 0 | t-1) = (1 - \zeta_{\text{prev}, \lambda}(t-1)) \cdot p(\lambda, t-1)


Next, we find the calibrated asthma incidence for the current year:

.. math::

  \zeta_{\text{inc}}(t) &= \sum_{\lambda=0}^{n} p(\lambda, A = 0 | t-1) \zeta_{\text{inc}, \lambda}(t) \\
  &= \sum_{\lambda=0}^{n} p(\lambda, A = 0 | t-1) \sigma(\beta_{\eta} + \log(\omega_{\lambda}) - \alpha)


where we recall that:

* :math:`\beta_{\eta} = \sigma^{-1}(\eta^{(i)}(t))` is determined by the output of the first model
* :math:`\eta^{(i)}(t)`, defined above, is the predicted incidence from the first model
* :math:`\alpha = \sum_{\lambda=1}^{n} p(\lambda, A = 0 | t-1) \cdot \beta_{\lambda}` is the
  correction / calibration term for the incidence
* :math:`\zeta^{(i)} = \sum_{\lambda=0}^{n} p(\lambda, A = 0 | t-1) \zeta_{\lambda}^{(i)}` is the
  predicted asthma incidence for the model. We want this to be as close as possible to
  :math:`\eta^{(i)}`
* :math:`\zeta_{\lambda}^{(i)}` is the predicted asthma incidence from the model for the
  risk factor combination indexed by :math:`\lambda`
* :math:`p(\lambda, A = 0 | t-1)` is the joint probability of the risk factor combination
  indexed by :math:`\lambda`, for a person who did not have asthma at time :math:`t-1`


We again want to find a correction term :math:`\alpha` such that the predicted asthma incidence
:math:`\zeta` is as close as possible to the asthma incidence from the first model, :math:`\eta`.
To do this, we use the ``Broyden-Fletcher-Goldfarb-Shanno (BFGS)`` algorithm to minimize the
absolute difference between :math:`\zeta` and :math:`\eta`.


Optimizing the Beta Parameters
--------------------------------------------

Before we begin, let us first define what we mean by a ``contingency table``. A contingency
table is a table that displays two categorical variables and their joint frequency distribution.
For example, suppose we have two categorical variables: ``smoking`` and ``lung cancer``, with
``n = 300`` patients in total:

.. raw:: html

    <table class="table">
        <thead>
        <tr>
            <th></th>
            <th>lung cancer</th>
            <th>no lung cancer</th>
            <th></th>
        </tr>
        </thead>
        <tbody>
        <tr>
            <td>smoker</td>
            <td><code class="notranslate">a0 = 100</code></td>
            <td><code class="notranslate">b0 = 50</code></td>
            <td><code class="notranslate">n1 = 150</code></td>
        </tr>
        <tr>
            <td>non-smoker</td>
            <td><code class="notranslate">c0 = 25</code></td>
            <td><code class="notranslate">d0 = 125</code></td>
            <td></td>
        </tr>
        <tr>
            <td></td>
            <td><code class="notranslate">n2 = 125</code></td>
            <td></td>
            <td><code class="notranslate">n = 300</code></td>
        </tr>
        </tbody>
    </table>

The variables ``n1`` and ``n2`` are called the marginal totals:

.. math::

    n_1 &= a_0 + b_0 = \text{total number of patients with lung cancer} \\
    n_2 &= c_0 + d_0 = \text{total number of patients who smoke}

The variable ``n`` is the total number of patients:

.. math::

    n = a_0 + b_0 + c_0 + d_0 = \text{total number of patients}

In our model, we want to compute the contingency table for the risk factor combinations
:math:`\lambda` and the asthma diagnosis.


Past Contingency Table
^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

    <table class="table">
        <thead>
        <tr>
            <th></th>
            <th>variable 2, outcome +</th>
            <th>variable 2, outcome -</th>
            <th></th>
        </tr>
        </thead>
        <tbody>
        <tr>
            <td>variable 1, outcome +</td>
            <td><code class="notranslate">a0</code></td>
            <td><code class="notranslate">b0</code></td>
            <td><code class="notranslate">n1</code></td>
        </tr>
        <tr>
            <td>variable 1, outcome -</td>
            <td><code class="notranslate">c0</code></td>
            <td><code class="notranslate">d0</code></td>
            <td></td>
        </tr>
        <tr>
            <td></td>
            <td><code class="notranslate">n2</code></td>
            <td></td>
            <td><code class="notranslate">n</code></td>
        </tr>
        </tbody>
    </table>


We want to calculate :math:`a_0`, :math:`b_0`, :math:`c_0`, and :math:`d_0` using :math:`n_1`,
:math:`n_2`, :math:`n`, and :math:`\omega_{\lambda}`. Now, we have the probabilities of each of the
risk factor combinations, :math:`p(\lambda)`, but for the contingency table, we only want to
consider one risk factor combination at a time. To do this, we compute the conditional probability:

.. math::

    p(\Lambda = \lambda ~|~ \Lambda \in \{0, \lambda\}) = 
      \dfrac{p(\Lambda = \lambda)}{p(\Lambda = \lambda) + p(\Lambda = 0)}

To obtain :math:`n_1`, the number of people with risk factor combination :math:`\lambda` with or
without an asthma diagnosis, we multiply the conditional probability by the total population
:math:`n`:

.. math::
    n_1 = p(\Lambda = \lambda | \Lambda \in \{0, \lambda\}) \cdot n

To obtain :math:`n_2`, the number of people diagnosed with asthma with or without risk factor
combination :math:`\lambda`:

.. math::
    n_2 = (1 - p(\Lambda = \lambda | \Lambda \in \{0, \lambda\})) \cdot \zeta_{\text{prev}, 0}(t=0) \cdot n +
      p(\Lambda = \lambda | \Lambda \in \{0, \lambda\}) \cdot \zeta_{\text{prev}, \lambda}(t=0) \cdot n

From this, we can calculate the values for the contingency table:

.. math::

    b_0 &= n_1 - a_0 \\
    c_0 &= n_2 - a_0 \\
    d_0 &= n - n_1 - n_2 - a_0

To obtain :math:`a_0`, we follow the methods described in the paper :cite:`dipietrantonj2006`.
See :doc:`conv_2x2 <../dev/api/data_generation/leap.data_generation.utils>` for the Python
implementation of this method.

Current Contingency Table: Reassessment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

According to our model, an asthma diagnosis is not static; a patient may be diagnosed with asthma
and then later be reassessed as not having asthma. We would like to compute the updated contingency
table:

.. raw:: html

    <table class="table">
        <thead>
        <tr>
            <th></th>
            <th>asthma, outcome +</th>
            <th>asthma, outcome -</th>
            <th></th>
        </tr>
        </thead>
        <tbody>
        <tr>
            <td>risk factor <code class="notranslate">λ</code>, outcome +</td>
            <td><code class="notranslate">a1_ra</code></td>
            <td><code class="notranslate">b1_ra</code></td>
            <td><code class="notranslate">n1</code></td>
        </tr>
        <tr>
            <td>risk factor <code class="notranslate">λ</code>, outcome -</td>
            <td><code class="notranslate">c1_ra</code></td>
            <td><code class="notranslate">d1_ra</code></td>
            <td></td>
        </tr>
        <tr>
            <td></td>
            <td><code class="notranslate">n2</code></td>
            <td></td>
            <td><code class="notranslate">n</code></td>
        </tr>
        </tbody>
    </table>

To calculate the updated contingency table, we have:

.. math::
    a_1 &= a_0 \cdot \rho \\
    b_1 &= a_0 \cdot (1 - \rho) \\
    c_1 &= c_0 \cdot \rho \\
    d_1 &= c_0 \cdot (1 - \rho)

where:

.. raw:: html

    <table class="table">
        <thead>
        <tr>
            <th></th>
            <th>Risk Factors</th>
            <th>t=0</th>
            <th>t=1</th>
        </tr>
        </thead>
        <tbody>
        <tr>
            <td><code class="notranslate">a1_ra</code></td>
            <td><code class="notranslate">λ</code></td>
            <td>has asthma diagnosis</td>
            <td>reassessed: has asthma diagnosis</td>
        </tr>
        <tr>
            <td><code class="notranslate">b1_ra</code></td>
            <td><code class="notranslate">λ</code></td>
            <td>has asthma diagnosis</td>
            <td>reassessed: no asthma diagnosis</td>
        </tr>
        <tr>
            <td><code class="notranslate">c1_ra</code></td>
            <td>None</td>
            <td>has asthma diagnosis</td>
            <td>reassessed: has asthma diagnosis</td>
        </tr>
        <tr>
            <td><code class="notranslate">d1_ra</code></td>
            <td>None</td>
            <td>has asthma diagnosis</td>
            <td>reassessed: no asthma diagnosis</td>
        </tr>
        </tbody>
    </table>

* :math:`a_1` is the proportion of the population with risk factor combination :math:`\lambda`
  who had an asthma diagnosis at :math:`t=0` and still have it at :math:`t=1`
* :math:`b_1` is the proportion of the population with risk factor combination :math:`\lambda`
  who had an asthma diagnosis at :math:`t=0` but no longer have it at :math:`t=1`
* :math:`c_1` is the proportion of the population with no risk factors (:math:`\lambda = 0`)
  who had an asthma diagnosis at :math:`t=0` and still have it at :math:`t=1`
* :math:`d_1` is the proportion of the population with no risk factors (:math:`\lambda = 0`)
  who had an asthma diagnosis at :math:`t=0` but no longer have it at :math:`t=1`
* :math:`\rho` is the probability that a person would be reassessed as having an asthma diagnosis
  at :math:`t=1` given that they had an asthma diagnosis at :math:`t=0`


Current Contingency Table: New Diagnosis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the reassessment table, we considered only the patients who were diagnosed with asthma.
We will now consider those who were not diagnosed with asthma:

.. raw:: html

    <table class="table">
        <thead>
        <tr>
            <th></th>
            <th>asthma, outcome +</th>
            <th>asthma, outcome -</th>
            <th></th>
        </tr>
        </thead>
        <tbody>
        <tr>
            <td>risk factor <code class="notranslate">λ</code>, outcome +</td>
            <td><code class="notranslate">a1_dx</code></td>
            <td><code class="notranslate">b1_dx</code></td>
            <td><code class="notranslate">n1</code></td>
        </tr>
        <tr>
            <td>risk factor <code class="notranslate">λ</code>, outcome -</td>
            <td><code class="notranslate">c1_dx</code></td>
            <td><code class="notranslate">d1_dx</code></td>
            <td></td>
        </tr>
        <tr>
            <td></td>
            <td><code class="notranslate">n2</code></td>
            <td></td>
            <td><code class="notranslate">n</code></td>
        </tr>
        </tbody>
    </table>


To calculate the updated contingency table, we have:

.. math::
    a_{1, \text{dx}} &= b_0 \cdot \zeta_{\text{inc}, \lambda}(t=1) \\
    b_{1, \text{dx}} &= b_0 \cdot (1 - \zeta_{\text{inc}, \lambda}(t=1)) \\
    c_{1, \text{dx}} &= d_0 \cdot \zeta_{\text{inc}, 0}(t=1) \\
    d_{1, \text{dx}} &= d_0 \cdot (1 - \zeta_{\text{inc}, 0}(t=1))

where:

.. raw:: html

    <table class="table">
        <thead>
        <tr>
            <th></th>
            <th>Risk Factors</th>
            <th>t=0</th>
            <th colspan="3">t=1</th>
        </tr>
        <tr>
            <th></th>
            <th></th>
            <th></th>
            <th>incidence</th>
            <th>net</th>
        </thead>
        <tbody>
        <tr>
            <td><code class="notranslate">a1_dx</code></td>
            <td><code class="notranslate">λ</code></td>
            <td>no asthma diagnosis</td>
            <td>new asthma diagnosis</td>
            <td>asthma</td>
        </tr>
        <tr>
            <td><code class="notranslate">b1_dx</code></td>
            <td><code class="notranslate">λ</code></td>
            <td>no asthma diagnosis</td>
            <td>no new asthma diagnosis</td>
            <td>no asthma</td>
        </tr>
        <tr>
            <td><code class="notranslate">c1_dx</code></td>
            <td>None</td>
            <td>no asthma diagnosis</td>
            <td>new asthma diagnosis</td>
            <td>asthma</td>
        </tr>
        <tr>
            <td><code class="notranslate">d1_dx</code></td>
            <td>None</td>
            <td>no asthma diagnosis</td>
            <td>no new asthma diagnosis</td>
            <td>no asthma</td>
        </tr>
        </tbody>
    </table>

* :math:`a_{1, \text{dx}}` is the proportion of the population with risk factor combination :math:`\lambda`
  who didn't have an asthma diagnosis at :math:`t=0` and were diagnosed at :math:`t=1`
  :math:`\rightarrow` have asthma at :math:`t=1`
* :math:`b_{1, \text{dx}}` is the proportion of the population with risk factor combination :math:`\lambda`
  who didn't have an asthma diagnosis at :math:`t=0` and were not diagnosed with asthma at :math:`t=1`,
  :math:`\rightarrow` don't have asthma at :math:`t=1`
* :math:`c_{1, \text{dx}}` is the proportion of the population with no risk factors (:math:`\lambda = 0`)
  who didn't have an asthma diagnosis at :math:`t=0` and were diagnosed at :math:`t=1`
  :math:`\rightarrow` have asthma at :math:`t=1`
* :math:`d_{1, \text{dx}}` is the proportion of the population with no risk factors (:math:`\lambda = 0`)
  who didn't have an asthma diagnosis at :math:`t=0` and were not diagnosed with asthma at
  :math:`t=1`, :math:`\rightarrow` don't have asthma at :math:`t=1`


Current Contingency Table
^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, we can compute the contingency table for the current year, :math:`t=1`:

.. raw:: html

    <table class="table">
        <thead>
        <tr>
            <th></th>
            <th>asthma, outcome +</th>
            <th>asthma, outcome -</th>
            <th></th>
        </tr>
        </thead>
        <tbody>
        <tr>
            <td>risk factor <code class="notranslate">λ</code>, outcome +</td>
            <td><code class="notranslate">a1</code></td>
            <td><code class="notranslate">b1</code></td>
            <td><code class="notranslate">n1</code></td>
        </tr>
        <tr>
            <td>risk factor <code class="notranslate">λ</code>, outcome -</td>
            <td><code class="notranslate">c1</code></td>
            <td><code class="notranslate">d1</code></td>
            <td></td>
        </tr>
        <tr>
            <td></td>
            <td><code class="notranslate">n2</code></td>
            <td></td>
            <td><code class="notranslate">n</code></td>
        </tr>
        </tbody>
    </table>


where:

.. math::
    a_1 &= a_{1, \text{ra}} + a_{1, \text{dx}} \\
    b_1 &= b_{1, \text{ra}} + b_{1, \text{dx}} \\
    c_1 &= c_{1, \text{ra}} + c_{1, \text{dx}} \\
    d_1 &= d_{1, \text{ra}} + d_{1, \text{dx}}

From these values, we can compute the odds ratio:

.. math::
    \Omega = \dfrac{a_1 \cdot d_1}{b_1 \cdot c_1}


Optimization
^^^^^^^^^^^^^^^^^

We want to minimize the difference between the predicted odds ratio :math:`\Omega` and the
observed odds ratio :math:`\omega_{\lambda}`.

.. math::
    \sum_{i=1}^{N}\sum_{\lambda=1}^{n} 
      \dfrac{\left| \log(\Omega^{(i)}) - \log(\omega_{\lambda}^{(i)}) \right|}{N}