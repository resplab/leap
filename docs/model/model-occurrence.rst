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

  \zeta_{\lambda}^{(i)} = \sigma(\beta_{\eta} + \log(\omega_{\lambda}^{(i)}) + \alpha)

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

Let's break this formula down. The antibiotic terms were fit by Lee et al.
:cite:`lee2024`, using a random effects meta-regression model:

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

