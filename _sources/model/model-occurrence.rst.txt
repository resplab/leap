Occurrence Model
================

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


Since our model projects into the future, we
would like to be able to extend this data beyond ``2019``. Our model also makes predictions at
1-year age intervals, not 5-year age intervals. To obtain these projections, we use a
``Generalized Linear Model (GLM)``. A ``GLM`` is a type of regression analysis which is a
generalized form of linear regression. See :doc:`model-glm` for more information on ``GLMs``.

When fitting a ``GLM``, first you must choose a distribution for the ``response variable``. In our
case, the response variable is the asthma prevalence or incidence. Incidence and prevalence are
counts of the number of people diagnosed with asthma and the number of people with asthma,
respectively, in a given time interval (a year, in our case). Since these are counts, we need a
discrete probability distribution. The ``Poisson distribution`` is a good choice for our data.

.. math::

    P(Y = y) = p(y; \mu^{(i)}) = \dfrac{(\mu^{(i)})^{y} ~ e^{-\mu^{(i)}}}{y!}

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