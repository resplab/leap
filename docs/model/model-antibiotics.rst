.. _antibiotic_exposure_model:

Antibiotic Exposure Model
==========================

In this section, we will describe the model used to predict the number of antibiotics prescribed
to infants in their first year of life. This model will be incorporated into the risk factors
for developing asthma later in life.

Datasets
*********

We obtained data from the ``BC Ministry of Health`` on the number of antibiotics prescribed to
infants in their first year of life. The data is available for the years ``2000`` to ``2018``. The
data is formatted as follows:

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
          format <code>XXXX</code>, e.g <code>2000</code>, range <code>[2000, 2018]</code>
        </td>
      </tr>
        <td><code class="notranslate">sex</code></td>
        <td>
          <code class="notranslate">int</code>
        </td>
        <td>
          1 = Female, 2 = Male
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">n_abx</code></td>
        <td>
          <code class="notranslate">int</code>
        </td>
        <td>
          The total number of antibiotic courses prescribed to infants in BC in a given year.
          Note that this is not per infant, it is the total for all infants.
        </td>
      </tr>
    </tbody>
    </table>

Since the ``n_abx`` column gives us the *total* number of antibiotics prescribed, we need to use
population data to convert this to a *per infant* value. We obtained population data from the
``StatCan`` census data (the same that is used in the ``Birth`` module):

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
          format <code>XXXX</code>, e.g <code>2000</code>, range <code>[1999, 2021]</code>
        </td>
      </tr>
        <td><code class="notranslate">sex</code></td>
        <td>
          <code class="notranslate">int</code>
        </td>
        <td>
          <code>"M"</code> = male, <code>"F"</code> = female
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">n_birth</code></td>
        <td>
          <code class="notranslate">int</code>
        </td>
        <td>
          The total number of births in BC in a given year.
        </td>
      </tr>
    </tbody>
    </table>


Model: Generalized Linear Model - Negative Binomial
****************************************************

Since our model projects into the future, we would like to be able to extend this data beyond
``2018``. To obtain these projections, we use a ``Generalized Linear Model (GLM)``. A ``GLM`` is a
type of regression analysis which is a generalized form of linear regression.
See :doc:`model-glm` for more information on ``GLMs``.


Probability Distribution
---------------------------------------

When fitting a ``GLM``, first you must choose a distribution for the ``response variable``. In our
case, the response variable is the number of antibiotics prescribed during the first year of life.
The number of antibiotics prescribed is a count variable, in a given time interval
(a year, in our case). Since it is count data, we need a discrete probability distribution.
The ``Poisson distribution`` is a good choice for our data, but it has some limitations. The
``Poisson distribution`` assumes that the mean and variance are equal, i.e:

.. math::
  
  \mu = \sigma^2
  
However, in our data, the variance is greater than the mean. This is a common problem in count data,
and it is called ``overdispersion``. The ``Negative Binomial`` distribution is a generalization of
the ``Poisson`` distribution that allows for overdispersion. The ``Negative Binomial`` distribution
has an extra parameter, :math:`\theta`, which controls the amount of overdispersion. Typically,
the distribution is written as:

.. math::

    P(Y = k; r, p) := \binom{k+r-1}{k}(1-p)^k p^r

where:

* :math:`k` is the number of failures before :math:`r` successes occur
* :math:`p` is the probability of a success
* :math:`r` is the number of successes

We can reparametrize this with :math:`\mu` and :math:`\theta` using the following equations:

.. math::

    p &= \dfrac{\mu}{\sigma^2} \\
    r &= \dfrac{\mu^2}{\sigma^2 - \mu} \\
    \sigma^2 &= \mu + \dfrac{\mu^2}{\theta}

Doing some algebra, we have:

.. math::

    p &= \dfrac{\mu}{\sigma^2} = \dfrac{\mu}{\mu + \dfrac{\mu^2}{\theta}} = \dfrac{\theta}{\theta + \mu} \\
    r &= \dfrac{\mu^2}{\sigma^2 - \mu} = \dfrac{\mu^2}{\mu + \dfrac{\mu^2}{\theta} - \mu} 
    = \dfrac{\mu^2}{\dfrac{\mu^2}{\theta}} 
    = \theta

Letting :math:`y = k`, we have:

.. math::

    P(Y = y; \mu, \theta) &= \binom{y + \theta - 1}{y}
        \left(1-\dfrac{\theta}{\theta + \mu}\right)^y 
        \left(\dfrac{\theta}{\theta + \mu}\right)^{\theta} \\
    &= \binom{y + \theta - 1}{y}
        \left(\dfrac{\theta + \mu - \theta}{\theta + \mu}\right)^y 
        \left(\dfrac{\theta}{\theta + \mu}\right)^{\theta} \\
    &= \binom{y + \theta - 1}{y}
        \dfrac{\mu^y \theta^{\theta}}{(\theta + \mu)^{y+\theta}}


We added an upper bound on the mean parameter to prevent unrealistic extrapolation:

.. math::

    \mu^{(i)} = E(Y | X = x^{(i)}) \leq 0.05

In other words, we are saying that the mean number of antibiotics prescribed to an infant in their
first year of life is less than or equal to ``0.05``.

So we have:

.. math::

    P(Y = y; \mu, \theta) &= p(y; \mu^{(i)}, \theta^{(i)}) = 
        \binom{y + \theta^{(i)} - 1}{y}
        \dfrac{(\mu^{(i)})^y (\theta^{(i)})^{\theta^{(i)}}}{(\theta^{(i)} + \mu^{(i)})^{y + \theta^{(i)}}} \\
    \mu^{(i)} &= \text{max}(0.05, \mu^{(i)})



Link Function
-----------------

We also need to choose a ``link function``. Recall that the link function :math:`g(\mu^{(i)})`
is used to relate the mean to the predicted value :math:`\eta^{(i)}`:

.. math::

    g(\mu^{(i)}) &= \eta^{(i)} \\
    \mu^{(i)} &= E(Y | X = x^{(i)})

How do we choose a link function? Well, we are free to choose any link function we like, but there
are some constraints. For example, in the Negative Binomial distribution, the mean is always
``>= 0``. However, :math:`\eta^{(i)}` can be any real number. Therefore, we need a link function
that maps real numbers to non-negative numbers. The ``log link function`` is a good choice for this:

.. math::

    g(\mu^{(i)}) = \log(\mu^{(i)}) = \eta^{(i)}


Formula
-----------------

Now that we have our distribution and link function, we need to decide on a formula for
:math:`\eta^{(i)}`. We are permitted to use linear combinations of functions of the features
in our dataset.

For our dataset, we want a formula using ``sex`` and ``year``. Since prescribing practices change
over time, and since infections requiring antibiotic prescriptions also change over time,
we should include year in our formula. We also want to include sex, since there are sex differences
in antibiotic prescriptions.

There is an additional factor specific to BC regulations. In 2005, the BC government introduced
an antibiotic conservation program, which reduced the number of antibiotics prescribed
:cite:`mamun2019`. It stands to reason that the formula may change before and after 2005. To
account for this, we will introduce a ``Heaviside step function``, which returns ``0`` for values
below a given threshold, and ``1`` for values above the threshold. In our case, the threshold
is ``2005``.


.. math::

    \eta^{(i)} = \beta_0 + \beta_s \cdot s^{(i)} + \beta_t \cdot t^{(i)} +
        \beta_h \cdot H(t^{(i)} - 2005) + 
        \beta_{th} \cdot t \cdot H(t^{(i)} - 2005)

where:

* :math:`s^{(i)}` is the sex of the infant
* :math:`t^{(i)}` is the year of birth of the infant
* :math:`H(t^{(i)} - 2005)` is the ``Heaviside step function``, which is ``0`` for years before
  ``2005`` and ``1`` for years after ``2005``


