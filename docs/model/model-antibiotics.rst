.. _antibiotic_exposure_model:

Antibiotic Exposure Model
==========================

In this section, we will describe the model used to predict the number of antibiotics prescribed
to infants in their first year of life. This model will be incorporated into the risk factors
for developing asthma later in life.

Input Datasets
**************

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
population data to convert this to a *per infant* value. We obtained population data from
`Table 17-10-00005-01 from Statistics Canada
<https://www150.statcan.gc.ca/t1/tbl1/en/cv.action?pid=1710000501>`_,
the same past-data source used in the :ref:`birth-model`:

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
        <td><code class="notranslate">timepoint</code></td>
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
          The total number of births in BC in a given time interval.
        </td>
      </tr>
    </tbody>
    </table>


Merged Dataset: InfantAbxBC.csv
*********************************

The two input datasets are merged on ``year`` and ``sex`` to produce
`InfantAbxBC.csv <https://github.com/resplab/leap/blob/main/leap/processed_data/InfantAbxBC.csv>`_,
saved in the ``processed_data`` directory. This combined dataset is what the GLM is fitted on:

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
        <td><code class="notranslate">int</code></td>
        <td>
          Calendar year, range <code>[2000, 2018]</code>
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">sex</code></td>
        <td><code class="notranslate">str</code></td>
        <td>
          <code>"Female"</code> or <code>"Male"</code>
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">N</code></td>
        <td><code class="notranslate">int</code></td>
        <td>
          Total number of births in BC for the given year and sex.
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">N_abx</code></td>
        <td><code class="notranslate">int</code></td>
        <td>
          Total number of antibiotic courses dispensed to infants in BC for the given year and sex.
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">rate</code></td>
        <td><code class="notranslate">float</code></td>
        <td>
          Antibiotic prescription rate per 1,000 births (<code>N_abx / N * 1000</code>).
        </td>
      </tr>
    </tbody>
  </table>


Model: Generalized Linear Model - Negative Binomial
****************************************************

Since our model projects into the future, we would like to be able to extend this data beyond
``2018``. To obtain these projections, we use a ``Generalized Linear Model (GLM)``. A ``GLM`` is a
type of regression analysis which is a generalized form of linear regression.

The response variable — number of antibiotic courses prescribed during the first year of life —
is overdispersed count data, so we use the ``Negative Binomial`` distribution with a ``log``
:ref:`link function <glm-link-function>`. See :ref:`negative-binomial-glm` in
:doc:`model-statistical-background` for the full distributional derivation and motivation.

The mean parameter has a lower bound to prevent unrealistically small extrapolations into future
years:

.. math::

    \mu_i = \max\!\left(\mu_i,\; 0.05\right)

In other words, the predicted mean number of antibiotic courses per infant is at least ``0.05``.


Formula
-------

The birth data enters the model as a log offset on ``n_birth``, so the GLM predicts a
per-capita rate directly rather than a raw count.

Since prescribing practices change over time, and since infections requiring antibiotic
prescriptions also change over time, birth year is included in the formula. Sex is also
included, since there are known sex differences in antibiotic prescriptions.

There is an additional factor specific to BC regulations. In 2005, the BC government introduced
an antibiotic conservation program, which reduced the number of antibiotics prescribed
:cite:`mamun2019`. To account for this structural break, the formula includes a
``Heaviside step function`` :math:`H`, which is ``0`` for years before ``2005`` and ``1``
for years from ``2005`` onward.

.. math::

    \log(\mu_i) = \beta_0
        + \beta_{\text{sex}} \cdot s_i
        + \beta_{\text{year}} \cdot t_i
        + \beta_{\text{2005}} \cdot H(t_i - 2005)
        + \beta_{\text{year,2005}} \cdot t_i \cdot H(t_i - 2005)

where:

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Coefficient
     - Term
     - Description
   * - :math:`\beta_0`
     - :math:`1`
     - intercept
   * - :math:`\beta_{\text{sex}}`
     - :math:`s_i`
     - sex main effect
   * - :math:`\beta_{\text{year}}`
     - :math:`t_i`
     - birth year main effect
   * - :math:`\beta_{\text{2005}}`
     - :math:`H(t_i - 2005)`
     - Heaviside step at 2005
   * - :math:`\beta_{\text{year,2005}}`
     - :math:`t_i \cdot H(t_i - 2005)`
     - birth year × Heaviside interaction

And :math:`s_i` is the sex, :math:`t_i` is the birth year, and :math:`H` is the Heaviside
step function.


Usage in Simulation
********************

Once fitted, the β coefficients and θ (the Negative Binomial overdispersion parameter;
see :ref:`negative-binomial-glm`) are stored in ``config.json`` and used at runtime. When
an agent is initialised at birth, the simulation draws their antibiotic exposure count directly
from the Negative Binomial distribution. For an agent with sex :math:`s_i` and birth year
:math:`t_i`:

1. Compute the linear predictor :math:`\eta_i` using the formula above.
2. Convert to the mean: :math:`\mu_i = \max(\exp(\eta_i),\; 0.05)`.
3. Convert to the Negative Binomial success probability: :math:`p_i = \theta / (\theta + \mu_i)`.
4. Draw the exposure count: :math:`n_{\text{abx},i} \sim \text{NegBin}(\theta,\, p_i)`.

This count is fixed for the agent's lifetime and is capped at 3 courses when computing the
antibiotic exposure odds ratio :math:`\omega_{\text{abx}}` in the
:ref:`Asthma Occurrence Model <occurrence-model-2>`. See :doc:`model-simulation` for how
antibiotic exposure is assigned during agent initialisation.

The worked example below demonstrates steps 1–4 for a specific agent (sex = male,
birth year = 2008), and then visualises the resulting distribution across multiple birth
years and sexes.

Example
*******

.. include:: model-antibiotics.ipynb
    :parser: myst_nb.docutils_
