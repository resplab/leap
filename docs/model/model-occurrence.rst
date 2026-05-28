.. _occurrence-model:

========================
Asthma Occurrence Model
========================

This document describes the asthma occurrence model, which is used to predict the incidence and
prevalence of asthma in British Columbia. The model is divided into two parts:

1. :ref:`occurrence-model-1`: A ``Generalized Linear Model (GLM)`` that predicts asthma incidence
   and prevalence based on age, sex, and year.
2. :ref:`occurrence-model-2`: A model that incorporates risk factors such as family history and
   antibiotic use during infancy to predict asthma incidence and prevalence, along with the
   results from the first model.

.. _occurrence-model-1:

Occurrence Model 1: Crude Occurrence
=====================================

In the first model, we will use data collected from the ``BC Ministry of Health`` on the
incidence and prevalence of asthma in British Columbia. We will use this data to fit a 
``Generalized Linear Model (GLM)`` to predict the incidence and prevalence of asthma
based on the age, sex, and year. However, asthma occurrence doesn't just depend on someone's age or
sex, but it also depends on risk factors such as family history and antibiotic use during
infancy. We will address these in the second model: :ref:`occurrence-model-2`.

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
To obtain these projections, we use a ``Generalized Linear Model (GLM)`` with a
``Poisson distribution`` and ``log link function``. Incidence and prevalence are counts of
people diagnosed with or living with asthma in a given year, making the Poisson distribution a
natural choice. See :doc:`model-glm` for more information on ``GLMs``, including the Poisson
distribution and log link function.

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

.. TODO: Why was this formula chosen?


.. math::

    \eta^{(i)} = \beta_0 + \beta_s \cdot s^{(i)} + \beta_t \cdot t^{(i)} + \beta_{ts} \cdot t^{(i)} \cdot s^{(i)}
        + \sum_{k=1}^{5} \left( \beta_k \cdot (a^{(i)})^k + \beta_{ks} \cdot (a^{(i)})^k \cdot s^{(i)} \right)


where:

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Coefficient
     - Term
     - Description
   * - :math:`\beta_0`
     - :math:`1`
     - intercept
   * - :math:`\beta_s`
     - :math:`s^{(i)}`
     - sex main effect
   * - :math:`\beta_t`
     - :math:`t^{(i)}`
     - year main effect
   * - :math:`\beta_{ts}`
     - :math:`t^{(i)} \cdot s^{(i)}`
     - year × sex interaction
   * - :math:`\beta_k` (:math:`k = 1, \ldots, 5`)
     - :math:`(a^{(i)})^k`
     - age polynomial terms
   * - :math:`\beta_{ks}` (:math:`k = 1, \ldots, 5`)
     - :math:`(a^{(i)})^k \cdot s^{(i)}`
     - age × sex interaction terms

And :math:`a^{(i)}` is the age, :math:`t^{(i)}` is the year, :math:`s^{(i)}` is the sex.

There are :math:`4 + 5 + 5 = 14` coefficients in the incidence model.


Next we have the ``prevalence``. We again want a formula using ``age``, ``sex``, and ``year``.
Since asthma prevalence depends on the number of people who have asthma, and this number changes
from year to year, we should include year in our formula. Asthma prevalence also depends on age,
so we should include age in our formula. Finally, there is a sex difference
in asthma incidence and hence prevalence, so we should include sex in our formula.


.. math::

    \begin{align}
    \eta^{(i)} &= \beta_0 + \beta_s \cdot s^{(i)} \\
        &+ \sum_{k=1}^{5} \left( \beta_k \cdot (a^{(i)})^k + \beta_{ks} \cdot (a^{(i)})^k \cdot s^{(i)} \right) \\
        &+ \sum_{\ell=1}^{2} \left( \beta_{t^\ell} \cdot (t^{(i)})^\ell + \beta_{t^\ell s} \cdot (t^{(i)})^\ell \cdot s^{(i)} \right) \\
        &+ \sum_{\ell=1}^{2} \sum_{k=1}^{5} \left( \beta_{k\ell} \cdot (a^{(i)})^k \cdot (t^{(i)})^\ell
        + \beta_{k\ell s} \cdot (a^{(i)})^k \cdot (t^{(i)})^\ell \cdot s^{(i)} \right)
    \end{align}


where:

.. list-table::
   :widths: 30 30 40
   :header-rows: 1

   * - Coefficient
     - Term
     - Description
   * - :math:`\beta_0`
     - :math:`1`
     - intercept
   * - :math:`\beta_s`
     - :math:`s^{(i)}`
     - sex main effect
   * - :math:`\beta_k` (:math:`k = 1, \ldots, 5`)
     - :math:`(a^{(i)})^k`
     - age polynomial terms
   * - :math:`\beta_{ks}` (:math:`k = 1, \ldots, 5`)
     - :math:`(a^{(i)})^k \cdot s^{(i)}`
     - age × sex interactions
   * - :math:`\beta_{t^\ell}` (:math:`\ell = 1, 2`)
     - :math:`(t^{(i)})^\ell`
     - year polynomial terms
   * - :math:`\beta_{t^\ell s}` (:math:`\ell = 1, 2`)
     - :math:`(t^{(i)})^\ell \cdot s^{(i)}`
     - year × sex interactions
   * - :math:`\beta_{k\ell}` (:math:`k = 1, \ldots, 5`,  :math:`\ell = 1, 2`)
     - :math:`(a^{(i)})^k \cdot (t^{(i)})^\ell`
     - age × year interactions
   * - :math:`\beta_{k\ell s}` (:math:`k = 1, \ldots, 5`, :math:`\ell = 1, 2`)
     - :math:`(a^{(i)})^k \cdot (t^{(i)})^\ell \cdot s^{(i)}`
     - age × year × sex interactions

And :math:`a^{(i)}` is the age, :math:`t^{(i)}` is the year, :math:`s^{(i)}` is the sex.

There are :math:`(1 + 1 + 5 + 5) + (2 + 2 + 10 + 10) = 36` coefficients in the prevalence model.


Assumptions
-----------------

The following assumptions are applied when generating predictions from the fitted model:

* **Age**: The training data contains 5-year age bands up to a maximum midpoint of 63 years.
  For ages greater than 63, incidence and prevalence are assumed to remain constant at the
  rates predicted for age 63.

* **Year**: Incidence and prevalence trends are predicted for the years ``2020–2025`` using
  the fitted GLM. From ``2025`` onwards, to the end of the model time horizon, incidence and
  prevalence are assumed to remain constant at the ``2025`` rates.


Processed Data
-----------------

The processed data produced by this model is stored in ``asthma_occurrence_predictions.csv``.
The data contains predicted asthma incidence and prevalence at 1-year age and year intervals,
for each sex. The variables are:

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
          Calendar year of the prediction
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">sex</code></td>
        <td><code class="notranslate">str</code></td>
        <td>
          <code class="notranslate">"F"</code> = Female,
          <code class="notranslate">"M"</code> = Male
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">age</code></td>
        <td><code class="notranslate">int</code></td>
        <td>
          Age in years
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">incidence</code></td>
        <td><code class="notranslate">float</code></td>
        <td>
          Predicted asthma incidence for the given year, age, and sex, per 100 people
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">prevalence</code></td>
        <td><code class="notranslate">float</code></td>
        <td>
          Predicted asthma prevalence for the given year, age, and sex, per 100 people
        </td>
      </tr>
    </tbody>
  </table>


.. _occurrence-model-2:

Occurrence Model 2: Risk Factors
=================================

Model 1 produces age-, sex-, and year-specific rates for the general population, but it treats
everyone in a stratum identically. In reality, individuals differ in ways that affect their
asthma risk — most notably whether a parent has asthma, and whether they received antibiotics
in early life. Model 2 builds on Model 1 by incorporating these risk factors, so that
the simulation can assign each agent an individualised probability of asthma incidence or
prevalence rather than a population average.

This is done in two phases:

* **Offline calibration** (run once during data generation): for every combination of age,
  sex, and year, a calibration term :math:`\alpha` is computed that ensures the
  population-weighted average of the risk-factor-adjusted probabilities still matches the
  target rates :math:`\bar{p}_{\text{prev}}` and :math:`\bar{p}_{\text{inc}}` from Model 1.
  The results are saved to
  ``asthma_occurrence_correction.csv``.

* **Online simulation** (at runtime): each agent's individual risk factors are combined with
  :math:`\alpha` from the lookup table to produce a personalised asthma probability on every
  simulated year of life.

Model: Risk Factors
******************************

We want to incorporate the effects of family history and antibiotic use on asthma incidence and
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
          <code class="notranslate">1</code>: at least one parent has asthma<br>
          <code class="notranslate">0</code>: neither parent has asthma
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


The effect of each risk factor on asthma risk is expressed as an **odds ratio (OR)** sourced
from external published studies. An odds ratio of :math:`\omega` means that a person with the
risk factor has :math:`\omega` times the odds of having asthma compared to a person without it.
We work in log-odds (logit) space rather than probability space because log-odds are additive:
the combined effect of independent risk factors is simply the sum of their individual
log-odds contributions. Because independent ORs are multiplicative, their logarithms are additive:

.. math::

  \log(\omega_{\lambda}) = \log(\omega_{\text{fhx}}) + \log(\omega_{\text{abx}})

Applying individual risk factor ORs directly to the Model 1 log-odds would shift the
population-weighted average probability away from the target :math:`\bar{p}_{\text{prev}}`. The calibration
term :math:`\alpha` corrects for this: it is a single scalar per (age, sex, year) stratum that
shifts the baseline log-odds so that the population-weighted average of :math:`p_{\text{prev}}`
across all 8 risk factor combinations matches :math:`\bar{p}_{\text{prev}}`. It plays the same role as an
intercept correction in a regression model.

The predicted prevalence for an individual agent is:

.. math::

  \text{logit}(p_{\text{prev}}) = \text{logit}(\bar{p}_{\text{prev}}) + \log(\omega_{\text{fhx}}) + \log(\omega_{\text{abx}}) - \alpha

.. list-table::
   :widths: 20 18 12 50
   :header-rows: 1

   * - Variable
     - Domain
     - Role
     - Description
   * - :math:`\bar{p}_{\text{prev}}`
     - probability :math:`\in [0, 1]`
     - Input
     - predicted prevalence from Model 1 for this (age, sex, year) stratum
   * - :math:`\log(\omega_{\text{fhx}})`
     - log-odds :math:`\in \mathbb{R}`
     - Input
     - log-OR for family history of asthma; from Patrick et al. :cite:`patrick2020`
   * - :math:`\log(\omega_{\text{abx}})`
     - log-odds :math:`\in \mathbb{R}`
     - Input
     - log-OR for antibiotic exposure in infancy; from Lee et al. :cite:`lee2024`
   * - :math:`\alpha`
     - log-odds :math:`\in \mathbb{R}`
     - Intermediate
     - per-stratum calibration term; looked up from ``asthma_occurrence_correction.csv`` at runtime
   * - :math:`p_{\text{prev}}`
     - probability :math:`\in [0, 1]`
     - Output
     - predicted asthma prevalence for an individual agent

:math:`\alpha` is solved offline using the ``Broyden-Fletcher-Goldfarb-Shanno (BFGS)``
algorithm to find the value that minimises the difference between the population-weighted
average and the Model 1 target:

.. math::

  \sum_{\lambda} p(\lambda) \cdot p_{\text{prev},\lambda} = \bar{p}_{\text{prev}}

where :math:`p(\lambda)` is the proportion of the population with risk factor combination
:math:`\lambda`.


Incidence
--------------------------------------------

The incidence formula has the same structure as prevalence, but applies only to the at-risk
population — agents who do not currently have asthma. The predicted incidence for an
individual agent is:

.. math::

  \text{logit}(p_{\text{inc}}) = \text{logit}(\bar{p}_{\text{inc}}) + \log(\omega_{\text{fhx}}) + \log(\omega_{\text{abx}}) - \alpha

.. list-table::
   :widths: 20 18 12 50
   :header-rows: 1

   * - Variable
     - Domain
     - Role
     - Description
   * - :math:`\bar{p}_{\text{inc}}`
     - probability :math:`\in [0, 1]`
     - Input
     - predicted incidence from Model 1 for this (age, sex, year) stratum
   * - :math:`\log(\omega_{\text{fhx}})`
     - log-odds :math:`\in \mathbb{R}`
     - Input
     - log-OR for family history of asthma; from Patrick et al. :cite:`patrick2020`
   * - :math:`\log(\omega_{\text{abx}})`
     - log-odds :math:`\in \mathbb{R}`
     - Input
     - log-OR for antibiotic exposure in infancy; from Lee et al. :cite:`lee2024`
   * - :math:`\alpha`
     - log-odds :math:`\in \mathbb{R}`
     - Intermediate
     - per-stratum calibration term; looked up from ``asthma_occurrence_correction.csv`` at runtime
   * - :math:`p_{\text{inc}}`
     - probability :math:`\in [0, 1]`
     - Output
     - predicted asthma incidence for an individual agent

:math:`\alpha` is solved offline using the ``Broyden-Fletcher-Goldfarb-Shanno (BFGS)``
algorithm to find the value that minimises the difference between the population-weighted
average and the Model 1 target:

.. math::

  \sum_{\lambda} p_{\text{no asthma},\lambda}(t-1) \cdot p_{\text{inc},\lambda} = \bar{p}_{\text{inc}}

where :math:`p_{\text{no asthma},\lambda}(t-1)` is the proportion of the population who
are asthma-free at :math:`t-1` and have risk factor combination :math:`\lambda`. Only
asthma-free agents are included because incidence counts new diagnoses only.

.. _optimizing-beta-parameters:

Calibrating Age-Dependent Odds Ratios for Prevalence and Incidence
------------------------------------------------------------------

Both :math:`\log(\omega_{\text{fhx}})` and :math:`\log(\omega_{\text{abx}})` are
age-dependent — the strength of association with asthma changes as a child ages:

* **Family history**: the OR is aplied from age 3, with the age-dependent slope applying
  up to age 5, after which it plateaus.
* **Antibiotic exposure**: the OR is only active for agents aged ≤ 7 who received
  antibiotics in infancy; it is zero otherwise.

These same age-dependent ORs feed into both the prevalence and incidence formulas. In the
prevalence formula, each OR is evaluated at the agent's age at entry. In the incidence
formula, the ORs are re-evaluated each year as the agent ages, changing until the agent
ages out of the relevant window.

The age-dependent slopes are not fixed by external studies and must be estimated from data.
They are calibrated in the incidence context — where agents pass through multiple ages
year by year, making the slope estimable from the change in OR across age groups — and the
resulting slopes are then used in both formulas. This is done by finding the slope values
that minimise the difference between the model-predicted odds ratios and the odds ratios
observed in the literature across age groups, using contingency tables.
See :doc:`model-contingency-tables` for an introduction to contingency tables and worked
examples.

In our model, we want to compute the contingency table for the risk factor combinations
:math:`\lambda` and the asthma diagnosis.


Past Contingency Table
^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

    <table class="table">
        <thead>
        <tr>
            <th></th>
            <th>asthma +</th>
            <th>asthma -</th>
            <th></th>
        </tr>
        </thead>
        <tbody>
        <tr>
            <td>risk factor &lambda; +</td>
            <td><code class="notranslate">a0</code></td>
            <td><code class="notranslate">b0</code></td>
            <td><code class="notranslate">n1</code></td>
        </tr>
        <tr>
            <td>risk factor &lambda; -</td>
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


Since the table is 2×2, we compare one risk factor combination at a time against the
no-risk-factor baseline (:math:`\lambda = 0`, i.e. no family history and no antibiotic
exposure). This produces 7 separate tables — one for each non-baseline combination. The
non-binary nature of antibiotic dose is handled implicitly through the :math:`\lambda`
indexing: dose levels 1, 2, and 3 each appear as distinct combinations and are each
compared independently against the baseline rather than against each other.

For each comparison, we have three quantities from the model: the population proportion
with risk factor combination :math:`\lambda`, the predicted prevalence for that combination
(:math:`p_{\text{prev},\lambda}`), and the predicted prevalence for the baseline
(:math:`p_{\text{prev},0}`). Together these determine the row total :math:`n_1` (people
with risk factor :math:`\lambda`) and the column total :math:`n_2` (people with asthma).

Given :math:`n_1`, :math:`n_2`, :math:`n`, and the odds ratio :math:`\omega_\lambda`, we
solve for the cell count :math:`a_0` (people with both risk factor :math:`\lambda` and
asthma) such that the implied odds ratio of the table matches :math:`\omega_\lambda`. This
is a non-trivial solve because all four cells are simultaneously constrained by the marginal
totals and the odds ratio — we use the method from Di Pietrantonj (2006)
:cite:`dipietrantonj2006`. The remaining cells follow directly from :math:`a_0` and the
marginal totals. See :doc:`conv_2x2 <../dev/api/data_generation/leap.data_generation.utils>`
for the Python implementation.

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
    a_{1, \text{ra}} &= a_0 \cdot \rho \\
    b_{1, \text{ra}} &= a_0 \cdot (1 - \rho) \\
    c_{1, \text{ra}} &= c_0 \cdot \rho \\
    d_{1, \text{ra}} &= c_0 \cdot (1 - \rho)

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

* :math:`a_{1, \text{ra}}` is the proportion of the population with risk factor combination :math:`\lambda`
  who had an asthma diagnosis at :math:`t=0` and still have it at :math:`t=1`
* :math:`b_{1, \text{ra}}` is the proportion of the population with risk factor combination :math:`\lambda`
  who had an asthma diagnosis at :math:`t=0` but no longer have it at :math:`t=1`
* :math:`c_{1, \text{ra}}` is the proportion of the population with no risk factors (:math:`\lambda = 0`)
  who had an asthma diagnosis at :math:`t=0` and still have it at :math:`t=1`
* :math:`d_{1, \text{ra}}` is the proportion of the population with no risk factors (:math:`\lambda = 0`)
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

We want to find the beta parameters that minimize the difference between the predicted odds
ratio :math:`\Omega` and the observed odds ratio :math:`\omega_{\lambda}`.

.. math::
    \sum_{i=1}^{N}\sum_{\lambda=1}^{n}
      \dfrac{\left| \log(\Omega^{(i)}) - \log(\omega_{\lambda}^{(i)}) \right|}{N}


Processed Data
--------------

The calibration terms produced by Model 2 are stored in ``asthma_occurrence_correction.csv``.
Each row gives the value of :math:`\alpha` for a specific age, sex, year, and outcome type
(prevalence or incidence). This file is used at runtime by the simulation to look up the
correction for each agent at each year of life.

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
        <td>Calendar year of the prediction</td>
      </tr>
      <tr>
        <td><code class="notranslate">sex</code></td>
        <td><code class="notranslate">str</code></td>
        <td>
          <code class="notranslate">"F"</code> = Female,
          <code class="notranslate">"M"</code> = Male
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">age</code></td>
        <td><code class="notranslate">int</code></td>
        <td>Age in years</td>
      </tr>
      <tr>
        <td><code class="notranslate">correction</code></td>
        <td><code class="notranslate">float</code></td>
        <td>
          The calibration term :math:`\alpha` for this stratum. Subtracted from the log-odds
          in the simulation to ensure the population-weighted average probability matches
          :math:`\bar{p}_{\text{prev}}` or :math:`\bar{p}_{\text{inc}}` from Model 1.
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">type</code></td>
        <td><code class="notranslate">str</code></td>
        <td>
          <code class="notranslate">"prevalence"</code> or
          <code class="notranslate">"incidence"</code> — separate correction terms are
          computed for each outcome type.
        </td>
      </tr>
    </tbody>
  </table>