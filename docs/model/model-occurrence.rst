.. _occurrence-model:

========================
Asthma Occurrence Model
========================

This document describes the asthma occurrence model, which is used to predict the incidence and
prevalence of asthma in British Columbia. The model is divided into two parts:

1. :ref:`occurrence-model-1`: A ``Generalized Linear Model (GLM)`` that predicts asthma incidence
   and prevalence based on age, sex, and timepoint.
2. :ref:`occurrence-model-2`: A model that incorporates risk factors such as family history and
   antibiotic use during infancy to predict asthma incidence and prevalence, along with the
   results from the first model.

.. _occurrence-model-1:

Occurrence Model 1: Crude Occurrence
=====================================

In the first model, we will use data collected from the ``BC Ministry of Health`` on the
incidence and prevalence of asthma in British Columbia. We will use this data to fit a 
``Generalized Linear Model (GLM)`` to predict the incidence and prevalence of asthma
based on the age, sex, and timepoint. However, asthma occurrence doesn't just depend on someone's age or
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
``2019``. Our model also makes predictions at customized age intervals, not 5-year age intervals.
To obtain these projections, we use a ``Generalized Linear Model (GLM)`` with a
``Poisson distribution`` and ``log link function``. Incidence and prevalence are counts of
people diagnosed with or living with asthma in a given time interval, making the Poisson distribution a
natural choice. See :doc:`model-statistical-background` for more information on ``GLMs``, including the Poisson
distribution and log link function.

Formula
-----------------

Now that we have our distribution and link function, we need to decide on a formula for the
linear predictor. The log link means we are modelling :math:`\log(\bar{p}_{\text{inc}})` for
incidence and :math:`\log(\bar{p}_{\text{prev}})` for prevalence. The scaling of the data
(rates per 100 people) is absorbed into the intercept :math:`\beta_0`. We are permitted to
use linear combinations of functions of the features in our dataset.

Let's start with ``incidence``. We want a formula using ``age``, ``sex``, and ``timepoint``.
Since asthma depends on factors such as pollution and antibiotic use, and these factors change
over time, it follows that asthma incidence should depend on the timepoint. Antibiotic use
also depends on age, so we should include age in our formula. Finally, there is a sex difference
in asthma incidence, so we should include sex in our formula.

.. TODO: Why was this formula chosen?


.. math::

    \log(\bar{p}_{\text{inc},i}) = \beta_0 + \beta_{\text{sex}} \cdot s_i + \beta_{\text{time}} \cdot t_i + \beta_{\text{time,sex}} \cdot t_i \cdot s_i
        + \sum_{k=1}^{5} \left( \beta_{\text{age},k} \cdot a_i^k + \beta_{\text{age,sex},k} \cdot a_i^k \cdot s_i \right)


where:

.. list-table::
   :widths: 50 20 30
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
   * - :math:`\beta_{\text{time}}`
     - :math:`t_i`
     - timepoint main effect
   * - :math:`\beta_{\text{time,sex}}`
     - :math:`t_i \cdot s_i`
     - timepoint × sex interaction
   * - :math:`\beta_{\text{age},k}` (:math:`k = 1, \ldots, 5`)
     - :math:`a_i^k`
     - age polynomial terms
   * - :math:`\beta_{\text{age,sex},k}` (:math:`k = 1, \ldots, 5`)
     - :math:`a_i^k \cdot s_i`
     - age × sex interaction terms

And :math:`a_i` is the age, :math:`t_i` is the timepoint, :math:`s_i` is the sex of individual :math:`i`.

There are :math:`4 + 5 + 5 = 14` coefficients in the incidence model.


Next we have the ``prevalence``. We again want a formula using ``age``, ``sex``, and ``timepoint``.
Since asthma prevalence depends on the number of people who have asthma, and this number changes
over time, we should include timepoint in our formula. Asthma prevalence also depends on age,
so we should include age in our formula. Finally, there is a sex difference
in asthma incidence and hence prevalence, so we should include sex in our formula.


.. math::

    \begin{align}
    \log(\bar{p}_{\text{prev},i}) &= \beta_0 + \beta_{\text{sex}} \cdot s_i \\
        &+ \sum_{k=1}^{5} \left( \beta_{\text{age},k} \cdot a_i^k + \beta_{\text{age,sex},k} \cdot a_i^k \cdot s_i \right) \\
        &+ \sum_{\ell=1}^{2} \left( \beta_{\text{time},\ell} \cdot t_i^\ell + \beta_{\text{time,sex},\ell} \cdot t_i^\ell \cdot s_i \right) \\
        &+ \sum_{\ell=1}^{2} \sum_{k=1}^{5} \left( \beta_{\text{age,time},k,\ell} \cdot a_i^k \cdot t_i^\ell
        + \beta_{\text{age,time,sex},k,\ell} \cdot a_i^k \cdot t_i^\ell \cdot s_i \right)
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
   * - :math:`\beta_{\text{sex}}`
     - :math:`s_i`
     - sex main effect
   * - :math:`\beta_{\text{age},k}` (:math:`k = 1, \ldots, 5`)
     - :math:`a_i^k`
     - age polynomial terms
   * - :math:`\beta_{\text{age,sex},k}` (:math:`k = 1, \ldots, 5`)
     - :math:`a_i^k \cdot s_i`
     - age × sex interactions
   * - :math:`\beta_{\text{time},\ell}` (:math:`\ell = 1, 2`)
     - :math:`(t_i)^\ell`
     - timepoint polynomial terms
   * - :math:`\beta_{\text{time,sex},\ell}` (:math:`\ell = 1, 2`)
     - :math:`(t_i)^\ell \cdot s_i`
     - timepoint × sex interactions
   * - :math:`\beta_{\text{age,time},k,\ell}` (:math:`k = 1, \ldots, 5`,  :math:`\ell = 1, 2`)
     - :math:`a_i^k \cdot (t_i)^\ell`
     - age × timepoint interactions
   * - :math:`\beta_{\text{age,time,sex},k,\ell}` (:math:`k = 1, \ldots, 5`, :math:`\ell = 1, 2`)
     - :math:`a_i^k \cdot (t_i)^\ell \cdot s_i`
     - age × timepoint × sex interactions

And :math:`a_i` is the age, :math:`t_i` is the timepoint, :math:`s_i` is the sex of individual :math:`i`.

There are :math:`(1 + 1 + 5 + 5) + (2 + 2 + 10 + 10) = 36` coefficients in the prevalence model.


Assumptions
-----------------

The following assumptions are applied when generating predictions from the fitted model:

* **Age**: The training data contains 5-year age bands up to a maximum midpoint of 63 years.
  For ages greater than 63, incidence and prevalence are assumed to remain constant at the
  rates predicted for age 63.

* **Timepoint**: Incidence and prevalence trends are predicted for all timepoints using the
  fitted GLM. At runtime, the timepoint is capped at ``2026-01-01`` — the maximum timepoint in
  ``asthma_occurrence_correction.csv`` — so that incidence and prevalence remain constant at
  the ``2026`` rates for all subsequent timepoints.


Processed Data
-----------------

The processed data produced by this model is stored in ``asthma_occurrence_predictions.csv``
(under the ``time_delta_<days>`` directory, where ``<days>`` is the number of days in the
simulation's time step — e.g. ``time_delta_365`` for yearly intervals). The data contains predicted asthma incidence and
prevalence at 1-year age intervals, for each timepoint and sex. The variables are:

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
        <td><code class="notranslate">datetime</code></td>
        <td>
          The start of the time interval, e.g. <code>2024-01-01</code>
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
          Predicted asthma incidence for the given time interval, age, and sex, per 100 people.
          Used as \(\bar{p}_{\text{inc}}\) in <a href="#occurrence-model-2">Model 2</a> (divided by 100 to convert to a probability).
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">prevalence</code></td>
        <td><code class="notranslate">float</code></td>
        <td>
          Predicted asthma prevalence for the given time interval, age, and sex, per 100 people.
          Used as \(\bar{p}_{\text{prev}}\) in <a href="#occurrence-model-2">Model 2</a> (divided by 100 to convert to a probability).
        </td>
      </tr>
    </tbody>
  </table>


.. _occurrence-model-2:

Occurrence Model 2: Risk Factors
=================================

:ref:`Model 1 <occurrence-model-1>` produces age-, sex-, and timepoint-specific rates for the general
population, but it treats everyone in a stratum identically. In reality, individuals differ in ways
that affect their asthma risk — most notably whether a parent has asthma, and whether they received
antibiotics in early life. Model 2 builds on :ref:`Model 1 <occurrence-model-1>` by incorporating
these risk factors, so that the simulation can assign each agent an individualised probability of
asthma incidence or prevalence rather than a population average.

This is done in two phases:

* **Data generation** (run once, before simulation): for every combination of age,
  sex, and timepoint, a calibration term :math:`\alpha` is computed that ensures the
  population-weighted average of the risk-factor-adjusted probabilities still matches the
  target rates :math:`\bar{p}_{\text{prev}}` and :math:`\bar{p}_{\text{inc}}` from
  :ref:`Model 1 <occurrence-model-1>`.
  The results are saved to
  ``asthma_occurrence_correction.csv``.

* **Simulation** (at runtime): each agent's individual risk factors are combined with
  :math:`\alpha` from the lookup table to produce a personalised asthma probability on every
  simulated timepoint of life.

Within the data generation phase, there is a fixed order of operations across three steps.
Computing :math:`\alpha` requires knowing :math:`\log(\omega_{\text{fhx}})` and
:math:`\log(\omega_{\text{abx}})` for each risk factor combination :math:`\lambda` at each
(age, sex, timepoint) stratum, since these determine the individual-level probabilities that must be
population-weighted to match the :ref:`Model 1 <occurrence-model-1>` targets :math:`\bar{p}_{\text{prev}}` and
:math:`\bar{p}_{\text{inc}}`. The log-ORs depend on age-dependent slope parameters
:math:`\beta_{\lambda,\text{age}}`, which are known from the literature for prevalence but
must be estimated for incidence — which creates the following sequence:

1. **Prevalence calibration.** The age-dependent OR slope coefficients
   (:math:`\beta_{\text{fhx,age}}`, :math:`\beta_{\text{abx,age}}`) for prevalence are
   derived directly from the literature, so :math:`\log(\omega_{\text{fhx}})` and
   :math:`\log(\omega_{\text{abx}})` are fully determined at each age. 
   ``Broyden-Fletcher-Goldfarb-Shanno (BFGS)`` then solves for
   :math:`\alpha_{\text{prev}}` per stratum as the value that makes the population-weighted
   average of the individual prevalence probabilities match the Model 1 target
   :math:`\bar{p}_{\text{prev}}`.

2. **Incidence** :math:`\beta_{\lambda,\text{age}}` **estimation.** Because no published
   studies provide age-dependent OR slopes for incidence, these are estimated by optimisation
   under two constraining assumptions: the intercept :math:`\beta_0` and antibiotic dose
   coefficient :math:`\beta_{\text{abx,dose}}` in the incidence OR equations are inherited
   directly from the prevalence values and held fixed — only the age slopes
   :math:`\beta_{\text{fhx,age}}` and :math:`\beta_{\text{abx,age}}` are free to vary; and
   the prevalence and incidence OR equations are equal at age 3, which is guaranteed by the
   shared intercepts. The optimiser is initialised from the prevalence slope values and finds
   the age slopes that simultaneously satisfy two conditions: (i) the average incidence
   across risk factor combinations :math:`\lambda`, weighted by their population proportions
   :math:`\text{prop}(\lambda)` within each (age, sex, timepoint) stratum, matches :math:`\bar{p}_{\text{inc}}`
   from Model 1; and (ii) the ORs implied by
   the :ref:`contingency tables <optimizing-beta-parameters>` simulated forward one timepoint
   from the calibrated prevalence distribution at age :math:`t-1` match the
   literature-derived prevalence ORs across age groups. The converged slopes are saved to
   ``occurrence_calibration_parameters.json``.

3. **Incidence calibration.** With the estimated :math:`\beta_{\lambda,\text{age}}` slopes
   from step 2, :math:`\log(\omega_{\text{fhx}})` and :math:`\log(\omega_{\text{abx}})` for
   incidence are fully determined. BFGS then solves for :math:`\alpha_{\text{inc}}` per
   stratum: the value that makes the population-weighted average of the at-risk incidence
   probabilities match the Model 1 target :math:`\bar{p}_{\text{inc}}`.

The diagram below summarises how the components relate, from the Model 1 targets through the
offline calibration steps to the personalised probabilities used at runtime.

.. md-mermaid::

    flowchart TD
        M1["<b>Model 1 population targets</b><br/>$$\overline{\vphantom{b}p}_{\text{prev}} \text{ and } \overline{\vphantom{b}p}_{\text{inc}}$$"]
        LIT["<b>Literature ORs (prevalence)</b>"]

        subgraph OFFLINE["Data generation (run once)"]
            direction TB
            S1["<b>Step 1 — Prevalence calibration</b><br/>
            $$\sum_\lambda \text{prop}(\lambda) \cdot p_{\text{prev},\lambda} = \overline{\vphantom{b}p}_{\text{prev}}$$"]
            S2["<b>Step 2 — Incidence β_age estimation</b><br/>optimise $$\quad \beta_{fhx_{age}}, \beta_{abx_{age}}$$"]
            S3["<b>Step 3 — Incidence calibration</b><br/>
            $$\sum_\lambda \text{prop}(\lambda) \cdot p_{\text{inc},\lambda} = \overline{\vphantom{b}p}_{\text{inc}}$$"]
            S1 -->|"calibrated prevalence at age $$~ t-1$$"| S2
            S2 -->|"$$\beta_{fhx_{age}}, \beta_{abx_{age}}$$"| S3
        end

        SIM["<b>Simulation (runtime)</b><br/>
        $$\text{logit}(p) = \text{logit}(\overline{\vphantom{b}p}) + \log(\omega_{\text{fhx}}) + \log(\omega_{\text{abx}}) - \alpha$$
        "]

        M1 -->|"$$\overline{\vphantom{b}p}_{\text{prev}}$$"| S1
        LIT --> S1
        M1 -->|"$$\overline{\vphantom{b}p}_{\text{inc}}$$ target + literature prevalence ORs"| S2
        M1 -->|"$$\overline{\vphantom{b}p}_{\text{inc}}$$"| S3
        S1 -->|"$$\alpha_{\text{prev}}$$"| SIM
        S3 -->|"$$\alpha_{\text{inc}}$$"| SIM
        M1 -->|"$$\overline{\vphantom{b}p}_{\text{prev}}, \overline{\vphantom{b}p}_{\text{inc}}$$"| SIM

        classDef target fill:#e3f2fd,stroke:#1565c0,color:#0d2b45;
        classDef lit fill:#fff3e0,stroke:#e65100,color:#3a2400;
        classDef sim fill:#e8f5e9,stroke:#2e7d32,color:#10300f;
        class M1 target;
        class LIT lit;
        class SIM sim;

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

.. _occurrence-model-2-prevalence:

Prevalence
--------------------------------------------

Applying individual risk factor ORs directly to the :ref:`Model 1 <occurrence-model-1>` log-odds would shift the
population-weighted average probability away from the target :math:`\bar{p}_{\text{prev}}`. The calibration
term :math:`\alpha` corrects for this: it is a single scalar per (age, sex, timepoint) stratum that
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
     - predicted prevalence from :ref:`Model 1 <occurrence-model-1>` for this (age, sex, timepoint) stratum
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

Solving for the Correction Term
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The correction term :math:`\alpha` is solved once, prior to runtime, using the
``Broyden-Fletcher-Goldfarb-Shanno (BFGS)`` algorithm to find the value that minimises the
difference between the population-weighted average asthma prevalence and the
:ref:`Model 1 <occurrence-model-1>` target asthma prevalence:

.. math::

  \bar{p}_{\text{prev}} = \sum_{\lambda} \text{prop}(\lambda) \cdot p_{\text{prev},\lambda}

where :math:`\text{prop}(\lambda)` is the proportion of the population with risk factor combination
:math:`\lambda`.



Age-dependent Odds Ratios for Prevalence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The log-ORs for each risk factor are age-dependent — their functional forms and fixed
coefficients are derived from external published studies.

Antibiotic Exposure
~~~~~~~~~~~~~~~~~~~~

The antibiotic log-OR was derived by Lee et al. :cite:`lee2024` using a random effects
meta-regression model. It applies only to children aged 7 or under who received antibiotics
in infancy, and is zero otherwise:

.. math::

  \log(\omega_{\text{abx}}) =
    \begin{cases}
      \beta_{\text{abx}_0} +
      \beta_{\text{abx}_\text{age}} \cdot \min(a_i, 7) +
      \beta_{\text{abx}_\text{dose}} \cdot \min(d_i, 3)
      & d_i > 0 \text{ and } a_i \leq 7 \\[6pt]
      0 & \text{otherwise}
    \end{cases}

where :math:`a_i` is the agent's age and :math:`d_i` is the number of courses of antibiotics
taken during the first year of life (capped at 3). All three coefficients are sourced directly
from the Lee et al. meta-regression, which estimated age as an explicit covariate:

* :math:`\beta_{\text{abx}_0} = 1.826`
* :math:`\beta_{\text{abx}_\text{age}} = 0.225`
* :math:`\beta_{\text{abx}_\text{dose}} = 0.053`

Family History
~~~~~~~~~~~~~~~~

The family history log-OR was derived from the ``CHILD Study`` by Patrick et al.
:cite:`patrick2020` using logistic regression. It applies to all ages but the age-dependent
component plateaus at age 5:

.. math::

  \log(\omega_{\text{fhx}}) =
    \beta_{\text{fhx}_0} \cdot f_i +
    \beta_{\text{fhx}_\text{age}} \cdot (\min(a_i, 5) - 3) \cdot f_i

where :math:`f_i = 1` if at least one parent has asthma, 0 otherwise. Both coefficients
are derived from the two empirical ORs reported by Patrick et al. at ages 3 and 5 —
OR = 1.13 at age 3 and OR = 2.40 at age 5. The age-dependent slope is the linear
interpolation between those two points on the log-OR scale:

* :math:`\beta_{\text{fhx}_0} = \log(1.13) = 0.122`
* :math:`\beta_{\text{fhx}_\text{age}} = \dfrac{\log(2.40) - \log(1.13)}{2} = 0.377`

For ages above 5, the OR is held constant at the age-5 value.


For prevalence, all OR coefficients — including the age-dependent slopes — are fully
determined by the literature before calibration runs. The only quantity calibrated for
prevalence is the scalar correction term :math:`\alpha`, which shifts the population-weighted
average onto the :ref:`Model 1 <occurrence-model-1>` target :math:`\bar{p}_{\text{prev}}`.


.. _occurrence-model-2-incidence:

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
     - predicted incidence from :ref:`Model 1 <occurrence-model-1>` for this (age, sex, timepoint) stratum
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


Solving for the Correction Term
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The correction term :math:`\alpha` is solved once, prior to runtime, using the
``Broyden-Fletcher-Goldfarb-Shanno (BFGS)`` algorithm to find the value that minimises the
difference between the population-weighted average asthma incidence and the
:ref:`Model 1 <occurrence-model-1>` target asthma incidence:

.. math::

  \sum_{\lambda} \text{prop}_{\text{no asthma},\lambda}(t-1) \cdot p_{\text{inc},\lambda} = \bar{p}_{\text{inc}}

where :math:`\text{prop}_{\text{no asthma},\lambda}(t-1)` is the proportion of the population who
are asthma-free at :math:`t-1` and have risk factor combination :math:`\lambda`. Only
asthma-free agents are included because incidence counts new diagnoses only.



.. _optimizing-beta-parameters:

Calibrating Age-Dependent Odds Ratios
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unlike prevalence, there are no published studies that directly estimate the age-dependent
slopes of :math:`\log(\omega_{\text{fhx}})` and :math:`\log(\omega_{\text{abx}})` for
incidence. Instead, the slopes are estimated by optimisation under two constraining
assumptions:

* The intercept :math:`\beta_0` and antibiotic dose coefficient :math:`\beta_{\text{abx,dose}}`
  in the incidence OR equations are **inherited from prevalence** and held fixed. Only the age
  slopes :math:`\beta_{\text{fhx,age}}` and :math:`\beta_{\text{abx,age}}` are free to vary.
* The prevalence and incidence OR equations are **equal at age 3**, which is guaranteed by
  sharing the same intercept.

The optimiser is initialised from the corresponding prevalence slope values and must
simultaneously satisfy two conditions:

1. The average incidence across risk factor combinations :math:`\lambda`, weighted by their
   population proportions :math:`\text{prop}(\lambda)` within each (age, sex, timepoint) stratum, matches
   the :ref:`Model 1 <occurrence-model-1>` target :math:`\bar{p}_{\text{inc}}`.
2. The ORs implied by the contingency tables — simulated forward one timepoint from the calibrated
   prevalence distribution at age :math:`t-1` — match the literature-derived prevalence ORs
   across age groups.

Condition 2 ensures that introducing individual-level incidence risk does not distort the
aggregate OR structure established by the prevalence literature.
See :ref:`contingency-tables` for an introduction to contingency tables and worked examples.

To evaluate condition 2, we track a cohort across one timepoint using contingency tables of
risk factor combination :math:`\lambda` against asthma diagnosis. The labels :math:`t=0` and
:math:`t=1` denote two consecutive timepoints. The procedure is:

1. **Build the baseline table at** :math:`t=0` so that its odds ratio equals the (literature-anchored)
   prevalence OR at the previous timepoint.
2. **Step forward one timepoint** by applying reassessment to those who already had asthma and
   **new incidence — computed with the candidate age-slopes** :math:`\beta_{\lambda,\text{age}}` —
   to those who did not, producing the combined table at :math:`t=1`.
3. **Compare** the odds ratio of that combined table (the prevalence OR the model actually
   produces at age :math:`a`) against the target prevalence OR, and adjust
   :math:`\beta_{\lambda,\text{age}}` until they match.

The four tables below build this up step by step, and the diagram summarises the flow.

.. md-mermaid::

    flowchart LR
        BASE["<b>Baseline table (t=0)</b><br/>built from literature-based<br/>prevalence OR"]
        NEW["<b>New diagnoses (t=1)</b><br/>incidence with candidate $$~\beta_{age}$$"]
        EX["<b>Existing diagnoses (t=1)</b><br/>reassessment with prob $$~\rho$$"]
        COMB["<b>Combined table (t=1)</b><br/>prevalence at age a:<br/> evaluate prevalence OR vs target prevalence OR"]

        BASE -->|"no asthma at t=0"| NEW
        BASE -->|"had asthma at t=0"| EX
        NEW --> COMB
        EX --> COMB
        COMB -.->|"adjust $$~\beta_{age}$$"| NEW

        classDef base fill:#e3f2fd,stroke:#1565c0,color:#0d2b45;
        classDef mid fill:#fff3e0,stroke:#e65100,color:#3a2400;
        classDef comb fill:#e8f5e9,stroke:#2e7d32,color:#10300f;
        class BASE base;
        class NEW,EX mid;
        class COMB comb;


Baseline Contingency Table (t=0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
            <td><code class="notranslate">n0</code></td>
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

For each comparison, let :math:`N` be the total population across all risk factor
combinations (a hypothetical size, e.g. 100,000 — the odds ratio is scale-invariant).
The row and column totals are then:

.. math::

    n_1 &= \text{prop}(\lambda) \cdot N \\
    n_0 &= \text{prop}(0) \cdot N \\
    n   &= n_1 + n_0 \\
    n_2 &= p_{\text{prev},\lambda} \cdot n_1 + p_{\text{prev},0} \cdot n_0

where :math:`n_1` is the number of people with risk factor :math:`\lambda`, :math:`n_0` is
the number with no risk factors, :math:`n` is the grand total of the 2×2 table, and
:math:`n_2` is the total number with asthma across both groups.

Given :math:`n_1`, :math:`n_0`, :math:`n_2`, :math:`n`, and the odds ratio :math:`\omega_\lambda`, we
solve for the cell count :math:`a_0` (people with both risk factor :math:`\lambda` and
asthma) such that the implied odds ratio of the table matches :math:`\omega_\lambda`. This
is a non-trivial solve because all four cells are simultaneously constrained by the marginal
totals and the odds ratio — we use the method from Di Pietrantonj (2006)
:cite:`dipietrantonj2006`. The remaining cells follow directly from :math:`a_0` and the
marginal totals.

Existing Diagnoses: Reassessment at t=1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In our model, an asthma diagnosis is not permanent — a person may be reassessed and lose
their diagnosis from one timepoint to the next. This table tracks what happens at :math:`t=1` to
the :math:`a_0 + c_0` people who already had asthma at :math:`t=0`. Let :math:`\rho` be
the probability of retaining a diagnosis. Then each cell is simply the corresponding
baseline cell scaled by :math:`\rho` (retained) or :math:`1 - \rho` (lost):

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
            <td><code class="notranslate">a1_existing</code></td>
            <td><code class="notranslate">b1_existing</code></td>
            <td><code class="notranslate">a0</code></td>
        </tr>
        <tr>
            <td>risk factor &lambda; -</td>
            <td><code class="notranslate">c1_existing</code></td>
            <td><code class="notranslate">d1_existing</code></td>
            <td><code class="notranslate">c0</code></td>
        </tr>
        </tbody>
    </table>

.. math::
    a_{1, \text{existing}} &= a_0 \cdot \rho \\
    b_{1, \text{existing}} &= a_0 \cdot (1 - \rho) \\
    c_{1, \text{existing}} &= c_0 \cdot \rho \\
    d_{1, \text{existing}} &= c_0 \cdot (1 - \rho)

where :math:`\rho` is the probability of retaining an asthma diagnosis from one timepoint to the
next, applied equally regardless of risk factor status.


New Diagnoses at t=1
~~~~~~~~~~~~~~~~~~~~~~~~

This table tracks what happens at :math:`t=1` to the :math:`b_0 + d_0` people who were
asthma-free at :math:`t=0`. Each person may receive a new diagnosis based on their
incidence probability: :math:`p_{\text{inc},\lambda}` for those with risk factor
:math:`\lambda`, and :math:`p_{\text{inc},0}` for those with no risk factors.

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
            <td><code class="notranslate">a1_new</code></td>
            <td><code class="notranslate">b1_new</code></td>
            <td><code class="notranslate">b0</code></td>
        </tr>
        <tr>
            <td>risk factor &lambda; -</td>
            <td><code class="notranslate">c1_new</code></td>
            <td><code class="notranslate">d1_new</code></td>
            <td><code class="notranslate">d0</code></td>
        </tr>
        </tbody>
    </table>

.. math::
    a_{1, \text{new}} &= b_0 \cdot p_{\text{inc},\lambda} \\
    b_{1, \text{new}} &= b_0 \cdot (1 - p_{\text{inc},\lambda}) \\
    c_{1, \text{new}} &= d_0 \cdot p_{\text{inc},0} \\
    d_{1, \text{new}} &= d_0 \cdot (1 - p_{\text{inc},0})


Combined Contingency Table: Existing and New Diagnoses (t=1)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The combined table at :math:`t=1` sums the reassessment and new diagnosis components for
each cell, giving the full joint distribution of risk factor status and asthma diagnosis
at the end of the time interval:

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
            <td><code class="notranslate">a1</code></td>
            <td><code class="notranslate">b1</code></td>
            <td><code class="notranslate">n1</code></td>
        </tr>
        <tr>
            <td>risk factor &lambda; -</td>
            <td><code class="notranslate">c1</code></td>
            <td><code class="notranslate">d1</code></td>
            <td><code class="notranslate">n0</code></td>
        </tr>
        </tbody>
    </table>


where:

.. math::
    a_1 &= a_{1, \text{existing}} + a_{1, \text{new}} \\
    b_1 &= b_{1, \text{existing}} + b_{1, \text{new}} \\
    c_1 &= c_{1, \text{existing}} + c_{1, \text{new}} \\
    d_1 &= d_{1, \text{existing}} + d_{1, \text{new}}

From these values, we can compute the odds ratio:

.. math::
    \hat{\omega} = \dfrac{a_1 \cdot d_1}{b_1 \cdot c_1}


Optimization
~~~~~~~~~~~~~

We want to find the age-dependent slope values (:math:`\beta_{\lambda, \text{age}}`) 
that minimize the mean absolute difference between :math:`\log(\hat{\omega})`
and :math:`\log(\omega_{\lambda})` — the fixed, age-dependent log-ORs for prevalence sourced from
external studies — averaged across all age groups and all 7 non-baseline risk factor combinations.

Once optimised, these slopes (:math:`\beta_{\lambda, \text{age}}`) are used
to compute :math:`\log(\omega_{\text{fhx}})` and :math:`\log(\omega_{\text{abx}})` for
each (age, sex, timepoint) stratum. BFGS then uses those stratum-specific log-ORs to solve for
the calibration term :math:`\alpha` per stratum, which is stored in
`asthma_occurrence_correction.csv <https://github.com/resplab/leap/blob/main/leap/processed_data/time_delta_365/asthma_occurrence_correction.csv>`_
as described in the :ref:`occurrence-model-2-processed-data` section below.


.. _occurrence-model-2-processed-data:

Processed Data
--------------

The calibration terms produced by :ref:`Model 2 <occurrence-model-2>` are stored in
`asthma_occurrence_correction.csv <https://github.com/resplab/leap/blob/main/leap/processed_data/time_delta_365/asthma_occurrence_correction.csv>`_
(under the ``time_delta_<days>`` directory matching the simulation's time step). Each row gives
the value of :math:`\alpha` for a specific age, sex, timepoint, and outcome type
(prevalence or incidence). This file is used at runtime by the simulation to look up the
correction for each agent at each timepoint of life.

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
        <td><code class="notranslate">datetime</code></td>
        <td>The timepoint of the prediction, e.g. <code>2024-01-01</code></td>
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
          The calibration term \(\alpha\) for this stratum. Subtracted from the log-odds
          in the simulation to ensure the population-weighted average probability matches
          \(\bar{p}_{\text{prev}}\) or \(\bar{p}_{\text{inc}}\) from <a href=#occurrence-model-1>Model 1</a>.
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

Technical Appendix
------------------

For the full mathematical derivations behind the calibration model — including the risk factor
formula, correction term derivations, and the beta parameter optimisation — see the
:ref:`occurrence-model-appendix`.

.. toctree::
   :maxdepth: 1
   :hidden:

   model-occurrence-appendix