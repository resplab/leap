.. _exacerbation_severity_model:

=====================================
Asthma Exacerbation Severity Model
=====================================

Each asthma exacerbation is assigned one of four severity levels, classified retrospectively by
the level of healthcare utilization required to treat it, following the framework described by
the Global Initiative for Asthma (GINA) :cite:`gina2023`:

.. list-table::
   :widths: 10 20 70
   :header-rows: 1

   * - Level
     - Name
     - Healthcare utilization
   * - 1
     - Mild
     - managed with reliever medication alone
   * - 2
     - Moderate
     - requires a physician visit and a prescription of oral corticosteroids (OCS)
   * - 3
     - Severe
     - requires an emergency department (ED) visit
   * - 4
     - Very severe
     - requires hospital admission

The ``very severe`` level is the one used to compute :math:`P(\text{hosp})` and calibrate
:math:`\alpha` in the :ref:`Calibration <exacerbation-calibration>` section of the
:ref:`exacerbation-model`, and is what the Hospitalization Data described in that section measures.

Dirichlet-Multinomial Model
=============================

Given the total number of exacerbations :math:`N_{\text{exacerbations}}` an individual experiences
in a time interval (drawn from the Poisson model described in the :ref:`exacerbation-model`), the
number at each severity level is generated using a Dirichlet-Multinomial distribution. A preliminary
probability vector across the four levels is first drawn from a Dirichlet distribution:

.. math::

    \mathbf{w}^{\text{pre}} \sim \text{Dirichlet}(\boldsymbol{\delta})

:math:`\mathbf{w}^{\text{pre}} = (w^{\text{pre}}_{\text{mild}}, w^{\text{pre}}_{\text{moderate}},
w^{\text{pre}}_{\text{severe}}, w^{\text{pre}}_{\text{very severe}})` is a length-4 vector of
*probabilities* (summing to 1) giving this individual's personal probability of each severity
level. For each agent with asthma, :math:`\mathbf{w}^{\text{pre}}` is sampled once, independently
per agent, and held fixed for their simulated lifetime — representing individual heterogeneity in
exacerbation severity, distinct from (and prior to) the adjustment for previous hospitalization
described below. The actual exacerbation counts are determined later using
:math:`N_{\text{exacerbations}}` (the total count, from the Poisson model above) together with
this probability vector.

:math:`\boldsymbol{\delta} = \kappa \cdot \mathbf{p}` is the Dirichlet concentration vector,
:math:`\mathbf{p} = (p_{\text{mild}}, p_{\text{moderate}}, p_{\text{severe}}, p_{\text{very severe}})
= (0.495, 0.195, 0.283, 0.026)` are the same SYGMA II severity proportions used for
:math:`P(\text{hosp})` in the :ref:`Calibration <exacerbation-calibration>` section of the
:ref:`exacerbation-model` :cite:`leap2024`, and :math:`\kappa = 100` is an assumed concentration
multiplier controlling how tightly an individual's probabilities cluster around the
population proportions :math:`\mathbf{p}`.

Adjustment for Previous Hospitalization
=========================================

If the individual has previously been hospitalized for an asthma exacerbation, their probability
of a very severe exacerbation is increased, and the remaining probability mass is redistributed
proportionally across the other three levels:

.. math::

    w_{\text{very severe}} &= w^{\text{pre}}_{\text{very severe}} \cdot \beta_{\text{prev hosp}} \\
    w_j &= \dfrac{w^{\text{pre}}_j}{\sum_{l \,\in\, \{\text{mild, moderate, severe}\}} w^{\text{pre}}_l}
        \cdot (1 - w_{\text{very severe}}), \quad j \in \{\text{mild, moderate, severe}\}

where:

* :math:`j \in \{\text{mild, moderate, severe}\}`: this equation applies separately to each of
  these three levels
* :math:`l \in \{\text{mild, moderate, severe}\}`: a dummy index used only for the summation in
  the denominator
* :math:`\beta_{\text{prev hosp}}`: :math:`\beta_{\text{prev hosp,pediatric}} = 1.79` for
  individuals under 14 years of age, or :math:`\beta_{\text{prev hosp,adult}} = 2.88` for
  individuals 14 years of age or older.

These rate multipliers are taken from a Canadian cohort study of the long-term natural history of
severe asthma exacerbations :cite:`lee2022natural`, which found that a first follow-up severe
exacerbation was associated with a 79% increase (rate multiplier 1.79, 95% CI 1.11–2.89) in the
rate of subsequent exacerbations for pediatric patients, and a 188% increase (rate multiplier
2.88, 95% CI 1.35–5.15) for adult patients.

If the individual has no prior hospitalization, :math:`\mathbf{w} = \mathbf{w}^{\text{pre}}`.

Since :math:`\mathbf{w}^{\text{pre}}` already sums to 1, inflating :math:`w_{\text{very severe}}`
by :math:`\beta_{\text{prev hosp}}` alone would push the total above 1. The :math:`w_j` formula
corrects this: it proportionally shrinks the mild/moderate/severe probabilities so the full
vector :math:`\mathbf{w}` sums back to 1, while keeping their relative proportions to each other
unchanged from :math:`\mathbf{w}^{\text{pre}}`.


This raises the question of how "previously hospitalized" is determined for an agent whose
asthma history was not directly simulated cycle-by-cycle — for example, an agent assigned an
asthma label and a diagnosis age all at once when they enter the simulation. See
:ref:`step-4-check-hospitalizations` for how this is initialized in that case.

Finally, the number of exacerbations at each severity level is drawn from a Multinomial
distribution, using :math:`N_{\text{exacerbations}}` as the number of trials:

.. math::

    (n_{\text{mild}}, n_{\text{moderate}}, n_{\text{severe}}, n_{\text{very severe}})
        \sim \text{Multinomial}(N_{\text{exacerbations}}, \mathbf{w})
