.. _exacerbation-model:

===========================
Asthma Exacerbations Model
===========================

Data
====

Population Data
*****************

We use the Statistics Canada population data that was generated and saved as:
`processed_data/{time_delta_tag}/birth/initial_population.csv
<https://github.com/resplab/leap/blob/main/leap/processed_data/time_delta_365/birth/initial_population.csv>`_.
This file is produced by the same data-generation script as ``birth_estimate.csv`` — see
:ref:`birth-model` for details on how it is generated from Statistics Canada population data.


.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - ``timepoint``
     - :code:`datetime`
     - the starting date / time of the time interval that the data applies to
   * - ``age``
     - :code:`int`
     - the age of the person in years
   * - ``province``
     - :code:`str`
     - the province of the person
       (e.g., ``AB`` = Alberta, ``BC`` = British Columbia, etc.)
   * - ``n_age``
     - :code:`int`
     - the number of people in a given age group, time interval, province, and projection scenario
   * - ``n_birth``
     - :code:`int`
     - the number of births in that time interval, province, and projection scenario
   * - ``prop``
     - :code:`float`
     - the proportion of the population in that age group, time interval, province, and projection
       scenario relative to the number of births in that time interval, province, and projection
       scenario
   * - ``prop_male``
     - :code:`float`
     - the proportion of the population in a given age group, time interval, province, and
       projection scenario who are male
   * - ``projection_scenario``
     - :code:`str`
     - the projection scenario used to generate the data

This dataset has no ``sex`` column, so wherever a sex-specific population count
:math:`N(a, s, t)` is needed elsewhere in this document, it is derived from age :math:`a` and
timepoint :math:`t`'s ``n_age`` and ``prop_male`` values:

.. math::

    N(a, \text{M}, t) &= n_{\text{age}} \cdot p_{\text{male}} \\
    N(a, \text{F}, t) &= n_{\text{age}} \cdot (1 - p_{\text{male}})

where :math:`n_{\text{age}}` and :math:`p_{\text{male}}` are the ``n_age`` and ``prop_male``
columns respectively.


Occurrence Data
******************

We use the occurrence data that was generated and saved as:
`processed_data/{time_delta_tag}/asthma_occurrence_predictions.csv
<https://github.com/resplab/leap/blob/main/leap/processed_data/time_delta_365/asthma_occurrence_predictions.csv>`_

See :ref:`occurrence-model-1` for more details about this dataset.


.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - ``timepoint``
     - :code:`datetime`
     - The start of the time interval, e.g. 2024-01-01
   * - ``sex``
     - :code:`str`
     - ``F`` = female, ``M`` = male
   * - ``age``
     - :code:`int`
     - the age of the patient in years
   * - ``incidence``
     - :code:`float`
     - the predicted asthma incidence for the given time interval, age, and sex
   * - ``prevalence``
     - :code:`float`
     - the predicted asthma prevalence for the given time interval, age, and sex


Hospitalization Data
***********************

The data is from the ``Hospital Morbidity Database (HMDB)`` from the
`Canadian Institute for Health Information (CIHI)
<https://www.cihi.ca/en/hospital-morbidity-database-hmdb-metadata>`_.

The hospitalization data was collected from patients presenting to a hospital in Canada
due to an asthma exacerbation. We will use this data to calibrate the exacerbation model.

Per-province Structure
^^^^^^^^^^^^^^^^^^^^^^^

The raw data is saved under
`original_data/asthma_hosp/{province}/
<https://github.com/resplab/leap/tree/main/leap/original_data/asthma_hosp>`_,
with one subfolder per province/territory: ``AB``, ``BC``, ``MB``, ``NB``, ``NL``, ``NS``,
``ON``, ``PE``, ``QC``, ``SK``. In addition there are two combined regions:

* ``CA``: all of Canada combined.
* ``TR``: Nunavut, Northwest Territories, and Yukon combined into a single category, since
  each of these territories individually has too few cases to report reliably.

Each province subfolder contains 5 files that share the same columns and shape, differing
only in what value is reported:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - File
     - Description
   * - ``tab1_rate.csv``
     - the hospitalization rate per 100 000 people. This is the file used to calibrate the
       exacerbation model (see below).
   * - ``tab1_count.csv``
     - the number of people hospitalized with asthma.
   * - ``tab1_N.csv``
     - the total number of people in the category (the denominator used to compute the rate).
   * - ``tab1_lower.csv``
     - the lower error bar for the hospitalization rate.
   * - ``tab1_upper.csv``
     - the upper error bar for the hospitalization rate.

Each province subfolder also contains a ``los.csv`` file with length-of-stay statistics
(``avg``, ``med``, ``q1``, ``q3``) by ``fiscal_year``, ``age_group``, and ``sex``. This dataset
is not currently used by the exacerbation model.

.. _tab1-rate-columns:

``tab1_rate.csv`` Columns
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - ``fiscal_year``
     - :code:`int`
     - the year the data was collected
   * - ``N``
     - :code:`float`
     - the hospitalization rate for all ages and sexes in that year.
   * - ``M``
     - :code:`float`
     - the rate for all ages who are male in that year.
   * - ``F``
     - :code:`float`
     - the rate for all ages who are female in that year.
   * - ``0``
     - :code:`float`
     - the rate for all sexes aged 0 in that year.
   * - ``1``
     - :code:`float`
     - the rate for all sexes aged 1 in that year.
   * - ``...``
     - :code:`...`
     - ...
   * - ``90``
     - :code:`float`
     - the rate for all sexes aged 90 in that year.
   * - ``90+``
     - :code:`float`
     - the rate for all sexes aged over 90 in that year.
   * - ``F_0``
     - :code:`float`
     - the rate for all females aged 0 in that year.
   * - ``...``
     - :code:`...`
     - ...
   * - ``F_90+``
     - :code:`float`
     - the rate for all females aged over 90 in that year.
   * - ``M_0``
     - :code:`float`
     - the rate for all males aged 0 in that year.
   * - ``...``
     - :code:`...`
     - ...
   * - ``M_90+``
     - :code:`float`
     - the rate for all males aged over 90 in that year.

The hospitalization rate in this table is the hospitalization rate per 100 000 people.
For example, in the category ``F_90+``, the value would be the rate for people hospitalized who
are female and over 90 during the given year.

Therefore, the observed number of hospitalizations for a given age :math:`a`, sex :math:`s`, and
timepoint :math:`t` can be recovered from the rate:

.. math::

    N_{\text{hosp}}(a, s, t) = \dfrac{\text{hospitalization rate}(a, s, t)}{100\,000} \cdot N(a, s, t)

Here, :math:`N(a, s, t)` is the population count from the Population Data described above.

This is used to calibrate :math:`\alpha` below.

The calibration step only uses the per-sex, per-age columns (``F_0`` ... ``F_90+`` and
``M_0`` ... ``M_90+``); the sex- and age-aggregated columns (``N``, ``M``, ``F``, ``0`` ... ``90+``)
are not used.

.. _province-coverage:

Province Coverage
^^^^^^^^^^^^^^^^^^

Although hospitalization data is available for every province/territory listed above,
calibration is currently only implemented for ``BC`` and ``CA`` using the corresponding
``tab1_rate.csv`` file (see
`leap/data_generation/exacerbation_data.py
<https://github.com/resplab/leap/blob/main/leap/data_generation/exacerbation_data.py>`_).
The resulting ``exacerbation_calibration.csv`` used at runtime therefore only contains
calibration multipliers for these two regions.



Model
======

The number of exacerbations in a given time interval is modelled using a Poisson distribution.
The formula is:

.. math::

    N_{\text{exacerbations}} \sim \text{Poisson}(\lambda) = \dfrac{\lambda^k e^{-\lambda}}{k!}


Here :math:`\lambda` is the expected number of exacerbations per time interval, and :math:`k` is
the number of exacerbations (a non-negative integer) for which we are computing the
probability. To obtain :math:`\lambda`, we must perform a Poisson regression. The Poisson
regression assumes that the value we are interested in can be approximated using the following
formula:

.. math::

    \ln(\lambda) = \ln(\alpha) + \beta_0^{(i)} + \sum_{i=1}^3 \beta_i c_i


where:

.. list-table::
   :widths: 25 50
   :header-rows: 1

   * - Variable
     - Description
   * - :math:`\alpha`
     - the calibration multiplier that adjusts the model to match the hospitalization data
   * - :math:`\beta_0^{(i)}`
     - patient-specific random effect; :math:`\beta_0^{(i)} \sim \mathcal{N}(0, \sigma^2)`
   * - :math:`c_i`
     - relative time spent in control level :math:`i`, given by the probability of control level
       :math:`i` from the :ref:`Control Model <control-model>`
   * - :math:`\beta_i`
     - control level constant, derived from the EBA and GOAL studies (see Calibration below)

For each agent with asthma, :math:`\beta_0^{(i)}` is sampled once and held fixed for their
simulated lifetime, representing individual heterogeneity in exacerbation risk beyond what is
explained by age, sex, control level, and the population-level calibration :math:`\alpha`. The
mean and variance of this distribution are configuration-dependent; in the current default
configuration they are set to approximately :math:`\mathcal{N}(0, 0)`, effectively disabling this
source of individual variation.

This gives us the rate :math:`\lambda` of exacerbations of *any* severity. Once an individual's
total number of exacerbations for a time interval is drawn from this Poisson model, each
exacerbation is further assigned a severity level (mild, moderate, severe, or very severe) — see
:ref:`exacerbation_severity_model` below.

The diagram below summarises how these pieces fit together conceptually, for a given age
:math:`a` and sex :math:`s`, from calibration through to the final severity-level counts.

.. md-mermaid::

    flowchart TD
        ALPHA["<b>Calibration multiplier</b>&nbsp;<span style='font-size:1.3em'>$$\alpha$$</span><br/>predicted vs. observed<br/>hospitalizations in CIHI data"]
        BI["<b>Control-level rate constants</b>&nbsp;<span style='font-size:1.3em'>$$\beta_i$$</span><br/>derived from EBA + GOAL studies"]
        CI["<b>Control-level probabilities</b>&nbsp;<span style='font-size:1.3em'>$$c_i$$</span><br/>predicted by the Control Model"]

        RATE["<b>Rate of exacerbations</b>&nbsp;<span style='font-size:1.3em'>$$\lambda$$</span><br/>(any severity)"]

        ALPHA --> RATE
        BI --> RATE
        CI --> RATE

        COUNT["<b>Total exacerbations</b>&nbsp;<span style='font-size:1.3em'>$$N_{\text{exacerbations}}$$</span><br/>this time interval"]
        RATE --> COUNT

        SEV["<b>Severity probabilities</b>&nbsp;<span style='font-size:1.3em'>$$\mathbf{w}$$</span><br/>based on SYGMA II<br/>severity/hospitalization proportions"]
        HIST["<b>History of very severe</b><br/><b>exacerbations</b>&nbsp;<span style='font-size:1.3em'>$$\beta_{\text{prev hosp}}$$</span><br/>(hospitalization)"]
        HIST -->|"increases probability<br/>of very severe"| SEV

        OUTPUT["<b>Exacerbations by</b><br/><b>severity level</b>&nbsp;<span style='font-size:1.3em'>$$(n_{\text{mild}}, \ldots, n_{\text{very severe}})$$</span>"]
        COUNT --> OUTPUT
        SEV --> OUTPUT

        classDef input fill:#fff3e0,stroke:#e65100,color:#3a2400;
        classDef stage fill:#e3f2fd,stroke:#1565c0,color:#0d2b45;
        classDef output fill:#e8f5e9,stroke:#2e7d32,color:#10300f;
        class ALPHA,BI,CI,HIST input;
        class RATE,COUNT,SEV stage;
        class OUTPUT output;

        classDef input fill:#fff3e0,stroke:#e65100,color:#3a2400;
        classDef formula fill:#e3f2fd,stroke:#1565c0,color:#0d2b45;
        classDef sim fill:#e8f5e9,stroke:#2e7d32,color:#10300f;
        class ALPHA,B0,BI,CI input;
        class FORMULA,POISSON formula;
        class W,HIST,MULTI sim;

.. _exacerbation-calibration:

Calibration
******************

We are interested in calculating :math:`\alpha`. If we rewrite the equation, the meaning of
:math:`\alpha` becomes more apparent:

.. math::

    \lambda = \alpha \cdot e^{\beta_0^{(i)}} \prod_{i=1}^3 e^{\beta_i c_i}


How do we obtain :math:`\alpha`? We again assume that the mean value has the same form as in a
Poisson regression, with the following formula:

.. math::

    \ln(\lambda_{C}(a, s)) = \sum_{i=1}^3 \beta_i c_i


* :math:`\lambda_C(a, s)`: the predicted mean number of exacerbations per year for a given age
  :math:`a` and sex :math:`s` — note this has no timepoint argument, since the
  :ref:`Control Model <control-model>` assumes control level probabilities do not vary by
  calendar year
* :math:`c_i`: the age- and sex-specific probability of control level :math:`i`,
  :math:`P(y^{(i)} = k)`, from the :ref:`Control Model <control-model>`'s ordinal regression —
  obtained by differencing consecutive cumulative probabilities,
  :math:`P(y^{(i)} = k) = P(y^{(i)} \leq k) - P(y^{(i)} \leq k-1)`
* :math:`\beta_i`: control level constant, derived below from the EBA and GOAL studies

The :math:`\beta_i` values are derived by combining two literature sources, as described in
:cite:`leap2024`:

* the `Economic Burden of Asthma (EBA) study <https://bmjopen.bmj.com/content/3/9/e003360.long>`_
  (Chen et al. 2013) — a prospective, representative observational study of 618 participants aged
  1-85 years (74% aged 18 or older) with self-reported, physician-diagnosed asthma from BC, in
  which asthma control and the number of exacerbations were measured every 3 months over a year.
  EBA gives us the overall mean annual exacerbation rate for a person with asthma,
  :math:`r = 0.347`, and the overall proportion of time the EBA cohort as a whole spent in each
  control level over the study period: :math:`\text{prop}_{\text{wc}} = 0.340` (well-controlled),
  :math:`\text{prop}_{\text{pc}} = 0.474` (partially-controlled), and
  :math:`\text{prop}_{\text{uc}} = 0.186` (uncontrolled).
* the `GOAL Study <https://doi.org/10.1164/rccm.200401-033OC>`_
  (Bateman et al. 2004) — a one-year, randomized, double-blind clinical trial of 3,421
  participants aged 12-80 years with uncontrolled asthma at study entry, with asthma
  exacerbations as the primary outcome. An analysis of the GOAL data gives rounded annual
  exacerbation rates for each control level: well-controlled = 0.1, partially-controlled = 0.2,
  uncontrolled = 0.3 — i.e. the partially-controlled rate is twice the well-controlled rate, and
  the uncontrolled rate is three times the well-controlled rate.

Combining these two sources was necessary because the EBA cohort, while representative, did not
have enough exacerbation events on its own to robustly estimate a rate for each control level.
Instead, we take only the *relative* rates from GOAL, and solve for an absolute well-controlled
rate :math:`r_{\text{wc}}` such that the population-weighted average across the three control
levels — using the EBA time-in-control proportions together with the GOAL rate ratios — equals
the EBA overall rate :math:`r`:

.. math::

    r = \text{prop}_{\text{wc}} \cdot r_{\text{wc}} + \text{prop}_{\text{pc}} \cdot (2 r_{\text{wc}})
        + \text{prop}_{\text{uc}} \cdot (3 r_{\text{wc}})
    \quad \Longrightarrow \quad
    r_{\text{wc}} = \dfrac{r}{\text{prop}_{\text{wc}} + 2 \cdot \text{prop}_{\text{pc}}
        + 3 \cdot \text{prop}_{\text{uc}}}

The partially-controlled and uncontrolled rates follow directly from the GOAL ratios:
:math:`r_{\text{pc}} = 2 r_{\text{wc}}` and :math:`r_{\text{uc}} = 3 r_{\text{wc}}`. Taking the
natural log of each rate gives the :math:`\beta_i` values:

.. math::

    \beta_1 &:= \ln(r_{\text{wc}}) = \ln(0.1880058) \\
    \beta_2 &:= \ln(r_{\text{pc}}) = \ln(0.3760116) \\
    \beta_3 &:= \ln(r_{\text{uc}}) = \ln(0.5640174)


The number of exacerbations predicted by the model is then:

.. math::

    N_{\text{asthma}}(a, s, t) &= N(a, s, t) \cdot \eta_{\text{prev}}(a, s, t) \\
    N_{\text{exac}}^{\text{(pred)}}(a, s, t) &= \lambda_C(a, s) \cdot N_{\text{asthma}}(a, s, t) \\

* :math:`N_{\text{asthma}}(a, s, t)`: the number of people of age :math:`a`, sex :math:`s`, at
  timepoint :math:`t` with asthma
* :math:`N(a, s, t)`: the number of people of age :math:`a`, sex :math:`s`, at timepoint :math:`t`,
  from the Population Data described above
* :math:`\eta_{\text{prev}}(a, s, t)`: the prevalence of asthma for age :math:`a`, sex :math:`s`,
  at timepoint :math:`t`, from :ref:`occurrence-model-1`
* :math:`\lambda_C(a, s)`: as defined above

and number of hospitalizations is:

.. math::

    N_{\text{hosp}}^{\text{(pred)}}(a, s, t) = N_{\text{exac}}^{\text{(pred)}}(a, s, t) \cdot P(\text{hosp})


* :math:`N_{\text{exac}}^{\text{(pred)}}(a, s, t)`: the predicted number of exacerbations (of any
  severity) for age :math:`a`, sex :math:`s`, at timepoint :math:`t`
* :math:`P(\text{hosp})`: the probability of hospitalization due to asthma given the patient has an
  asthma exacerbation (a single constant, not stratified by :math:`a`, :math:`s`, or :math:`t`)

As described in :ref:`exacerbation_severity_model` below, exacerbations are classified into four
severity levels, where **very severe** is defined as requiring hospital admission. We treat
:math:`P(\text{hosp})` as the proportion of all exacerbations that are very severe, taken from the
`Symbicort Given as Needed in Mild Asthma II (SYGMA II) study
<https://www.nejm.org/doi/10.1056/NEJMoa1715275>`_ (Bateman et al. 2018), a double-blind,
multi-centre clinical trial of 4,176 individuals with mild asthma. SYGMA II reports the
distribution of exacerbation severity as 49.5% mild, 19.5% moderate, 28.3% severe, and 2.6% very
severe, giving :math:`P(\text{hosp}) = 0.026` :cite:`leap2024`.

Finally, :math:`\alpha` can be computed as the ratio of the **observed** to the **predicted**
number of hospitalizations, for a given age :math:`a`, sex :math:`s`, and timepoint :math:`t`:

.. math::

    \alpha(a, s, t) = \dfrac{N_{\text{hosp}}(a, s, t)}{N_{\text{hosp}}^{\text{(pred)}}(a, s, t)}

where :math:`N_{\text{hosp}}(a, s, t)` is the **observed** number of hospitalizations, determined
from the observed hospitalization rate in CIHI as described in :ref:`tab1-rate-columns` above.

.. info:: Math: Why α (from Hospitalizations) Applies to λ (All Severities)
  :collapsible:

  Although :math:`\alpha` is computed from hospitalizations alone, it is applied to
  :math:`\lambda`, the rate of exacerbations of *any* severity — this is valid because
  :math:`P(\text{hosp})` is treated as a fixed constant, independent of age, sex, province, and
  timepoint. Substituting
  :math:`N_{\text{hosp}}^{\text{(pred)}} = N_{\text{exac}}^{\text{(pred)}} \cdot P(\text{hosp})`,
  and writing the *observed* hospitalizations as the *true* total number of exacerbations times
  that same constant, :math:`N_{\text{hosp}} = N_{\text{exac}}^{\text{(true)}} \cdot P(\text{hosp})`,
  the :math:`P(\text{hosp})` terms cancel:

  .. math::

      \alpha = \dfrac{N_{\text{hosp}}}{N_{\text{hosp}}^{\text{(pred)}}}
          = \dfrac{N_{\text{exac}}^{\text{(true)}} \cdot P(\text{hosp})}
              {N_{\text{exac}}^{\text{(pred)}} \cdot P(\text{hosp})}
          = \dfrac{N_{\text{exac}}^{\text{(true)}}}{N_{\text{exac}}^{\text{(pred)}}}

  So :math:`\alpha` computed from the hospitalization ratio is algebraically identical to the
  ratio of true to predicted *total* exacerbations, provided :math:`P(\text{hosp})` is indeed
  constant by age, sex, province, and timepoint (otherwise variation in :math:`P(\text{hosp})`
  would be attributed to :math:`\alpha`). Hospitalizations are used to compute it — rather than
  total exacerbations directly — because they are captured completely in CIHI's national data,
  stratified by age/sex/province/year.

:math:`\alpha` is computed once per province, age, sex, and timepoint as part of data generation,
and saved as:
`processed_data/{time_delta_tag}/exacerbation_calibration.csv
<https://github.com/resplab/leap/blob/main/leap/processed_data/time_delta_365/exacerbation_calibration.csv>`_.
This file is looked up at simulation runtime — by province, timepoint, sex, and age — to
calibrate each agent's :math:`\lambda`.

Processed Data
^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - ``timepoint``
     - :code:`datetime`
     - The start of the time interval, e.g. 2024-01-01
   * - ``sex``
     - :code:`str`
     - ``F`` = female, ``M`` = male
   * - ``age``
     - :code:`int`
     - the age of the patient in years
   * - ``province``
     - :code:`str`
     - the 2-letter province abbreviation (currently only ``BC`` and ``CA``, see
       :ref:`province-coverage` above)
   * - ``calibrator_multiplier``
     - :code:`float`
     - the calibration multiplier :math:`\alpha(a, s, t)` for the given age, sex, timepoint, and
       province

.. _exacerbation_severity_model:

Severity
******************

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
:math:`\alpha` in the Calibration section above, and is what the Hospitalization Data described
earlier measures.

Dirichlet-Multinomial Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given the total number of exacerbations :math:`N_{\text{exacerbations}}` an individual experiences
in a time interval (drawn from the Poisson model above), the number at each severity level is
generated using a Dirichlet-Multinomial distribution. A preliminary probability vector across the
four levels is first drawn from a Dirichlet distribution:

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
:math:`P(\text{hosp})` in the :ref:`Calibration <exacerbation-calibration>` section above
:cite:`leap2024`, and :math:`\kappa = 100` is an assumed concentration
multiplier controlling how tightly an individual's probabilities cluster around the
population proportions :math:`\mathbf{p}`.

Adjustment for Previous Hospitalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
