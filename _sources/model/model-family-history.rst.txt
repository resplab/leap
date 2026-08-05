.. _family-history-model:

========================
Family History Model
========================

Family history of asthma is a binary risk factor assigned to each agent at birth. It represents
whether at least one parent of the agent has asthma, and it affects the agent's probability of
developing asthma throughout their simulated life via the
:ref:`Asthma Occurrence Model <occurrence-model>`.

Data
====

The family history model is derived from an original analysis of the
`CHILD Cohort Study <https://childcohort.ca/>`_, a prospective cohort study of Canadian children. Two quantities are taken from this study:

1. **The prevalence of parental asthma** — the probability that a newborn agent has at least
   one parent with asthma. This is used to assign family history status at birth.
2. **Age-dependent odds ratios** — the relationship between family history and asthma risk
   at ages 3 and 5, used to parameterise both the prevalence and incidence risk factor
   corrections in the :ref:`Occurrence Model <occurrence-model-2>`.

The ``CHILD`` study cohort data is not open to the public. One can obtain access by following
the instructions on the `CHILD Cohort Study website <https://childstudy.ca/for-researchers/data-access/>`_.

Model: Bernoulli Distribution
==============================

At birth, each simulated agent is assigned a family history status by drawing from a
``Bernoulli distribution``:

.. math::

   f_i \sim \text{Bernoulli}(p)

where:

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Variable
     - Domain
     - Description
   * - :math:`f_i`
     - :math:`\{0, 1\}`
     - family history status of agent :math:`i`; ``1`` = at least one parent has asthma,
       ``0`` = neither parent has asthma
   * - :math:`p`
     - probability :math:`\in [0, 1]`
     - probability that a newborn agent has a family history of asthma

The parameter :math:`p` is estimated from the ``CHILD Cohort Study`` as the overall (unadjusted) proportion of
enrolled families in which at least one parent had an asthma diagnosis:

.. math::

   p = 0.2927

Family history status is fixed at birth and does not change over the agent's simulated
lifetime.

Role in the Occurrence Model
==============================

Once assigned, :math:`f_i` is used as a risk factor in the
:ref:`Asthma Occurrence Model <occurrence-model-2>`. The CHILD Study also provides the two
empirical odds ratios that anchor the age-dependent log-OR formula used there —
OR = 1.13 at age 3 and OR = 2.40 at age 5 :cite:`patrick2020`. See
:ref:`occurrence-model-2` for the full derivation of how these ORs are converted into
log-OR coefficients and combined with the antibiotic exposure log-OR and calibration
term :math:`\alpha` to produce each agent's individual asthma probability.
