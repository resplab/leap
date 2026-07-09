.. _mortality-model:

===========================
Mortality Model
===========================

.. toctree::
  :maxdepth: 2
  :hidden:

  model-death-kannisto
  model-death-life-expectancy

Data
====

.. _death-model-data-past:

Past Data: 1996 - 2021
*************************

For past years, we used
`Table 13-10-00837-01 from StatCan <https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1310083701>`_.

The ``*.csv`` file can be downloaded from here:
`13100837-eng.zip <https://www150.statcan.gc.ca/n1/tbl/csv/13100837-eng.zip>`_
and is saved as:
`LEAP/leap/original_data/13100837.csv
<https://github.com/resplab/leap/blob/main/leap/original_data/13100837.csv>`_.

The relevant columns are:

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - ``REF_DATE``
     - :code:`int`
     - the calendar year
   * - ``AGE_GROUP``
     - :code:`str`
     - the age of the person in years
   * - ``GEO``
     - :code:`str`
     - the province or territory full name
   * - ``SEX``
     - :code:`str`
     - one of "Both sexes", "Females", or "Males"
   * - ``ELEMENT``
     - :code:`str`
     - describes what the variable of interest is; we want ``"Death probability between age x and x+1 (qx)"``
   * - ``VALUE``
     - :code:`int`
     - the probability of death between age ``x`` and ``x+1`` in that year, province, sex, and age group


.. _death-model-data-calibration:

Projected Data: 2021 - 2068
****************************

``Statistics Canada`` doesn't provide annual projections for death probabilities, but does
provide a projection for specific years (which we call calibration years):


.. list-table::
   :widths: 25 20 30 25
   :header-rows: 1

   * - Region
     - Year
     - Projection Scenario
     - Mortality Scenario
   * - Canada
     - ``2028``
     - ``LG``
     - ``HM``
   * - Canada
     - ``2028``
     - ``M1``
     - ``MM``
   * - Canada
     - ``2028``
     - ``M2``
     - ``MM``
   * - Canada
     - ``2028``
     - ``M3``
     - ``MM``
   * - Canada
     - ``2028``
     - ``M4``
     - ``MM``
   * - Canada
     - ``2028``
     - ``M5``
     - ``MM``
   * - Canada
     - ``2028``
     - ``HG``
     - ``LM`` 
   * - Canada
     - ``2028``
     - ``SA``
     - ``HM``
   * - Canada
     - ``2028``
     - ``FA``
     - ``LM``
   * - Canada
     - ``2048``
     - ``LG``
     - ``HM``
   * - Canada
     - ``2048``
     - ``M1``
     - ``MM``
   * - Canada
     - ``2048``
     - ``M2``
     - ``MM``
   * - Canada
     - ``2028``
     - ``M3``
     - ``MM``
   * - Canada
     - ``2048``
     - ``M4``
     - ``MM``
   * - Canada
     - ``2048``
     - ``M5``
     - ``MM``
   * - Canada
     - ``2048``
     - ``HG``
     - ``LM`` 
   * - Canada
     - ``2048``
     - ``SA``
     - ``HM``
   * - Canada
     - ``2048``
     - ``FA``
     - ``LM``
   * - Canada
     - ``2073``
     - ``LG``
     - ``HM``
   * - Canada
     - ``2073``
     - ``M1``
     - ``MM``
   * - Canada
     - ``2073``
     - ``M2``
     - ``MM``
   * - Canada
     - ``2073``
     - ``M3``
     - ``MM``
   * - Canada
     - ``2073``
     - ``M4``
     - ``MM``
   * - Canada
     - ``2073``
     - ``M5``
     - ``MM``
   * - Canada
     - ``2073``
     - ``HG``
     - ``LM`` 
   * - Canada
     - ``2073``
     - ``SA``
     - ``HM``
   * - Canada
     - ``2073``
     - ``FA``
     - ``LM``
   * - Provinces / Territories
     - ``2028``
     - ``HG``, ``FA``
     - ``LM``
   * - Provinces / Territories
     - ``2028``
     - ``M1``, ``M2``, ``M3``, ``M4``, ``M5``, ``M6``
     - ``MM``
   * - Provinces / Territories
     - ``2028``
     - ``LG``, ``SA``
     - ``HM``
   * - Provinces / Territories
     - ``2033``
     - ``HG``, ``FA``
     - ``LM``
   * - Provinces / Territories
     - ``2033``
     - ``M1``, ``M2``, ``M3``, ``M4``, ``M5``, ``M6``
     - ``MM``
   * - Provinces / Territories
     - ``2033``
     - ``LG``, ``SA``
     - ``HM``
   * - Provinces / Territories
     - ``2038``
     - ``HG``, ``FA``
     - ``LM``
   * - Provinces / Territories
     - ``2038``
     - ``M1``, ``M2``, ``M3``, ``M4``, ``M5``, ``M6``
     - ``MM``
   * - Provinces / Territories
     - ``2038``
     - ``LG``, ``SA``
     - ``HM``
   * - Provinces / Territories
     - ``2043``
     - ``HG``, ``FA``
     - ``LM``
   * - Provinces / Territories
     - ``2043``
     - ``M1``, ``M2``, ``M3``, ``M4``, ``M5``, ``M6``
     - ``MM``
   * - Provinces / Territories
     - ``2043``
     - ``LG``, ``SA``
     - ``HM``
   * - Provinces / Territories
     - ``2048``
     - ``HG``, ``FA``
     - ``LM``
   * - Provinces / Territories
     - ``2048``
     - ``M1``, ``M2``, ``M3``, ``M4``, ``M5``, ``M6``
     - ``MM``
   * - Provinces / Territories
     - ``2048``
     - ``LG``, ``SA``
     - ``HM``
   * - Provinces / Territories
     - ``2053``
     - ``HG``, ``FA``
     - ``LM``
   * - Provinces / Territories
     - ``2053``
     - ``M1``, ``M2``, ``M3``, ``M4``, ``M5``, ``M6``
     - ``MM``
   * - Provinces / Territories
     - ``2053``
     - ``LG``, ``SA``
     - ``HM``
   * - Provinces / Territories
     - ``2058``
     - ``HG``, ``FA``
     - ``LM``
   * - Provinces / Territories
     - ``2058``
     - ``M1``, ``M2``, ``M3``, ``M4``, ``M5``, ``M6``
     - ``MM``
   * - Provinces / Territories
     - ``2058``
     - ``LG``, ``SA``
     - ``HM``
   * - Provinces / Territories
     - ``2063``
     - ``HG``, ``FA``
     - ``LM``
   * - Provinces / Territories
     - ``2063``
     - ``M1``, ``M2``, ``M3``, ``M4``, ``M5``, ``M6``
     - ``MM``
   * - Provinces / Territories
     - ``2063``
     - ``LG``, ``SA``
     - ``HM``
   * - Provinces / Territories
     - ``2068``
     - ``HG``, ``FA``
     - ``LM``
   * - Provinces / Territories
     - ``2068``
     - ``M1``, ``M2``, ``M3``, ``M4``, ``M5``, ``M6``
     - ``MM``
   * - Provinces / Territories
     - ``2068``
     - ``LG``, ``SA``
     - ``HM``
   * - Provinces / Territories
     - ``2073``
     - ``HG``, ``FA``
     - ``LM``
   * - Provinces / Territories
     - ``2073``
     - ``M1``, ``M2``, ``M3``, ``M4``, ``M5``, ``M6``
     - ``MM``
   * - Provinces / Territories
     - ``2073``
     - ``LG``, ``SA``
     - ``HM``



This data can be found in the ``Statistics Canada Population Projections Technical Report``:
`Table 3.1, Table 3.2, Table 5.2.1, Table 5.2.2, Table 5.2.3
<https://www150.statcan.gc.ca/n1/pub/91-620-x/91-620-x2025001-eng.htm>`_.
Statistics Canada assumes that the age distribution follows the Kannisto-Thatcher hazard model;
see :doc:`Kannisto-Thatcher Model <model-death-kannisto>` for details.


Model
========


.. raw:: html
  :file: ../../_static/img/lexis-diagram.svg

``Statistics Canada`` provides observed death probabilities for past years
(:ref:`1996-2021 <death-model-data-past>`) and life expectancy projections at a handful of future
calibration years (:ref:`2028, 2048, 2073 <death-model-data-calibration>`). To run the simulation,
we need death probabilities for every time interval across the full range — so we project forward
from the last observed year (2021).

We propose that 
the logit of the probability of death changes linearly over time at a sex-specific rate.
This gives the projection formula:

.. math::

    \text{logit}(q(x, \Delta x, t; \text{sex})) =
        \text{logit}(q(x, \Delta x, t_0; \text{sex})) -
        \beta_{\text{sex}} \cdot (t - t_0)

where:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Variable
     - Definition
   * - :math:`q(x, \Delta x, t; \text{sex})`
     - the probability that a person aged :math:`x` of sex ``sex`` at timepoint :math:`t` dies 
       between the ages ``[x, x + \Delta x)``
   * - :math:`t_0`
     - the last timepoint for which we have observed death probabilities (2021)
   * - :math:`t`
     - the timepoint for which we want to project death probabilities (up to 2068)
   * - :math:`\beta_{\text{sex}}`
     - the rate of mortality improvement — a negative value
       means mortality is declining over time. This parameter is calibrated separately for each
       sex, province, and projection scenario.
   * - :math:`\text{logit}(p)`
     - :math:`\ln\!\left(\tfrac{p}{1-p}\right)`

Calibrating the Beta Parameters
***************************************

The :math:`\beta_{\text{sex}}` parameter is not observed directly. We calibrate it separately
for each sex, province, and projection scenario by finding the value that makes our projected life
expectancy match Statistics Canada's published life expectancy targets at the calibration timepoints
(:ref:`2028, 2048, 2073 <death-model-data-calibration>`).

To evaluate a candidate :math:`\beta_{\text{sex}}`, we apply the projection formula above to
construct a full life table of death probabilities across all ages at a given calibration
timepoint, then compute life expectancy from that table (see :doc:`model-death-life-expectancy`).
The calibration minimises the discrepancy between this computed life expectancy and the
Statistics Canada target across all available calibration timepoints using
``scipy.optimize.leastsq``. Once :math:`\beta_{\text{sex}}` is fixed, the projection formula
is applied to fill in death probabilities for every time interval in the simulation.

Processed Data
=================

The past and projected death probabilities are combined by `leap/data_generation/death_data.py
<https://github.com/resplab/leap/blob/main/leap/data_generation/death_data.py>`_
into a set of processed files, with one file per time interval and province:

- `leap/processed_data/time_delta_365/death/life_table_{province}.csv
  <https://github.com/resplab/leap/blob/main/leap/processed_data/time_delta_365/death/life_table_CA.csv>`_ (annual)
- `leap/processed_data/time_delta_30/death/life_table_{province}.csv
  <https://github.com/resplab/leap/blob/main/leap/processed_data/time_delta_30/death/life_table_CA.csv>`_ (monthly)

Past data (from ``13100837.csv``) covers timepoints 1996 to 2021 using
death probabilities directly from Statistics Canada.

For projected timepoints (up to 2068), death probabilities for every time interval are filled in
by fitting a linear trend (in logit space) that connects the last historical timepoint to
Statistics Canada's life expectancy targets at the calibration timepoints. A separate :math:`beta`
is fitted for each sex, province, and projection scenario.

.. list-table::
   :widths: 25 25 50
   :header-rows: 1
   :class: long-table

   * - Column
     - Type
     - Description
   * - ``province``
     - :code:`str`
     - the 2-letter province or territory ID
       (e.g., ``BC`` = British Columbia, ``CA`` = Canada)
   * - ``projection_scenario``
     - :code:`str`
     - The two-letter projection scenario ID. One of:

       * ``past``: past data (1996-2021)
       * ``LG``: low-growth projection
       * ``HG``: high-growth projection
       * ``M1``: medium-growth 1 projection
       * ``M2``: medium-growth 2 projection
       * ``M3``: medium-growth 3 projection
       * ``M4``: medium-growth 4 projection
       * ``M5``: medium-growth 5 projection
       * ``M6``: medium-growth 6 projection
       * ``FA``: fast-aging projection
       * ``SA``: slow-aging projection

       See: `StatCan Projection Scenarios
       <https://www150.statcan.gc.ca/n1/pub/91-520-x/91-520-x2022001-eng.htm>`_.
   * - ``age``
     - :code:`int`
     - the age of the person in years
   * - ``sex``
     - :code:`str`
     - one of ``M`` = male, ``F`` = female
   * - ``timepoint``
     - :code:`datetime`
     - the starting date / time of the time interval that the data applies to, e.g. ``2021-04-01``.
   * - ``province``
     - :code:`str`
     - the 2-letter province or territory ID
       (e.g., ``BC`` = British Columbia, ``CA`` = Canada)
   * - ``projection_scenario``
     - :code:`str`
     - the population growth / mortality projection scenario, e.g. ``"FA"``, ``"M3"``;
       historical rows use ``"past"``
   * - ``prob_death``
     - :code:`float`
     - the probability of death between age ``[age, age + time_delta)`` for the given timepoint,
       province, projection scenario, and sex

.. toctree::
   :hidden:

   model-death-technical




