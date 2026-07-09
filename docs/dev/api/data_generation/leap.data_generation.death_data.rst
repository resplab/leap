Death Data
==========

To run the data generation for the mortality data:

.. code-block:: bash

   cd LEAP
   python3 leap/data_generation/death_data.py --time-delta P1Y


This will update the following data file: 

1. ``leap/processed_data/{time_delta_tag}life_table.csv``


The ``--time-delta`` argument must be in **ISO 8601** format:

.. list-table::
   :widths: 25 50
   :header-rows: 1

   * - ISO 8601
     - Meaning
   * - P1Y1M1DT1H1M1.1S
     - 1 year, 1 month, 1 day, 1 hour, 1 minute, 1 second, and 100 milliseconds
   * - P40D	
     - 40 days
   * - P1Y1D
     - 1 year and 1 day
   * - P3DT4H59M
     - 3 days, 4 hours, and 59 minutes
   * - PT2H30M
     - 2 hours and 30 minutes
   * - P1M
     - 1 month
   * - PT1M
     - 1 minute

Datasets
**********

To obtain the mortality data for each year, we used one table from ``StatCan``:

1. **1996 - 2021:**

   For past years, we used
   `Table 13-10-00837-01 from StatCan <https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1310083701>`_.

   The ``*.csv`` file can be downloaded from here:
   `13100837-eng.zip <https://www150.statcan.gc.ca/n1/tbl/csv/13100837-eng.zip>`_.

   and is saved as: ``LEAP/leap/original_data/13100837.csv``

2. **2021 - 2068:**

   ``StatCan`` doesn't provide annual projections for death probabilities, but does provide a
   projection for specific years (which we call calibration years) for the ``M3`` projection
   scenario only.
   For Canada, this is 2068, and for BC, 2043.


Data Processing
****************

1. Load Past Life Table
------------------------

First, we load the past life table from the ``StatCan`` data file, ``13100837.csv``. This file
contains the probability of death for each age, with a time delta of 1 year, for the years ``1996``
to ``2021``. The output is a dataframe with the following columns:


.. list-table::
   :widths: 15 10 75
   :header-rows: 1

   * - Name
     - Type
     - Description
   * - ``province``
     - ``str``
     - The two-letter abbreviation of the province, e.g. ``BC`` for British Columbia, or ``CA`` for
       Canada.
   * - ``age``
     - ``int``
     - The age of the person, in integer years.
   * - ``sex``
     - ``str``
     - One of ``"M"`` = male, ``"F"`` = female.
   * - ``timepoint``
     - ``datetime``
     - The starting timepoint of the interval during which the data was collected.
   * - ``prob_death``
     - ``float``
     - The probability that a person of the given age and sex, living in the given province, will
       die in the time interval ``[timepoint, timepoint + time_delta)``.

.. info:: Example: Past Life Table
  :collapsible:

  .. list-table::
    :widths: 10 10 10 30 30
    :header-rows: 1

    * - ``province``
      - ``age``
      - ``sex``
      - ``timepoint``
      - ``prob_death``
    * - BC
      - 0
      - M      
      - 1996-01-01
      - 0.005
    * - BC
      - ...
      - M      
      - 1996-01-01
      - ...
    * - BC
      - 110
      - M      
      - 1996-01-01
      - 0.0023
    * - BC
      - 0
      - M      
      - 1997-01-01
      - 0.0043
    * - BC
      - ...
      - M      
      - ...
      - ...
    * - BC
      - 110
      - M      
      - 2022-01-01
      - 0.0044
    * - BC
      - 0
      - F
      - 1996-01-01
      - 0.0054


2. Calibration Data
---------------------

Next, we load the calibration data from the ``StatCan`` files:

1. ``mortality_projections_table_3-2.csv``
2. ``mortality_projections_table_5-2.csv``

These files contain the projected life expectancies for a handful of specific years, which we will
use to calibrate our model. The output is a dataframe with the following columns:


.. list-table::
   :widths: 15 10 75
   :header-rows: 1

   * - Name
     - Type
     - Description
   * - ``province``
     - ``str``
     - The two-letter abbreviation of the province, e.g. ``BC`` for British Columbia, or ``CA`` for
       Canada.
   * - ``sex``
     - ``str``
     - One of ``"M"`` = male, ``"F"`` = female.
   * - ``timepoint``
     - ``datetime``
     - The starting timepoint of the interval during which the data was collected.
   * - ``projection_scenario``
     - ``str``
     - The two-letter projection scenario, e.g. ``M3``.
   * - ``mortality_scenario``
     - ``str``
     - The two-letter mortality scenario, e.g. ``HM``.
   * - ``life_expectancy``
     - ``float``
     - The projected life expectancy for a person of the given sex, living in the given
       province, under the specified projection and moortality scenarios.



.. info:: Example: Calibration Data
  :collapsible:

  .. list-table::
    :widths: 8 8 30 10 15 15
    :header-rows: 1

    * - ``province``
      - ``sex``
      - ``timepoint``
      - ``projection_scenario``
      - ``mortality_scenario``
      - ``life_expectancy``
    * - CA
      - M
      - 2028-01-01
      - LG
      - HM
      - 80.5
    * - CA
      - F
      - 2028-01-01
      - LG
      - HM
      - 84.2
    * - CA
      - M
      - 2048-01-01
      - LG
      - HM
      - 84.5
    * - CA
      - F
      - 2048-01-01
      - LG
      - HM
      - 86.7


3. Determine the Beta Parameters
---------------------------------

The following equation can be used to obtain the probability of death in future years:

.. math::

  \sigma^{-1}(q(x, \Delta x, t; \theta))
    = \sigma^{-1}(q(x, \Delta x, t_0; \theta)) - e^{\beta(\theta)(t - t_0)}


where:

* :math:`\sigma^{-1}` is the inverse sigmoid function, also known as the logit function:

.. math::

  \sigma^{-1}(p) = \ln\left(\dfrac{p}{1-p}\right)

and:

* :math:`x` is the age in years
* :math:`\Delta x` is the time between data points of the original data (in our case, 1 year)
* :math:`s` is the sex
* :math:`t_0` is the year the collected data ends (in our case, 2022)
* :math:`t` is the future year
* :math:`\theta = \{\text{sex}, \text{province}, \text{projection_scenario}\}` 
* :math:`q(x, \Delta x, t_0; \theta)` is the probability that a person of age :math:`x` and sex :math:`s`
  in the year :math:`t_0` will die between the ages :math:`x` and :math:`x + \Delta x`.
* :math:`q(x, \Delta x, t; \theta)` is the probability that a person of age :math:`x` and sex :math:`s`
  in the year :math:`t` will die between the ages :math:`x` and :math:`x + \Delta x`.

The parameter :math:`\beta(\theta)` is unknown, and so we first need to calculate it.
To do so, we set :math:`t = \text{calibration_year}`, and use the ``Brent`` root-finding
algorithm to optimize :math:`\beta(\theta)` such that the life expectancy in the calibration year
(which is known) matches the predicted life expectancy.

.. info:: Example: Beta Parameters
  :collapsible:

  .. code-block:: python
    
    {
      ("CA", "M", "LG"): 0.0123,
      ("CA", "F", "LG"): 0.0112,
      ("BC", "M", "LG"): 0.0134,
      ("BC", "F", "LG"): 0.0109
    }


Once we have found :math:`\beta(\theta)`, we can use this formula to find the projected death
probabilities.


.. info:: Example: Projected Death Probabilities (Original Time Delta)
  :collapsible:

  .. list-table::
    :class: max-width-table
    :widths: 8 18 8 10 25 25
    :header-rows: 1

    * - ``province``
      - ``projection_scenario``
      - ``age``
      - ``sex``
      - ``timepoint``
      - ``prob_death``
    * - BC
      - FA
      - 0
      - F
      - 2023-01-01
      - 0.004919
    * - BC
      - FA
      - 0
      - F
      - 2024-01-01
      - 0.003945
    * - BC
      - FA
      - 0
      - F
      - ...
      - ...
    * - BC
      - FA
      - 0
      - F
      - 2068-01-01
      - 0.003409


4. Converting the Time Delta for Death Probabilities
-----------------------------------------------------

We have annual data for the past years, but we may want to convert it to a different time delta,
e.g. 1 month. Let :math:`\Delta x_a` be the original time interval, and :math:`\Delta x_b` be the
new (desired) time interval. We will make the assumption that the ``hazard rate`` :math:`\mu(x, t)`
is constant over the time interval :math:`[x, x + \Delta x_a)`. The hazard rate is given by:

.. math::
  
    \mu(x, t) = \dfrac{F'_X(x, t)}{1 - F_X(x, t)} = - \dfrac{dS_X(x, t)}{dx} \dfrac{1}{S_X(x, t)}

where :math:`S_X(x, t)` is the survival function for age at death, :math:`X`, at time :math:`t`.
Solving this first order separable linear differential equation, we have:

.. math::
    S_X(x, t) = k e^{-\int \mu(x, t) dx}

.. info:: Math: Solving for :math:`S_X(x, t)`
  :collapsible:

  .. math::

      \int \dfrac{dS}{S_X} &= -\int \mu(x, t) dx \\
      \ln(S_X(x, t)) &= -\int \mu(x, t) dx + C \\
      S_X(x, t) &= k e^{-\int \mu(x, t) dx}

Assuming that :math:`\mu(x, t) = \lambda`, for some constant :math:`\lambda`, we have:

.. math::

    S_X(x, t) &= k e^{- \lambda x} \\
    F_X(x) &= 1 - k e^{- \lambda x}

Now, we can substitute this into the equation for :math:`q(x, \Delta x, t)`:

.. math::

    q(x, \Delta x, t) = \dfrac{F_X(x + \Delta x) - F_X(x)}{1 - F_X(x)} = 1 - e^{- \lambda \Delta x}

.. info:: Math: :math:`q(x, \Delta x, t)`
  :collapsible:

  .. math::

      q(x, \Delta x, t) &= \dfrac{F_X(x + \Delta x) - F_X(x)}{1 - F_X(x)} \\
      &= \dfrac{
          1 - k e^{- \lambda (x + \Delta x)} - 
          1 + k e^{- \lambda x}
      }{k e^{- \lambda x}} \\
      &= \dfrac{
          - k e^{- \lambda (x + \Delta x)}
          + k e^{- \lambda x}
      }{k e^{- \lambda x}} \\
      &= 1 - e^{- \lambda \Delta x}

We can write this in terms of the original time interval :math:`\Delta x_a`:

.. math::

    q(x, \Delta x_b, t) = 1 - (1 - q(x, \Delta x_a, t))^{\Delta x_b / \Delta x_a}

  
.. info:: Math: :math:`q(x, \Delta x_b, t)`
  :collapsible:

  We have the original probability of death for the time interval :math:`\Delta x_a`:

  .. math::

      q(x, \Delta x_a, t) &= 1 - e^{- \lambda \Delta x_a} \\
      e^{- \lambda \Delta x_a} &= 1 - q(x, \Delta x_a, t)

  and we want to find the probability of death for the time interval :math:`\Delta x_b`:

  .. math::

      q(x, \Delta x_b, t) &= 1 - e^{- \lambda \Delta x_b} \\
      &= 1 - e^{- \lambda \Delta x_a \cdot \Delta x_b / \Delta x_a} \\
      &= 1 - (e^{- \lambda \Delta x_a})^{\Delta x_b / \Delta x_a} \\
      &= 1 - (1 - q(x, \Delta x_a, t))^{\Delta x_b / \Delta x_a}


This will give us the probability of death for the new time interval :math:`\Delta x_b`, but only
at the timepoints for which we have data. To get the probability of death for all timepoints, we
note that since :math:`\mu(x, t)` is constant over the time interval :math:`[x, x + \Delta x_a)`:

.. math::

  q(x, \Delta x_b, t + n \Delta x_b) = q(x, \Delta x_b, t) \quad \forall ~ t \in [t, t + \Delta x_a)


.. info:: Math: :math:`q(x, \Delta x_b, t + n \Delta x_b)`
  :collapsible:

  .. math::

    q(x, \Delta x_b, t + n \Delta x_b) &= 1 - e^{- \lambda \Delta x_b} \\
    q(x, \Delta x_b, t) &= 1 - e^{- \lambda \Delta x_b} \\
    q(x, \Delta x_b, t + n \Delta x_b) &= q(x, \Delta x_b, t)


.. info:: Example: Projected Death Probabilities (Monthly Time Delta)
  :collapsible:

  .. list-table::
    :class: max-width-table
    :widths: 8 18 8 10 25 25
    :header-rows: 1

    * - ``province``
      - ``projection_scenario``
      - ``age``
      - ``sex``
      - ``timepoint``
      - ``prob_death``
    * - BC
      - FA
      - 0
      - F
      - 2023-01-01
      - 0.000919
    * - BC
      - FA
      - 0
      - F
      - 2023-02-01
      - 0.000945
    * - BC
      - FA
      - 0
      - F
      - 2023-03-01
      - 0.000294
    * - BC
      - FA
      - 0
      - F
      - ...
      - ...
    * - BC
      - FA
      - 0
      - F
      - 2023-12-01
      - 0.000545
    * - BC
      - FA
      - 0
      - F
      - ...
      - ...
    * - BC
      - FA
      - 0
      - F
      - 2068-01-01
      - 0.003409


leap.data\_generation.death\_data module
****************************************

.. automodule:: leap.data_generation.death_data
   :members:
   :undoc-members:
   :show-inheritance:
