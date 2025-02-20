Death Data
==========

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
   The following equation can be used to obtain the probability of death in future years:

   .. math::

      \sigma^{-1}(p(s, a, y)) = \sigma^{-1}(p(s, a, y_0)) - e^{\beta(s)(y - y_0)}


   where:
   
   * :math:`\sigma^{-1}` is the inverse logit function
   * :math:`a` is the age
   * :math:`s` is the sex
   * :math:`y_0` is the year the collected data ends (in our case, 2020)
   * :math:`y` is the future year
   * :math:`p(s, a, y_0)` is the probability of death for a person of that age/sex in
     the year the collected data ends (in our case, 2020)
   * :math:`p(s, a, y)` is the probability of death for a person of that age/sex in a future year.

   The parameter :math:`\beta(s)` is unknown, and so we first need to calculate it.
   To do so, we set :math:`y = \text{calibration_year}`, and use the ``Brent`` root-finding
   algorithm to optimize :math:`\beta(s)` such that the life expectancy in the calibration year
   (which is known) matches the predicted life expectancy.

   Once we have found :math:`\beta(s)`, we can use this formula to find the projected death
   probabilities.

To run the data generation for the mortality data:

.. code-block:: bash

   cd LEAP
   python3 leap/data_generation/death_data.py


This will update the following data file: 

1. ``leap/processed_data/life_table.csv``


leap.data\_generation.death\_data module
****************************************

.. automodule:: leap.data_generation.death_data
   :members:
   :undoc-members:
   :show-inheritance:
