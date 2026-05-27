Birth Data
==========

Original Data
*************

To obtain the population data for each year, we used two tables from ``StatCan``:

1. **1999 - 2021:**

   For past years, we used 
   `Table 17-10-00005-01 from StatCan <https://www150.statcan.gc.ca/t1/tbl1/en/cv.action?pid=1710000501>`_.

   The ``*.csv`` file can be downloaded from here:
   `17100005-eng.zip <https://www150.statcan.gc.ca/n1/tbl/csv/17100005-eng.zip>`_

   and is saved as:
   ``LEAP/leap/original_data/17100005.csv``

2. **2021 - 2065:**

   For future years, we used
   `Table 17-10-0057-01 from StatCan <https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1710005701>`_.

   The ``*.csv`` file can be downloaded from here:
   `17100057-eng.zip <https://www150.statcan.gc.ca/n1/tbl/csv/17100057-eng.zip>`_.

   and is saved as:
   ``LEAP/leap/original_data/17100057.csv``


Generating Processed Data
**************************

To run the data processing for the population data, with data points taken every year:

.. code-block:: bash

   cd LEAP
   python leap/data_generation/birth_data.py --time-delta P1Y


This will update the following data files: 

1. ``leap/processed_data/{time_delta_tag}/birth/birth_estimate.csv``
2. ``leap/processed_data/{time_delta_tag}/birth/initial_pop_distribution_prop.csv``

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


Processed Data
**************

The output of the data generation for the ``Birth`` module is two ``.csv`` files:

``birth_estimate.csv``

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - ``timepoint``
     - :code:`dt.datetime`
     - the date and time of the start of the time interval (e.g. ``2020-01-01 00:00:00``)
   * - ``province``
     - :code:`str`
     - the 2-letter province or territory ID
       (e.g., ``BC`` = British Columbia, ``AB`` = Alberta, ``CA`` = Canada)
   * - ``N``
     - :code:`int`
     - total number of births (both sexes) during the given time interval and in the given province
   * - ``prop_male``
     - :code:`float`
     - the proportion of births that are male
   * - ``projection_scenario``
     - :code:`str`
     - ``past`` for historical data, or one of the projection scenario IDs for future data:

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



``initial_pop_distribution_prop.csv``

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - ``timepoint``
     - :code:`dt.datetime`
     - the date and time of the start of the time interval (e.g. ``2020-01-01 00:00:00``)
   * - ``province``
     - :code:`str`
     - the 2-letter province or territory ID
       (e.g., ``BC`` = British Columbia, ``AB`` = Alberta, ``CA`` = Canada)
   * - ``age``
     - :code:`int`
     - age in years
   * - ``prop_male``
     - :code:`float`
     - the proportion of births that are male
   * - ``n_age``
     - :code:`float`
     - number of people of the given age living in the given province during the given time interval
   * - ``n_birth``
     - :code:`float`
     - number of births in the given province during the given time interval
   * - ``projection_scenario``
     - :code:`str`
     - ``past`` for historical data, or one of the projection scenario IDs for future data:

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




leap.data\_generation.birth\_data module
****************************************

.. automodule:: leap.data_generation.birth_data
   :members:
   :undoc-members:
   :show-inheritance:
