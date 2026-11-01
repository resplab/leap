.. _birth-model:

===========================
Birth Model
===========================

Data
====

To obtain the population data for each time interval, we used two tables from ``Statistics Canada``:

Past Data: 1999 - 2021
*************************

For past years, we used 
`Table 17-10-00005-01 from Statistics Canada
<https://www150.statcan.gc.ca/t1/tbl1/en/cv.action?pid=1710000501>`_.
This is data that the Canadian government collects and publishes on the population of
the country, stratified by different variables.

The ``*.csv`` file can be downloaded from here:
`17100005-eng.zip <https://www150.statcan.gc.ca/n1/tbl/csv/17100005-eng.zip>`_

and is saved as:
`LEAP/leap/original_data/17100005.csv 
<https://github.com/resplab/leap/blob/main/leap/original_data/17100005.csv>`_.

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
     - the province or terriroty full name
   * - ``SEX``
     - :code:`str`
     - one of "Both sexes", "Females", or "Males"
   * - ``VALUE``
     - :code:`int`
     - the population in that year, province, sex, and age group


Projected Data: 2021 - 2065
****************************


For future years, we used
`Table 17-10-0057-01 from Statistics Canada
<https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1710005701>`_.
Statistic Canada provides projected population data based on different projection scenarios.

The ``*.csv`` file can be downloaded from here:
`17100057-eng.zip <https://www150.statcan.gc.ca/n1/tbl/csv/17100057-eng.zip>`_

and is saved as:
`LEAP/leap/original_data/17100057.csv 
<https://github.com/resplab/leap/blob/main/leap/original_data/17100057.csv>`_.

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
     - the province or terriroty full name
   * - ``Sex``
     - :code:`str`
     - one of "Both sexes", "Females", or "Males"
   * - ``Projection scenario``
     - :code:`str`
     - the projection scenario used to model population growth:
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
   * - ``VALUE``
     - :code:`int`
     - the population in that year, province, sex, age group, and projection scenario


Processed Data
***************

The two source tables are combined by `leap/data_generation/birth_data.py
<https://github.com/resplab/leap/blob/main/leap/data_generation/birth_data.py>`_
into a single processed file saved as:
`leap/processed_data/{time_delta_tag}/birth/birth_estimate.csv
<https://github.com/resplab/leap/blob/main/leap/processed_data/time_delta_365/birth/birth_estimate.csv>`_.

Past data (from ``17100005.csv``) covers years 1999 onwards using actual population counts.
Projected data (from ``17100057.csv``) begins the year after the last available past year and
covers projections up to 2065. In the projected source file, ``VALUE`` is stored in thousands
and is multiplied by 1000 during processing.

For both sources, only the ``AGE_GROUP = 0`` (newborns) rows are used. The ``N`` column
represents the total number of births (both sexes combined), and ``prop_male`` is derived as
the number of male births divided by the total.

The same two source tables, using all ``AGE_GROUP`` rows rather than just newborns, are also used
by ``birth_data.py`` to produce a second processed file:
`leap/processed_data/{time_delta_tag}/birth/initial_population.csv
<https://github.com/resplab/leap/blob/main/leap/processed_data/time_delta_365/birth/initial_population.csv>`_.
This file gives the population count by age (rather than just births), and is used by the
migration and exacerbation models — see the Population Data section in :doc:`model-migration` for
its schema.

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - ``timepoint``
     - :code:`int`
     - the starting date / time that the data applies to
   * - ``province``
     - :code:`str`
     - the 2-letter province or territory ID
       (e.g., ``BC`` = British Columbia, ``AB`` = Alberta, ``CA`` = Canada)
   * - ``N``
     - :code:`int`
     - the total number of births (both sexes) in that time interval and province
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

