.. _pollution-model:

===============================
Pollution Model
===============================

Data
====

PM2.5 Data
*************

The baseline projections are obtained from the ``ECCC`` using their ``GEM-MACH`` model, which
gives us a prediction for background PM2.5 levels for the years 2026, 2031, and 2036. All other timepoints are calculated using linear interpolation:

Wildfire PM2.5 is based on a separate model, ``RAQDPS``, which predicts
PM2.5 from anthropogenic and non-wildfire natural sources, and ``RAQDPS-FW``, which predicts PM2.5
from those same sources plus wildfires. The wildfire contribution for each month and census division
is the difference between the two models' predictions (``RAQDPS-FW`` minus ``RAQDPS``), averaged over
2018-2023 and floored at zero to remove small negative values caused by model noise. This historical
monthly wildfire contribution is then scaled by a future climate scaling factor, derived from
projected changes in wildfire-related PM2.5 under different IPCC Shared Socioeconomic Pathway (SSP)
scenarios, to estimate wildfire PM2.5 at future timepoints.

The climate scaling factor for each SSP scenario is derived from Table S2 of :cite:`xie2022`, which
reports, for three CMIP6-driven multiple linear regression models (``CESM2``, ``GFDL-ESM4.1``, and
``CNRM-ESM2-1``), the percent change in August-September mean PM2.5 over the western US by the late
21st century (2080-2100) relative to 1990-2010, under each SSP scenario. We take the average percent
change across the three models as the total wildfire PM2.5 increase by 2100 for that scenario
(``SSP1_2.6`` ~45%, ``SSP2_4.5`` ~85%, ``SSP3_7.0`` ~124%, ``SSP5_8.5`` ~136%), and linearly
interpolate the scaling factor between 1 (present day) and this value at 2100 for intermediate
timepoints.

The resulting data is stored as a separate dataset (one ``.csv`` file per SSP scenario), each
containing the columns below, including an ``SSP`` column identifying which scenario that
dataset corresponds to:

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - ``CDUID``
     - :code:`int`
     - the Statistics Canada Census Division Unique Identifier
   * - ``month``
     - :code:`int`
     - the month that the data applies to
   * - ``year``
     - :code:`int`
     - the year that the data applies to
   * - ``timepoint``
     - :code:`datetime`
     - the starting date / time of the time interval that the data applies to
   * - ``background_pm25``
     - :code:`float`
     - the average background PM2.5 levels for a given month, year, and CDUID
   * - ``wildfire_pm25``
     - :code:`float`
     - the average PM2.5 levels due to wildfires for a given month, year, and CDUID
   * - ``factor``
     - :code:`float`
     - the future climate scaling factor based on IPCC climate change scenarios
   * - ``wildfire_pm25_scaled``
     - :code:`float`
     - the average PM2.5 levels due to wildfires for a given month, year, and CDUID, scaled by the
       climate scaling factor:

       .. code-block:: python

         wildfire_pm25_scaled = wildfire_pm25 * factor

   * - ``total_pm25``
     - :code:`float`
     - the total average PM2.5 levels for a given month, year, and CDUID:

       .. code-block:: python

         total_pm25 = wildfire_pm25_scaled + background_pm25

   * - ``SSP``
     - :code:`str`
     - the Shared Socioeconomic Pathway scenario used to determine the climate scaling factor.
       Possible values for this column are:

       - ``SSP1_2.6``
       - ``SSP2_4.5``
       - ``SSP3_7.0``
       - ``SSP5_8.5``


Census Division Data
*********************

The pollution data is given with regards to ``CDUID`` regions. In our model, we assign each
agent to an arbitrary location within the selected province. Crudely, one could just randomly
assign each agent to a ``CDUID`` within the province. However, this would ignore the
population distribution within the province, and the fact that air quality depends quite a bit on
location. For example, background PM2.5 levels in Vancouver are typically much worse than any other
regions in British Columbia, but wildfire PM2.5 levels are typically much worse in the interior of
the province. To account for this, we used data on the census divisions from the
``Statistics Canada`` website:

https://www150.statcan.gc.ca/t1/tbl1/en/cv.action?pid=9810000701

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - ``year``
     - :code:`int`
     - the year that the census data was collected
   * - ``census_division_name``
     - :code:`str`
     - the official name of the census division
   * - ``DGUID``
     - :code:`str`
     - the dissemination geography unique identifier, in the format:

       ``2021 | A | 0003 | CDUID``

       * ``2021`` - year the data was collected
       * ``A`` - administrative (not important, a StatsCan identifier)
       * ``0003`` - schema indicating census division
       * ``CDUID`` - the census division identifier

       See `here <https://www150.statcan.gc.ca/n1/pub/92f0138m/92f0138m2019001-eng.htm>`_
       for more details.
   * - ``CDUID``
     - :code:`int`
     - the Statistics Canada Census Division Unique Identifier
   * - ``geographic_area_type``
     - :code:`str`
     - the type of region, for example ``CTY`` = city
   * - ``province``
     - :code:`str`
     - the two-letter province or territory code, e.g. ``BC`` = British Columbia
   * - ``population``
     - :code:`int`
     - the number of people living in that census division
   * - ``area_km2``
     - :code:`float`
     - the area of the census division in square kilometres
   * - ``population_density_per_km2``
     - :code:`float`
     - the number of people per square kilometre