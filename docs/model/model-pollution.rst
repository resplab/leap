===============================
Pollution Model
===============================

Data
====

PM2.5 Data
*************

The baseline projections are obtained from the ``ECCC`` using their ``GEM-MACH`` model, which
gives us a prediction for background PM2.5 levels and wildfire PM2.5 levels for the years
2026, 2031, and 2036. All other years are calculated using linear interpolation:

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - ``year``
     - :code:`int`
     - the calendar year
   * - ``month``
     - :code:`int`
     - the numeric month, with ``1`` = January, ``2`` = February, etc.
   * - ``date``
     - :code:`str`
     - the date in ``YYYY-MM-DD`` format
   * - ``CDUID``
     - :code:`int`
     - the Statistics Canada Census Division Unique Identifier
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