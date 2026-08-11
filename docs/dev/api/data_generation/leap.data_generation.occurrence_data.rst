Occurrence Data
================


To run the data processing for the occurrence data:

.. code-block:: bash

   cd LEAP
   python3 leap/data_generation/occurrence_data.py --time-delta P1Y


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
*********

Asthma Incidence / Prevalence Data
-----------------------------------

The BC Ministry of Health Administrative Dataset contains asthma incidence and prevalence data
for the years ``2000-2019``, in 5-year age intervals. This is a private dataset and is not
included in the ``LEAP`` repository.


.. list-table::
   :class: long-table
   :widths: 25 25 50
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - ``fiscal_year``
     - :code:`int`
     - format ``XXXX``, e.g ``2000``, range <code>[2000, 2019]</code>
   * - ``gender``
     - :code:`str`
     - ``M`` for male, ``F`` for female
   * - ``age_group_desc``
     - :code:`str`
     - an age group category, e.g. ``1-5 years``, ``6-10 years``, etc.
   * - ``incidence``
     - :code:`float`
     - the incidence of asthma in BC for a given year, age group, and sex, per 100 people
   * - ``prevalence``
     - :code:`float`
     - the prevalence of asthma in BC for a given year, age group, and sex, per 100 people


.. info:: Example: Asthma Incidence / Prevalence Data
  :collapsible:


  .. list-table::
    :class: long-table
    :widths: 20 20 20 20 20
    :header-rows: 1

    * - fiscal_year
      - gender
      - age_group_desc
      - incidence
      - prevalence
    * - 2000
      - F
      - 1-5 years
      - 0.034587
      - 0.029384
    * - 2000
      - F
      - 6-10 years
      - 0.0123984
      - 0.097435
    * - 2000
      - F
      - 11-15 years
      - 0.0098374
      - 0.012387
    * - 2000
      - F
      - 16-20 years
      - 0.0029348
      - 0.0239847
    * - 2000
      - F
      - 21-25 years
      - 0.001298
      - 0.023487
    * - 2000
      - F
      - 26-30 years
      - 0.0039485
      - 0.0293847



leap.data\_generation.occurrence\_data module
***********************************************

.. automodule:: leap.data_generation.occurrence_data
   :members:
   :undoc-members:
   :show-inheritance:
