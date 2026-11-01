Occurrence Data
================


To run the data processing for the occurrence data:

.. code-block:: bash

   cd LEAP
   python3 leap/data_generation/occurrence_data.py


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

Data Processing
****************


1. Load Asthma Incidence / Prevalence Data
-------------------------------------------

First, we load the asthma incidence / prevalence data from 
``original_data/private/asthma_inc_prev.csv``.


We end up with:

.. list-table::
   :class: long-table
   :widths: 15 10 75
   :header-rows: 1

   * - Name
     - Type
     - Description
   * - ``year``
     - ``int``
     - The starting year of the interval during which the data was collected.
   * - ``age``
     - ``int``
     - The average age of the age group, in years.
   * - ``age_group``
     - :code:`str`
     - Five-year age groups:

       * ``1-5 years``
       * ``6-10 years``
       * ``11-15 years``
       * ``16-20 years``
       * ``21-25 years``
       * ``26-30 years``
       * ``31-35 years``
       * ``36-40 years``
       * ``41-45 years``
       * ``46-50 years``
       * ``51-55 years``
       * ``56-60 years``
       * ``61-65 years``
   * - ``sex``
     - ``str``
     - One of ``"M"`` = male, ``"F"`` = female.
   * - ``incidence``
     - ``float``
     - The incidence of asthma in BC for a given year, age group, and sex, per 100 people.
   * - ``prevalence``
     - ``float``
     - The prevalence of asthma in BC for a given year, age group, and sex, per 100 people.


.. info:: Example: Asthma Incidence / Prevalence Data
  :collapsible:

  .. list-table::
    :class: max-width-table
    :widths: 10 10 10 10 20 30
    :header-rows: 1

    * - ``year``
      - ``age``
      - ``age_group``
      - ``sex``
      - ``timepoint``
      - ``incidence``
      - ``prevalence``
    * - 2000
      - 3
      - 1-5 years
      - F
      - 0.0565
      - 0.0565
    * - 2000
      - 8
      - 6-10 years
      - F
      - 0.0124
      - 0.1029
    * - 2000
      - ...
      - ...
      - F
      - ...
      - ...
    * - 2001
      - 3
      - 1-5 years
      - F
      - 0.0463
      - 0.0463
    * - ...
      - ...
      - ...
      - F
      - ...
      - ...
    * - 2019
      - 63
      - 61-65 years
      - F
      - 0.0283
      - 0.3647
    * - 2000
      - 3
      - 1-5 years
      - M
      - 0.0443
      - 0.0443
    * - ...
      - ...
      - ...
      - M
      - ...
      - ...
    * - 2019
      - 63
      - 61-65 years
      - M
      - 0.0204
      - 0.2837


leap.data\_generation.occurrence\_data module
***********************************************

.. automodule:: leap.data_generation.occurrence_data
   :members:
   :undoc-members:
   :show-inheritance:
