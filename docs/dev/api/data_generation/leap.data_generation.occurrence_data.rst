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



leap.data\_generation.occurrence\_data module
***********************************************

.. automodule:: leap.data_generation.occurrence_data
   :members:
   :undoc-members:
   :show-inheritance:
