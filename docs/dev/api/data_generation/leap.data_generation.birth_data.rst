Birth Data
==========

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


To run the data processing for the population data:

.. code-block:: bash

   cd LEAP
   python3 leap/data_generation/birth_data.py


This will update the following data files: 

1. ``leap/processed_data/birth/birth_estimate.csv``
2. ``leap/processed_data/birth/initial_pop_distribution_prop.csv``


leap.data\_generation.birth\_data module
****************************************

.. automodule:: leap.data_generation.birth_data
   :members:
   :undoc-members:
   :show-inheritance:
