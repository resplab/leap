Migration Data
==============

``StatCan`` does not contain immigration/emigration data broken down by the necessary
groups (age, sex, etc), so we do not have exact data for this category. Instead, we use the
following data files:

1. ``leap/processed_data/life_table.csv`` (generated by ``death_data.py``)
2. ``leap/processed_data/birth/initial_pop_distribution_prop.csv`` (generated by ``birth_data.py``)

The ``life_table.csv`` contains the probability of death during that year for each age, sex,
province, and projection scenario.

The ``initial_pop_distribution_prop.csv`` contains the number of people in a given age, sex,
province, and projection scenario, along with the number of births for that year. This data is the
net number of people, factoring in death, immigration, and emigration.

To obtain the net migration, for anyone of ``age > 0``, we compute the number of people in each age
group projected to die during that year based on the ``prob_death`` column in the ``life_table.csv``.
Then we calculate the net change in people using the ``n_age`` column in the
``initial_pop_distribution_prop.csv``. We subtract the number of people who died from the net
population change to get the net number of people who migrated:

.. code-block:: python

   delta_n = n - n_prev * (1 - prob_death)


To run the data generation for the migration data:

.. code-block:: bash

   cd LEAP
   python3 leap/data_generation/migration_data.py


leap.data\_generation.migration\_data module
********************************************

.. automodule:: leap.data_generation.migration_data
   :members:
   :undoc-members:
   :show-inheritance:
