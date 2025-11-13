
.. _migration-model:
===============================
Immigration / Emigration Model
===============================

Data
====

``Statistics Canada`` does not contain immigration/emigration data broken down by the necessary
groups (age, sex, etc), so we do not have exact data for this category. Instead, we use the
data from the birth and death models.

Population Data
*****************

We use the Statistics Canada population data that was generated and saved as:
`processed_data/birth/initial_pop_distribution_prop.csv 
<https://github.com/resplab/leap/blob/main/leap/processed_data/birth/initial_pop_distribution_prop.csv>`_.

This table contains the number of people in a given age, sex, province,
and projection scenario, along with the number of births for that year. This data is the net number
of people, factoring in death, immigration, and emigration.


.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - ``year``
     - :code:`int`
     - the calendar year
   * - ``age``
     - :code:`int`
     - the age of the person in years
   * - ``province``
     - :code:`str`
     - the province of the person
       (e.g., ``AB`` = Alberta, ``BC`` = British Columbia, etc.)
   * - ``n_age``
     - :code:`int`
     - the number of people in a given age group, year, province, and projection scenario
   * - ``n_birth``
     - :code:`int`
     - the number of births in that year, province, and projection scenario
   * - ``prop``
     - :code:`float`
     - the proportion of the population in that age group, year, province, and projection scenario
       relative to the number of births in that year, province, and projection scenario
   * - ``prop_male``
     - :code:`float`
     - the proportion of the population in a given age group, year, province, and projection scenario
       who are male
   * - ``projection_scenario``
     - :code:`str`
     - the projection scenario used to generate the data


Mortality Data
*****************

We use the Statistics Canada population data that was generated and saved as:
`processed_data/life_table.csv 
<https://github.com/resplab/leap/blob/main/leap/processed_data/life_table.csv>`_.


.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - ``year``
     - :code:`int`
     - the calendar year
   * - ``age``
     - :code:`int`
     - the age of the person in years
   * - ``province``
     - :code:`str`
     - the province of the person
       (e.g., ``AB`` = Alberta, ``BC`` = British Columbia, etc.)
   * - ``sex``
     - :code:`str`
     - ``F`` = female, ``M`` = male
   * - ``prob_death``
     - :code:`float`
     - the probability that a person of the given age and sex, living in the given province, will
       die during the given year.
   * - ``se``
     - :code:`float`
     - the standard error on the probability of death

Model
=====

To obtain the net migration, for anyone aged > 0, we compute the number of people in each age
group projected to die during that year based on the ``prob_death`` given by the mortality model.
Then we calculate the net change in people using the ``n_age`` column in the
``initial_pop_distribution_prop.csv``. We subtract the number of people who died from the net
population change to get the net number of people who migrated:

.. math::

    \Delta n = n - n_{\text{prev}} * (1 - q_x)

where :math:`\Delta n` is the net migration, :math:`n` is the number of people in that age
group, :math:`n_{\text{prev}}` is the number of people in that age group in the previous year,
and :math:`q_x` is the probability of death for that age group in that year.