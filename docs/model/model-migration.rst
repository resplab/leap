
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
`processed_data/{time_delta_tag}/birth/initial_population.csv 
<https://github.com/resplab/leap/blob/main/leap/processed_data/time_delta_365/birth/initial_population.csv>`_.

This table contains the number of people in a given age, sex, province,
and projection scenario, along with the number of births for that timepoint. This data is the net number
of people, factoring in death, immigration, and emigration.


.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - ``timepoint``
     - :code:`int`
     - the starting date / time of the time interval that the data applies to
   * - ``age``
     - :code:`int`
     - the age of the person in years
   * - ``province``
     - :code:`str`
     - the province of the person
       (e.g., ``AB`` = Alberta, ``BC`` = British Columbia, etc.)
   * - ``n_age``
     - :code:`int`
     - the number of people in a given age group, time interval, province, and projection scenario
   * - ``n_birth``
     - :code:`int`
     - the number of births in that time interval, province, and projection scenario
   * - ``prop``
     - :code:`float`
     - the proportion of the population in that age group, time interval, province, and projection scenario
       relative to the number of births in that time interval, province, and projection scenario
   * - ``prop_male``
     - :code:`float`
     - the proportion of the population in a given age group, time interval, province, and
       projection scenario who are male
   * - ``projection_scenario``
     - :code:`str`
     - the projection scenario used to generate the data


Mortality Data
*****************

We use the Statistics Canada population data that was generated and saved as:
`processed_data/{time_delta_tag}/life_table.csv 
<https://github.com/resplab/leap/blob/main/leap/processed_data/time_delta_365/life_table.csv>`_.


.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - ``timepoint``
     - :code:`datetime`
     - the starting date / time of the time interval that the data applies to
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
       die during the given time interval.
   * - ``se``
     - :code:`float`
     - the standard error on the probability of death

Model
=====

The following variables will be used in the equations below:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Column
     - Description
   * - :math:`a`
     - age in years
   * - :math:`s`
     - sex, one of ``M`` or ``F``
   * - :math:`t`
     - timepoint, in years
   * - :math:`n(a, s, t)`
     - number of people in the population of age :math:`a`, sex :math:`s`, at timepoint :math:`t`
   * - :math:`n(a, t)`
     - number of people in the population of age :math:`a` at timepoint :math:`t`, both sexes
   * - :math:`\Delta n(a, s, t)`
     - net migration at timepoint :math:`t` for people of age :math:`a` and sex :math:`s`
   * - :math:`n(\text{age}-\Delta t,\ \text{timepoint}-\Delta t)`
     - number of people one timestep younger at the previous timepoint
   * - :math:`n_{\text{birth}}(t)`
     - number of births during the time interval starting at timepoint :math:`t`
   * - :math:`q_x(\text{age}-\Delta t,\ \text{timepoint}-\Delta t)`
     - probability that a person of age :math:`\text{age}-\Delta t` at the previous timepoint will
       die during the time interval between the previous timepoint and the current timepoint

Net Migration
****************

To obtain the net migration during a time interval, for anyone aged > 0, we use the difference
between the population at the current timepoint and the population at the previous timepoint, after
accounting for deaths. This is computed separately for each combination of age, sex, province,
and projection scenario. Age 0 is excluded because newborns are handled separately by the
:ref:`birth-model`.

.. math::

    \Delta n(a,\ s,\ t) = n(a,\ s,\ t) - 
      n(a-\Delta t,\ s,\ t-\Delta t) \cdot \left(1 - q_{x}(a-\Delta t,\ s,\ t-\Delta t)\right)

If :math:`\Delta n > 0`, the surplus is attributed to immigration. If :math:`\Delta n < 0`,
the deficit is attributed to emigration.

.. note::

    Of course, this assumption is not in general true - for example, if :math:`\Delta n = 50`, it
    could be the case that :math:`100` people immigrated and :math:`50` people emigrated, as opposed
    to our assumption that :math:`50` people immigrated and :math:`0` people emigrated. However, for
    the purposes of our model, it doesn't matter how many people immigrated or emigrated, just the
    net effect on the population, so we make the simplifying assumption that all of the net change
    is due to either immigration or emigration, but not both.


Number of Immigrants
***********************

At each timepoint of the simulation, new agents are added to the model, and fall into one of two
categories: newborns (handled by the :ref:`birth-model`) or immigrants. The number of immigrants
at a given timepoint of a given age and sex is given by:


.. math::

    i(a, s, t) = i(t) \cdot \text{prop immigrants timepoint}(a,\ s,\ t)

The total number of immigrant agents created in a given time interval :math:`i(t)` is:

.. math::

    i(t) = \left\lceil 
      n_{\text{birth}}(t) \cdot \sum_{\substack{a,\ s \\ \Delta n > 0}}\ 
      \text{prop migrants birth}(a,\ s,\ t)
    \right\rceil

where :math:`n_{\text{birth}}(t)` is the number of simulated births in that time interval.


``prop_migrants_birth`` is computed as:


.. math::

    \text{prop migrants birth}(a,\ s,\ t)
    = \dfrac{\text{no. migrants}}{\text{no. births at timepoint } t}
    = \dfrac{\Delta n(a,\ s,\ t)}{n_{\text{birth}}(t)}


``prop_immigrants_timepoint`` is computed as:

.. math::

  \begin{align}
    \text{prop immigrants timepoint}(a,\ s,\ t)
    &= \dfrac{\text{no. immigrants of age $a$, sex $s$ at timepoint } t}{\text{no. immigrants at timepoint } t} \\
    &= \begin{cases}
      \dfrac{\Delta n(a,\ s,\ t)}{\sum_{a, s}\ \Delta n(a,\ s,\ t)}
      & \quad \text{if } \Delta n > 0 \\
      0 & \quad \text{ otherwise}
    \end{cases}
  \end{align}



Probability of Emigration
**************************

At runtime, the simulation uses ``prob_emigration`` directly in a Bernoulli trial for each timepoint
to determine whether an agent emigrates:

.. math::

    \text{emigrates} \sim \text{Bernoulli}(p_{\text{emigrate}}(\text{sex}, \text{age}, \text{timepoint}))

``prob_emigration`` is computed as:

.. math::

    \text{prob emigration}(a,\ s,\ t) 
    = \begin{cases}
      \dfrac{|\Delta n(a,\ s,\ t)|}{n(a,\ s,\ t)} & \text{if } \Delta n < 0 \\
      0 & \text{ otherwise}
    \end{cases}


Agents aged 0 are excluded — newborns never emigrate.

Both immigration and emigration are rooted in the same StatCan population-level counts
(:math:`\Delta n`), and are converted into agent-level operations to fit LEAP's microsimulation
framework:


* For **immigration**, rows where ``delta_n > 0`` are used. ``prop_migrants_birth`` (positive)
  determines how many immigrant agents to create at each timepoint, and ``prop_immigrants_timepoint``
  determines the age and sex of each agent.
* For **emigration**, rows where ``delta_n < 0`` are used. ``prob_emigration`` is applied to
  each existing agent individually via a Bernoulli trial for each timepoint.


Processed Data
==============

The migration model produces a single processed data file generated by
`leap/data_generation/migration_data.py
<https://github.com/resplab/leap/blob/main/leap/data_generation/migration_data.py>`_,
covering the provinces ``CA`` (all of Canada) and ``BC`` (British Columbia).

Migration Table
***************

Saved as:
`leap/processed_data/{time_delta_tab}/migration/migration_table.csv
<https://github.com/resplab/leap/blob/main/leap/processed_data/time_delta_365/migration/migration_table.csv>`_.

Each row corresponds to a unique combination of ``year``, ``province``, ``age``, ``sex``, and
``projection_scenario``. The table records the signed net migration (:math:`\Delta n`) for each
group, along with derived columns used by the simulation at runtime for both immigration and
emigration.

:math:`\Delta n` is computed independently for each sex. Because males and females are computed
separately, it is possible for one sex to have net immigration while the other has net emigration
for the same age and year.

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - ``timepoint``
     - :code:`datetime`
     - the calendar year
   * - ``province``
     - :code:`str`
     - the province abbreviation (``BC`` or ``CA``)
   * - ``age``
     - :code:`int`
     - the age in years
   * - ``sex``
     - :code:`str`
     - ``M`` = male, ``F`` = female
   * - ``projection_scenario``
     - :code:`str`
     - the StatCan population projection scenario
   * - ``delta_n``
     - :code:`float`
     - the signed net migration for this age, sex, timepoint, province, and projection scenario;
       positive values indicate net immigration, negative values indicate net emigration
   * - ``prop_migrants_birth``
     - :code:`float`
     - ``delta_n`` divided by the number of births during that time interval; signed — positive for
       net immigration cells, negative for net emigration cells
   * - ``prop_immigrants_timepoint``
     - :code:`float`
     - for cells where ``delta_n > 0``, each age and sex group's share of all immigrants
       arriving in a given year (denominator is the sum of positive ``delta_n`` values only);
       zero for emigration cells
   * - ``prop_emigrants_timepoint``
     - :code:`float`
     - for cells where ``delta_n < 0``, each age and sex group's share of all emigrants
       leaving in a given year (denominator is the sum of negative ``delta_n`` values only);
       zero for immigration cells
   * - ``prob_emigration``
     - :code:`float`
     - for cells where ``delta_n < 0``, the per-person annual probability of emigrating,
       computed as :math:`|\Delta n| / N`; zero for immigration cells


