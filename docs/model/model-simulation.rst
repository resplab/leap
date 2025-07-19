===========
Simulation
===========

The ``LEAP`` model simulates asthma outcomes over a lifetime, incorporating different factors such
as age, sex, location, antibiotic use, and other health-related variables.

First, let us take a look at the different input parameters:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Parameter
     - Description
   * - ``province``
     - The province in Canada to run the simulation. Currently only ``BC`` and ``CA`` are
       supported.
   * - ``population_growth_type``
     - Statistics Canada uses different population growth types to model different population
       growth possibilities.
   * - ``num_births_initial``
     - The number of births in the initial year of the simulation. This determines the overall
       population size at the start of the simulation.
   * - ``max_age``
     - The maximum age of a person in the model.
   * - ``until_all_die``
     - Whether or not to keep the simulation running until all members of the population have
       died.
   * - ``min_year``
     - The year to start the simulation.
   * - ``time_horizon``
     - How many years to run the simulation for.


Iterating Over Years
=====================

Initial Year
*************

To start the simulation, we begin with the initial year, ``min_year``. For the first year, we
create a list of ``agents``, or people, who are born in that year, aged 0. The number of agents
in this list is determined by the ``num_births_initial`` parameter.

Subsequent Years
*****************

For all subsequent years, we create a list of agents who are either born in that year or who
immigrated to the province in that year. The number of births in subsequent years is determined by
the :ref:`birth_model`, which factors in the population growth type and the number of births
in the previous year. The number of immigrants is determined by the :ref:`immigration_model`.

Example
*************

Let's suppose we have:

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - Parameter
     - Value
   * - starting year
     - 2021
   * - time horizon
     - 10 years
   * - province
     - BC
   * - population growth type
     - HG (High Growth)
   * - number of births in initial year
     - 100
   * - maximum age
     - 90

In the first year, we would create ``100`` agents, all aged ``0``:

.. list-table:: Year: 2021
   :widths: 25 25 25 25
   :header-rows: 1

   * - Agent
     - age
     - sex
     - immigrant
   * - Agent 1
     - 0
     - F
     - no
   * - Agent 2
     - 0
     - M
     - no
   * - Agent 3
     - 0
     - M
     - no
   * - ...
     - ...
     - ...
     - ...
   * - Agent 100
     - 0
     - F
     - no

In the second year, 2022, we would calculate the number of births using the birth model:

.. math::
    
    n(2022) &= n(2021) * \dfrac{N(2022)}{N(2021)} \\
    &= 100 * \dfrac{46400}{42653} \\
    &= 1.09\\
    &\approx 109

where:  

* :math:`n(2022)` is the number of births in 2022, in the simulation
* :math:`n(2021)` is the number of births in 2021, in the simulation
* :math:`N(2022)` is the total number of births in 2022, for the ``HG`` scenario, from Statistics Canada
* :math:`N(2021)` is the total number of births in 2021, for the ``HG`` scenario, from Statistics Canada

We would then create 109 new agents, all aged 0, and add them to the list of agents for that year.
For the number of immigrants:

.. math::

    i(2022) &= n(2022) * \dfrac{I(2022)}{N(2022)} \\
    &= 109 * \dfrac{67190}{46400} \\
    &\approx 158

where:

* :math:`i(2022)` is the number of immigrants to Canada in 2022, in the simulation
* :math:`I(2022)` is the total number of immigrants to Canada in 2022, for the ``HG`` scenario,
  from our model

Now that we have the total number of immigrants for 2022, we want to distribute them across age
and sex. We do this using the data from the immigration model.

So we have a list of agents for 2022 that looks like this:

.. list-table:: Year: 2021
   :widths: 25 25 25 25
   :header-rows: 1

   * - Agent
     - age
     - sex
     - immigrant
   * - Agent 1
     - 0
     - F
     - no
   * - Agent 2
     - 0
     - M
     - no
   * - Agent 3
     - 0
     - M
     - no
   * - ...
     - ...
     - ...
     - ...
   * - Agent 109
     - 0
     - F
     - no
   * - Agent 110
     - 29
     - M
     - yes
   * - Agent 111
     - 54
     - F
     - yes
   * - ...
     - ...
     - ...
     - ...
   * - Agent 267
     - 45
     - M
     - yes

Since our time horizon is 10 years, we would continue this process for each year until we
reach 2031.

Iterating Over Agents
========================

Once we have our list of agents for a given year, we then simulate the lifetime of each agent, from
birth or immigration until either death or the maximum age is reached.

Iterating Over Lifetime
=========================

For each agent, we simulate their lifetime by iterating over each year of their life. We start with
the year they were born or immigrated, and we continue until they reach the maximum age or
until they die. Each year, we update their age and check if they are still alive based
on the mortality model. If they are still alive, we update their health status, asthma control,
and other health-related variables based on the model's parameters and the agent's characteristics.
During a given year, the following events can occur:

1. Agent dies or reaches the maximum age. If so, we finish simulating that agent and go on to the next agent.
2. Agent has an asthma exacerbation.
3. Agent is hospitalized due to asthma.
4. Agent gets diagnosed with asthma.
5. Agent finds out that previous asthma diagnosis was incorrect.
6. Agent emigrates to another province or country.