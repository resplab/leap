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
the :ref:`birth-model`, which factors in the population growth type and the number of births
in the previous year. The number of immigrants is determined by the :ref:`migration-model`.

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


Agent Initialization
*********************

To initialize an agent, we set the following initial attributes:


.. list-table:: Agent Initialization
   :widths: 25 25
   :header-rows: 1

   * - Attribute
     - Description
   * - age
     - Either newborn, or the age at which the agent immigrated to the province.
   * - sex
     - Randomly assigned based on either sex proportions at birth or immigration data.
   * - immigrant
     - A boolean indicating whether the agent is an immigrant.
   * - census division
     - Each agent is randomly assigned to a census division based on the population distribution
       in the province.
   * - asthma control levels
     - Initial asthma control levels are set to:

       * probability of fully-controlled asthma: 0.33
       * probability of partially-controlled asthma: 0.33
       * probability of uncontrolled asthma: 0.33
   * - has asthma?
     - If the agent is less than 3 years old, they do not have asthma. Otherwise, they are assigned
       an asthma status based on the asthma prevalence model.
   * - exacerbation history
     - Initial exacerbation history is set to:

       * number of exacerbations previous year: 0
       * number of exacerbations current year: 0


Step 1: Check if the agent is over 3 years old
----------------------------------------------

If the agent is over 3 years old, go to Step 2. If not, skip to Step 3.

Step 2: Check if the agent has asthma
---------------------------------------

We use the asthma prevalence model to determine if the agent has asthma.

Step 3: Determine the age of asthma diagnosis
---------------------------------------------

If the agent was determined to have asthma in Step 2, we need to determine the age at which
the agent was diagnosed with asthma.

If the agent is 3 years old, the age of asthma diagnosis is 3.

If the agent is older than 3, we use the asthma incidence model to determine the age of asthma
diagnosis.

Step 4: Check hospitalizations
--------------------------------

Next, we check how many times agent has been hospitalized due to asthma in their lifetime.

Step 5: Determine asthma control levels
-----------------------------------------

We next determine the asthma control levels for the agent. The asthma control levels give the
probability of the agent having fully-controlled, partially-controlled, or uncontrolled asthma:

.. math::

    k &= 0: \text{fully-controlled asthma} \\
    k &= 1: \text{partially-controlled asthma} \\
    k &= 2: \text{uncontrolled asthma}


The probabilities of each control level are given by ordinal regression, where y = control level:

.. math::

    P(y \leq k) &= \sigma(\theta_k - \eta) \\
    P(y = k) &= P(y \leq k) - P(y < k + 1) \\
              &= \sigma(\theta_k - \eta) - \sigma(\theta_{k+1} - \eta)


where:

.. math::

    \eta = \beta_0 + 
        \text{age} \cdot \beta_{\text{age}} + 
        \text{sex} \cdot \beta_{\text{sex}} +
        \text{age} \cdot \text{sex} \cdot \beta_{\text{sexage}} + 
        \text{age}^2 \cdot \text{sex} \cdot \beta_{\text{sexage}^2} + 
        \text{age}^2 \cdot \beta_{\text{age}^2}


Step 6: Compute the number of asthma exacerbations in the current year
-----------------------------------------------------------------------

To compute the number of asthma exacerbations in the current year, we use the
exacerbation model, which takes into account the agent's asthma control level, age, and sex,
as well as the current year. The probability of having :math:`n` exacerbations is given by a
Poisson distribution:

.. math::

    P(n = k) = \dfrac{\lambda^{k}e^{-\lambda}}{k!}

We determine the expected number of exacerbations, :math:`\lambda`, using the following formula:

.. math::

    \lambda &= e^{\mu} \\
    \mu &= \beta_0 +
        \beta_{\text{age}} \cdot \text{age} +
        \beta_{\text{sex}} \cdot \text{sex} \\
        &+\beta_{uc} \cdot P(\text{uncontrolled}) +
        \beta_{pc} \cdot P(\text{partially controlled}) +
        \beta_{c} \cdot P(\text{fully controlled}) \\
        &+\log(\alpha(\text{year}, \text{sex}, \text{age}))

If the number of exacerbations is ``0``, skip to the end. Otherwise, go to Step 7.


Step 7: Compute the severity of the asthma exacerbations in the current year
-----------------------------------------------------------------------------

There are four levels of severity for asthma exacerbations:

.. math::

    k &= 1: \text{mild} \\
    k &= 2: \text{moderate} \\
    k &= 3: \text{severe} \\
    k &= 4: \text{very severe / hospitalization}

We assign the initial probability of each severity level using a Dirichlet distribution.

If the agent has been previously hospitalized due to asthma:

.. math::

    P(k = 4 \mid t) &= \begin{cases}
        P(k = 4 \mid t_0) \cdot \beta_{\text{prevhospped}} & \text{if age} < 14 \\
        P(k = 4 \mid t_0) \cdot \beta_{\text{prevhospadult}} & \text{if age} \geq 14
    \end{cases} \\
    P(k = 1 \mid t) &= P(k = 1 \mid t_0) \cdot (1 - P(k = 4 \mid t)) \\
    P(k = 2 \mid t) &= P(k = 2 \mid t_0) \cdot (1 - P(k = 4 \mid t)) \\
    P(k = 3 \mid t) &= P(k = 3 \mid t_0) \cdot (1 - P(k = 4 \mid t))


Otherwise, we use the initial probabilities of each severity level. Then, to determine the
number of exacerbations of each severity level, we use a multinomial distribution:

.. math::

    p(x_1, x_2, x_3, x_4; n, p) = 
        \dfrac{n!}{x_1! x_2! x_3! x_4!} \prod_{i=1}^4 P(k = i)^{x_i}, \quad \text{where} \quad \sum_{i=1}^4 x_i = n


Step 8: Update the number of hospitalizations
-----------------------------------------------------------------------------

We add the total previous hospitalizations to the current year's number of exacerbations at
severity level 4:

.. math::

    \text{hospitalizations} = \text{previous hospitalizations} + x_4

where :math:`x_4` is the number of exacerbations at severity level 4.


Iterating Over Lifetime
************************

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

Let's begin the simulation.

.. _step-1-lifetime-iteration:

Step 1: Check if agent has asthma
---------------------------------------------------

If the agent has asthma, go to Step 7. If not continue to Step 2.

Step 2: Check if agent gets a new asthma diagnosis
---------------------------------------------------

We use the asthma incidence model to determine if the agent gets a new asthma diagnosis
in the current year of their lifetime. If they do not, skip to Step 4. If they do, we
set the age of diagnosis to the current age of the agent and go to Step 3.

Step 3: Compute the control levels
---------------------------------------------------

We next determine the asthma control levels for the agent using the :ref:`control-model`.
The asthma control levels give the probability of the agent having fully-controlled,
partially-controlled, or uncontrolled asthma:

.. math::

    k &= 0: \text{fully-controlled asthma} \\
    k &= 1: \text{partially-controlled asthma} \\
    k &= 2: \text{uncontrolled asthma}


The probabilities of each control level are given by ordinal regression, where y = control level:

.. math::

    P(y \leq k) &= \sigma(\theta_k - \eta) \\
    P(y = k) &= P(y \leq k) - P(y < k + 1) \\
              &= \sigma(\theta_k - \eta) - \sigma(\theta_{k+1} - \eta)


where:

.. math::

    \eta = \beta_0 + 
        \text{age} \cdot \beta_{\text{age}} + 
        \text{sex} \cdot \beta_{\text{sex}} +
        \text{age} \cdot \text{sex} \cdot \beta_{\text{sexage}} + 
        \text{age}^2 \cdot \text{sex} \cdot \beta_{\text{sexage}^2} + 
        \text{age}^2 \cdot \beta_{\text{age}^2}


Step 4: Compute the number of asthma exacerbations in the current year
-----------------------------------------------------------------------

To compute the number of asthma exacerbations in the current year, we use the
:ref:`exacerbation-model`, which takes into account the agent's asthma control level, age, and sex,
as well as the current year. The probability of having :math:`n` exacerbations is given by a
Poisson distribution:

.. math::

    P(n = k) = \dfrac{\lambda^{k}e^{-\lambda}}{k!}

We determine the expected number of exacerbations, :math:`\lambda`, using the following formula:

.. math::

    \lambda &= e^{\mu} \\
    \mu &= \beta_0 +
        \beta_{\text{age}} \cdot \text{age} +
        \beta_{\text{sex}} \cdot \text{sex} \\
        &+\beta_{uc} \cdot P(\text{uncontrolled}) +
        \beta_{pc} \cdot P(\text{partially controlled}) +
        \beta_{c} \cdot P(\text{fully controlled}) \\
        &+\log(\alpha(\text{year}, \text{sex}, \text{age}))

If the number of exacerbations is ``0``, skip to Step 8. Otherwise, go to Step 5.


Step 5: Compute the severity of the asthma exacerbations in the current year
-----------------------------------------------------------------------------

There are four levels of severity for asthma exacerbations:

.. math::

    k &= 1: \text{mild} \\
    k &= 2: \text{moderate} \\
    k &= 3: \text{severe} \\
    k &= 4: \text{very severe / hospitalization}

If the agent has been previously hospitalized due to asthma:

.. math::

    P(k = 4 \mid t) &= \begin{cases}
        P(k = 4 \mid t - 1) \cdot \beta_{\text{prevhospped}} & \text{if age} < 14 \\
        P(k = 4 \mid t - 1) \cdot \beta_{\text{prevhospadult}} & \text{if age} \geq 14
    \end{cases} \\
    P(k = 1 \mid t) &= P(k = 1 \mid t - 1) \cdot (1 - P(k = 4 \mid t)) \\
    P(k = 2 \mid t) &= P(k = 2 \mid t - 1) \cdot (1 - P(k = 4 \mid t)) \\
    P(k = 3 \mid t) &= P(k = 3 \mid t - 1) \cdot (1 - P(k = 4 \mid t))


Otherwise, we use the initial probabilities of each severity level. Then, to determine the
number of exacerbations of each severity level, we use a multinomial distribution:

.. math::

    p(x_1, x_2, x_3, x_4; n, p) = 
        \dfrac{n!}{x_1! x_2! x_3! x_4!} \prod_{i=1}^4 P(k = i)^{x_i}, \quad \text{where} \quad \sum_{i=1}^4 x_i = n


Step 6: Update the number of hospitalizations
-----------------------------------------------------------------------------

We add the total previous hospitalizations to the current year's number of exacerbations at
severity level 4:

.. math::

    \text{hospitalizations} = \text{previous hospitalizations} + x_4

where :math:`x_4` is the number of exacerbations at severity level 4.
Continue to Step 8.

Step 7: Reassess asthma diagnosis
---------------------------------------------------

If the agents has been diagnosed with asthma, we want to check if they were misdiagnosed, or if
they lose their diagnosis. For this we use a Bernoulli distribution:

.. math::

    p &:= \text{probability of a person with asthma keeping their diagnosis} \\
    P(\text{agent keeps diagnosis}) &= p \\
    P(\text{agent loses diagnosis}) &= q = 1 - p

Step 8: Compute utility
---------------------------

Utility is a measure of the agent's quality of life, which can be affected by asthma control,
exacerbations, and hospitalizations. We compute the utility for the agent in the current year
using the :ref:`utility-model`:

.. math::

    u := u_{\text{baseline}} - A \cdot \left(
      \sum_{S=1}^{4} d_E(S) \cdot n_E(S) + \sum_{L=1}^{3} d_C(L) \cdot C(L)
    \right)

where:

* :math:`u_{\text{baseline}}` is the baseline utility for a person of a given age and sex
  (without asthma)
* :math:`d_{E}(S)` is the disutility due to an asthma exacerbation of severity level :math:`S`
* :math:`n_E(S)` is the number of asthma exacerbations of severity level :math:`S` in a year
* :math:`S \in \{1, 2, 3, 4\}` is the asthma exacerbation severity level (1 = mild, 2 =
  moderate, 3 = severe, 4 = very severe)
* :math:`d_{C}` is the disutility due to having asthma at control level :math:`L`
* :math:`C(L)` is the proportion of the year spent at asthma control level :math:`L`
* :math:`L \in \{1, 2, 3\}` is the asthma control level (1 = well-controlled, 2 =
  partially-controlled, 3 = uncontrolled)
* :math:`A` is a boolean indicating whether the person has asthma


Step 9: Compute cost
---------------------------

Next, we compute the cost due to asthma for the agent in Canadian dollars using the
:ref:`cost-model`:

.. math::

  \text{cost} = \sum_{S=1}^4 n_E(S) \cdot \text{cost}_E(S) + 
    \sum_{L=1}^3 P(L) \cdot \text{cost}_C(L)


where:

* :math:`n_E(S)` is the number of exacerbations at severity level :math:`S`
* :math:`\text{cost}_E(S)` is the cost of an exacerbation at severity level :math:`S`
* :math:`P(L)` is the probability of being at asthma control level :math:`L`
* :math:`\text{cost}_C(L)` is the cost of being at asthma control level :math:`L`

Step 10: Check if agent dies
-------------------------------

We use the :ref:`mortality-model` to determine if the agent dies in the current year of their lifetime.
If the agent dies, we finish simulating that agent and go on to the next agent. If the agent does
not die, we increment their age by 1 and go back to :ref:`step-1-lifetime-iteration` to repeat the
process for the next year of their life.

To compute the probability of death, we use the following formula:

.. math::

  \sigma^{-1}(q_x(\text{sex}, \text{age})) = 
      \sigma^{-1}(q_{x_0}(\text{sex}, \text{age})) -
      \beta_{\text{sex}}(\text{year} - \text{year}_0)

where:

* :math:`q_x` is the probability of death between age :math:`x` and :math:`x + 1`
* :math:`\sigma^{-1}` is the logit function
* :math:`q_{x_0}` is the baseline probability of death between age :math:`x_0` and :math:`x_0 + 1`,
  where :math:`x_0` is the age in the base year :math:`\text{year}_0`
* :math:`\text{year}` is the current year in the simulation
* :math:`\text{year}_0` is the starting year in the simulation

Finally, we determine if the agent dies using a Bernoulli distribution:

.. math::

  \text{is_dead} \sim \text{Bernoulli}(q_x(\text{sex}, \text{age}))

Step 11: Check if agent emigrates
-----------------------------------

We use the :ref:`migration-model` to determine if the agent emigrates to another province or country
in the current year of their lifetime. If the agent emigrates, we finish simulating that agent and
go on to the next agent. If the agent does not emigrate, we increment their age by 1 and go back to
:ref:`step-1-lifetime-iteration` to repeat the process for the next year of their life.

.. math::

  \text{emigrates} \sim \text{Bernoulli}(p_{\text{emigrate}}(\text{sex}, \text{age}, \text{year}))

