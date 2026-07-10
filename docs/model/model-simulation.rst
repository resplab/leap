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
     - The number of agents aged 0 in the initial timepoint of the simulation. This acts as a
       scaling factor: agents at all other ages are created in proportion to the Statistics
       Canada age distribution for the starting timepoint and province, so the total initial
       population is typically much larger than this value.
   * - ``max_age``
     - The maximum age of a person in the model.
   * - ``until_all_die``
     - Whether or not to keep the simulation running until all members of the population have
       died.
   * - ``min_timepoint``
     - The date and time to start the simulation.
   * - ``time_horizon``
     - How long to run the simulation for.
   * - ``time_delta``
     - The time delta to use for the simulation. For example, if ``time_delta`` is 1 year,
       then we will simulate the population at yearly time intervals. If ``time_delta`` is 1 month,
       then we will simulate the population at monthly time intervals.


Iterating Over Timepoints
==========================

Initial Timepoint
**********************

To start the simulation, we begin with the initial timepoint, ``min_timepoint``. At each timepoint,
a time interval is defined as:

.. math::

    \text{time interval} = [\text{timepoint}, \text{timepoint} + \text{time delta})


For example, if the ``min_timepoint`` is January 1, 2020, and the ``time_delta`` is 1 year, then the
first time interval would be ``[2020-01-01, 2021-01-01)``, i.e., the year ``2020``.

Unlike at subsequent timepoints, at the initial timepoint the simulation does not introduce only
newborns — instead, it creates a cross-sectional population spanning all ages from ``0`` to
``max_age``, representing the full age distribution of the province at the initial timepoint, drawn
from Statistics Canada population data (see the :ref:`birth-model`).

The ``num_births_initial`` parameter sets the number of agents at age 0. The number of agents at
every other age is determined by multiplying ``num_births_initial`` by that age group's ``prop``
value — the ratio of that age group's size to the newborn cohort in the Statistics Canada data.
As a result, the total number of agents created in the initial timepoint is generally much larger
than ``num_births_initial``, but is determined by this value.

Subsequent Timepoints
**********************

For all subsequent timepoints, we create a list of agents who are either born during that time
interval or who immigrated to the province during that time interval. The number of births at
subsequent timepoints is determined by the :ref:`birth-model`, which factors in the population
growth type and the number of births during the previous time interval. The number of immigrants is
determined by the :ref:`migration-model`.

Example
*************

Let's suppose we have:

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - Parameter
     - Value
   * - starting timepoint
     - 2021-01-01 00:00:00
   * - time horizon
     - 10 years
   * - time delta
     - 1 year
   * - province
     - BC
   * - population growth type
     - HG (High Growth)
   * - number of births in initial time interval
     - 100
   * - maximum age
     - 90

In the first time interval, we would create agents at all ages from 0 to 90, with the count at each
age scaled from ``num_births_initial`` using Statistics Canada population proportions. For example,
if age 30 has a ``prop`` of 2.5, then ``100 × 2.5 = 250`` agents are created at age 30. The
total initial population is the sum of ``num_births_initial × prop`` across all age groups.
A simplified excerpt:

.. list-table:: Year: 2021 (initial population, all ages)
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
   * - ...
     - ...
     - ...
     - ...
   * - Agent 100
     - 0
     - F
     - no
   * - Agent 101
     - 1
     - M
     - no
   * - ...
     - ...
     - ...
     - ...
   * - Agent N
     - 90
     - F
     - no

In the second time interval, which is the year 2022, we would calculate the number of births using
the birth model:

.. math::

    n(2022) &= n(2021) * \dfrac{N(2022)}{N(2021)} \\
    &= 100 * \dfrac{49200}{42653} \\
    &\approx 115

where:

* :math:`n(2022)` is the number of births in 2022, in the simulation
* :math:`n(2021)` is the number of births in 2021, in the simulation
* :math:`N(2022)` is the total number of births in 2022, for the ``HG`` scenario, from Statistics Canada
* :math:`N(2021)` is the total number of births in 2021, for the ``HG`` scenario, from Statistics Canada

We would then create 115 new agents, all aged 0, and add them to the list of agents for that
time interval.
For the number of immigrants:

.. math::

    i(2022) &= n(2022) * \dfrac{I(2022)}{N(2022)} \\
    &= 115 * \dfrac{67190}{49200} \\
    &\approx 157

where:

* :math:`i(2022)` is the number of immigrants to Canada in 2022, in the simulation
* :math:`I(2022)` is the total number of immigrants to Canada in 2022, for the ``HG`` scenario,
  from our model

Now that we have the total number of immigrants for 2022, we want to distribute them across age
and sex. We do this using the data from the immigration model.

So we have a list of agents for 2022 that looks like this:

.. list-table:: Year: 2022
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
   * - Agent 115
     - 0
     - F
     - no
   * - Agent 116
     - 29
     - M
     - yes
   * - Agent 117
     - 54
     - F
     - yes
   * - ...
     - ...
     - ...
     - ...
   * - Agent 272
     - 45
     - M
     - yes

Since our time horizon is 10 years, we would continue this process for each year until we
reach 2031.

Iterating Over Agents
========================

Once we have our list of agents for a given timepoint, we then simulate the lifetime of each agent,
from birth or immigration until either death or the maximum age is reached.


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
   * - family history of asthma
     - Assigned via a Bernoulli draw using the population prevalence of parental asthma.
       1 = at least one parent has asthma, 0 = neither parent has asthma.
       See :ref:`family-history-model` for details on the assignment process, and
       :ref:`occurrence-model-2` for how this affects the asthma probability.
   * - antibiotic exposure in infancy
     - Assigned via a Negative Binomial draw using the fitted GLM parameters for the agent's
       sex and birth year. This gives the number of antibiotic courses taken during the first
       year of life and can be 0, 1, 2, or 3. See :ref:`antibiotic_exposure_model` for details on the assignment process,
       and :ref:`occurrence-model-2` for how this affects the asthma probability.
   * - has asthma?
     - If the agent is less than 3 years old, they do not have asthma. Otherwise, they are assigned
       an asthma status based on the asthma prevalence model (see Step 2 below).
   * - exacerbation history
     - Initial exacerbation history is set to:

       * number of exacerbations previous timepoint: 0
       * number of exacerbations current timepoint: 0
   * - total hospitalizations
     - The agent's cumulative count of very severe (hospitalized) exacerbations. For an agent with
       asthma, this is initialized via the mini-simulation described in
       :ref:`step-4-check-hospitalizations` below; otherwise it starts at 0. It is incremented
       each subsequent time interval — see Step 8 and Step 6.


Step 1: Check if the agent is over 3 years old
----------------------------------------------

If the agent is over 3 years old, go to Step 2. If not, skip to Step 3.

Step 2: Check if the agent has asthma
---------------------------------------

We use the asthma prevalence model (:ref:`occurrence-model-2`) to assign a probability of
asthma at the agent's current age, sex, and timepoint, adjusted for their individual risk factors:

.. math::

    \text{logit}(p_{\text{prev}}) = \text{logit}(\bar{p}_{\text{prev}}) + \log(\omega_{\text{fhx}}) + \log(\omega_{\text{abx}}) - \alpha

where :math:`\bar{p}_{\text{prev}}` is the Model 1 prevalence target for the agent's stratum,
:math:`\omega_{\text{fhx}}` and :math:`\omega_{\text{abx}}` are the odds ratios for the agent's
family history and antibiotic exposure, and :math:`\alpha` is the per-stratum calibration term
looked up from ``asthma_occurrence_correction.csv``. The agent is then assigned an asthma status
via a Bernoulli draw:

.. math::

    \text{has asthma} \sim \text{Bernoulli}(p_{\text{prev}})

Step 3: Determine the age of asthma diagnosis
---------------------------------------------

If the agent was determined to have asthma in Step 2, we need to determine the age at which
the agent was diagnosed with asthma.

If the agent is 3 years old, the age of asthma diagnosis is 3.

If the agent is older than 3, we use the asthma incidence model to determine the age of asthma
diagnosis.

.. _step-4-check-hospitalizations:

Step 4: Check hospitalizations
--------------------------------

Next, we determine whether the agent has previously had a very severe (i.e. hospitalized)
exacerbation at some point since their asthma diagnosis (Step 3). This is used in the :ref:`exacerbation_severity_model` to increase the probability of a very severe exacerbation
for agents with a prior hospitalization.

Since this agent's asthma history was not directly simulated cycle-by-cycle (it was assigned all
at once in Steps 2 and 3), we cannot just look up whether a hospitalization occurred. Instead, we assume the agent's asthma was **not reversible** between their diagnosis age and
two years before their current age: that is, we treat the agent as having had asthma
continuously over that span. Under this assumption, we run a mini-simulation that sums the
expected exacerbation rate :math:`\lambda^{(i)}` for this agent :math:`i` (using the Step 6
formula below, with that year's age, sex, and control levels) over every year from the asthma
diagnosis age up to two years before the agent's current age:

.. math::

    \text{total rate}^{(i)} = \sum_{y} \lambda_y^{(i)}

where :math:`y` ranges over each year (not the simulation's own timepoint, but a calendar
year) from the asthma diagnosis age up to two years before the agent's current age, and
:math:`\lambda_y^{(i)}` is :math:`\lambda^{(i)}` evaluated at that year (Step 6 below).

We then compute the probability that *none* of the agent's exacerbations over this span were very
severe. Each agent's probability of a very severe exacerbation is
:math:`w^{\text{pre},(i)}_{\text{very severe}}` — the same pre-adjustment Dirichlet-drawn quantity
described in the :ref:`exacerbation_severity_model`. This
probability of zero very severe exacerbations out of ``total rate`` expected exacerbations is the
marginal (Dirichlet-multinomial) probability, expressed using the Gamma function :math:`\Gamma`
(the continuous extension of the factorial, :math:`\Gamma(n+1) = n!`):

.. math::

    P(\text{no very severe exacerbations}) = \dfrac{1}{\Gamma(\text{total rate}^{(i)} + 1)} \cdot
        \dfrac{\Gamma(\text{total rate}^{(i)} + 1 - w^{\text{pre},(i)}_{\text{very severe}})}
            {\Gamma(1 - w^{\text{pre},(i)}_{\text{very severe}})}

Finally, we toss a coin — a Bernoulli draw — to decide whether the agent had at least one very
severe exacerbation over this span:

.. math::

    \text{prev hosp} \sim \text{Bernoulli}\left(1 - P(\text{no very severe exacerbations})\right)


Step 5: Determine asthma control levels
-----------------------------------------

We next determine the asthma control levels for the agent. The asthma control levels give the
probability of the agent having fully-controlled, partially-controlled, or uncontrolled asthma
(:math:`k = 1, 2, 3` respectively). The probabilities are given by the :ref:`control-model`:

.. math::

    \small
    \text{logit}(P(y^{(i)} \leq k)) = \theta_k
        + \beta_{\text{age}} \cdot a^{(i)}
        + \beta_{\text{sex}} \cdot s^{(i)}
        + \beta_{\text{age}^2} \cdot {a^{(i)}}^2
        + \beta_{\text{age,sex}} \cdot a^{(i)} \cdot s^{(i)}
        + \beta_{\text{age}^2\text{,sex}} \cdot {a^{(i)}}^2 \cdot s^{(i)}
        + \beta_0^{(i)}

.. math::

    P(y^{(i)} = k) = P(y^{(i)} \leq k) - P(y^{(i)} \leq k-1)


Step 6: Compute the number of asthma exacerbations in the current timepoint
----------------------------------------------------------------------------

To compute the number of asthma exacerbations in the current timepoint, we use the exacerbation
model, which draws a count from a Poisson distribution parameterized by the agent's control
levels, patient-specific random effect, and the calibration multiplier :math:`\alpha` for their
age, sex, province, and timepoint. See :ref:`exacerbation-model` for the full derivation.

If the number of exacerbations is ``0``, skip to the end. Otherwise, go to Step 7.


Step 7: Compute the severity of the asthma exacerbations in the current timepoint
----------------------------------------------------------------------------------

Each of the :math:`n` exacerbations from Step 6 is assigned a severity level (mild, moderate,
severe, or very severe) via the Dirichlet-Multinomial model described in
:ref:`exacerbation_severity_model`, including the adjustment applied if the agent has a history
of very severe exacerbations (hospitalization).


Step 8: Update the number of hospitalizations
-----------------------------------------------------------------------------

We add the current timepoint's number of very severe exacerbations to ``total_hosp``, the
agent's cumulative hospitalization count first set during :ref:`step-4-check-hospitalizations`
of Agent Initialization above:

.. math::

    \text{total hosp}^{(i)} = \text{total hosp}^{(i)} + n_{\text{very severe}}^{(i)}


Iterating Over Lifetime
************************

For each agent, we simulate their lifetime by iterating over the time intervals of their life. 

We start with the timepoint when they were born or immigrated, and we continue until they reach the
maximum age or until they die. At each timepoint, we update their age and check if they are still
alive based on the mortality model. If they are still alive, we update their health status, asthma control,
and other health-related variables based on the model's parameters and the agent's characteristics.
During a given time interval, the following events can occur:

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

We use the asthma incidence model (:ref:`occurrence-model-2`) to compute the probability that
the agent receives a new asthma diagnosis in the current time interval, adjusted for their
individual risk factors:

.. math::

    \text{logit}(p_{\text{inc}}) = \text{logit}(\bar{p}_{\text{inc}}) + \log(\omega_{\text{fhx}}) + \log(\omega_{\text{abx}}) - \alpha

where :math:`\bar{p}_{\text{inc}}` is the Model 1 incidence target for the agent's stratum,
:math:`\omega_{\text{fhx}}` and :math:`\omega_{\text{abx}}` are the odds ratios for the agent's
family history and antibiotic exposure (both age-dependent and zero once the agent ages out of
the relevant windows), and :math:`\alpha` is the per-stratum calibration term looked up from
``asthma_occurrence_correction.csv``.

.. math::

    \text{new diagnosis} \sim \text{Bernoulli}(p_{\text{inc}})

If they do not receive a diagnosis, skip to Step 4. If they do, we set the age of diagnosis
to the current age of the agent and go to Step 3.

Step 3: Compute the control levels
---------------------------------------------------

We next determine the asthma control levels for the agent using the :ref:`control-model`.
The asthma control levels give the probability of the agent having fully-controlled,
partially-controlled, or uncontrolled asthma
(:math:`k = 1, 2, 3` respectively). The probabilities are given by the :ref:`control-model`:

.. math::

    \small
    \text{logit}(P(y^{(i)} \leq k)) = \theta_k
        + \beta_{\text{age}} \cdot a^{(i)}
        + \beta_{\text{sex}} \cdot s^{(i)}
        + \beta_{\text{age}^2} \cdot {a^{(i)}}^2
        + \beta_{\text{age,sex}} \cdot a^{(i)} \cdot s^{(i)}
        + \beta_{\text{age}^2\text{,sex}} \cdot {a^{(i)}}^2 \cdot s^{(i)}
        + \beta_0^{(i)}

.. math::

    P(y^{(i)} = k) = P(y^{(i)} \leq k) - P(y^{(i)} \leq k-1)


Step 4: Compute the number of asthma exacerbations in the current timepoint
----------------------------------------------------------------------------

To compute the number of asthma exacerbations in the current timepoint, we use the
:ref:`exacerbation-model`, which draws a count from a Poisson distribution parameterized by the
agent's control levels, patient-specific random effect, and the calibration multiplier
:math:`\alpha` for their age, sex, province, and timepoint.

If the number of exacerbations is ``0``, skip to Step 8. Otherwise, go to Step 5.


Step 5: Compute the severity of the asthma exacerbations in the current timepoint
----------------------------------------------------------------------------------

Each of the :math:`n` exacerbations from Step 4 is assigned a severity level (mild, moderate,
severe, or very severe) via the Dirichlet-Multinomial model described in
:ref:`exacerbation_severity_model`, including the adjustment applied if the agent has a history
of very severe exacerbations (hospitalization).


Step 6: Update the number of hospitalizations
-----------------------------------------------------------------------------

We add the current timepoint's number of very severe exacerbations to ``total_hosp``, the
agent's cumulative hospitalization count first set during :ref:`step-4-check-hospitalizations`
of Agent Initialization above:

.. math::

    \text{total hosp}^{(i)} = \text{total hosp}^{(i)} + n_{\text{very severe}}^{(i)}

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

We compute the health state utility for the agent in the current timepoint
using the :ref:`utility-model`:

.. math::

    \small
    u^{(i)} =
    \begin{cases}
        u_{\text{age}, \text{sex}}^{(i)} & \text{if agent } i \text{ does not have asthma} \\[6pt]
        \max\left(0,\ u_{\text{age}, \text{sex}}^{(i)} - \left(
          \sum_{S=1}^{4} n_{\text{Exac}}^{(i)}(S) \cdot d_E(S) + \sum_{k=1}^{3} P(y^{(i)} = k) \cdot d_C(k)
        \right)\right) & \text{if agent } i \text{ has asthma}
    \end{cases}

where:

* :math:`u_{\text{age}, \text{sex}}^{(i)}` is the baseline utility for agent :math:`i`, of the
  given age and sex (without asthma)
* :math:`n_{\text{Exac}}^{(i)}(S)` is the number of exacerbations at severity level :math:`S` in a
  time interval, for agent :math:`i`
* :math:`d_E(S)` is the disutility due to an asthma exacerbation of severity level :math:`S`
* :math:`S \in \{\text{mild}, \text{moderate}, \text{severe}, \text{very severe}\}` is the asthma
  exacerbation severity level
* :math:`P(y^{(i)} = k)` is the probability of agent :math:`i` being at asthma control level
  :math:`k`
* :math:`d_C(k)` is the disutility due to being at asthma control level :math:`k`
* :math:`k \in \{\text{well-controlled}, \text{partially-controlled}, \text{uncontrolled}\}` is
  the asthma control level


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

We use the :ref:`mortality-model` to determine if the agent dies in the current time interval of
their lifetime. If the agent dies, we finish simulating that agent and go on to the next agent.
If the agent does not die, we increment their age by 1 and go back to :ref:`step-1-lifetime-iteration`
to repeat the process for the next time interval of their life.

To compute the probability of death, we use the following formula:

.. math::

  \sigma^{-1}(q_x(\text{sex}, \text{age})) = 
      \sigma^{-1}(q_{x_0}(\text{sex}, \text{age})) -
      \beta_{\text{sex}}(\text{timepoint} - \text{timepoint}_0)

where:

* :math:`q_x` is the probability of death between age :math:`x` and :math:`x + 1`
* :math:`\sigma^{-1}` is the logit function
* :math:`q_{x_0}` is the baseline probability of death between age :math:`x_0` and :math:`x_0 + 1`,
  where :math:`x_0` is the age at the base timepoint :math:`\text{timepoint}_0`
* :math:`\text{timepoint}` is the current timepoint in the simulation
* :math:`\text{timepoint}_0` is the starting timepoint in the simulation

Finally, we determine if the agent dies using a Bernoulli distribution:

.. math::

  \text{is dead} \sim \text{Bernoulli}(q_x(\text{sex}, \text{age}))

Step 11: Check if agent emigrates
-----------------------------------

We use the :ref:`migration-model` to determine if the agent emigrates to another province or country
in the current time interval of their lifetime. If the agent emigrates, we finish simulating that agent and
go on to the next agent. If the agent does not emigrate, we increment their age by 1 and go back to
:ref:`step-1-lifetime-iteration` to repeat the process for the next time interval of their life.

.. math::

  \text{emigrates} \sim \text{Bernoulli}(p_{\text{emigrate}}(\text{sex}, \text{age}, \text{timepoint}))

