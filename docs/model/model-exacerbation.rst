.. _exacerbation-model:

===========================
Asthma Exacerbations Model
===========================

Data
====

Population Data
*****************

We use the Statistics Canada population data that was generated and saved as:
`processed_data/birth/initial_pop_distribution_prop.csv 
<https://github.com/resplab/leap/blob/main/leap/processed_data/birth/initial_pop_distribution_prop.csv>`_.


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


Occurrence Data
******************

We use the occurrence data that was generated and saved as:
`processed_data/asthma_occurrence_predictions.csv 
<https://github.com/resplab/leap/blob/main/leap/data_generation/processed_data/asthma_occurrence_predictions.csv>`_

See :ref:`occurrence-model-1` for more details about this dataset.


.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - ``year``
     - :code:`int`
     - the calendar year
   * - ``sex``
     - :code:`str`
     - ``F`` = female, ``M`` = male
   * - ``age``
     - :code:`int`
     - the age of the patient in years
   * - ``incidence``
     - :code:`float`
     - the predicted asthma incidence for the given year, age, and sex
   * - ``prevalence``
     - :code:`float`
     - the predicted asthma prevalence for the given year, age, and sex


Hospitalization Data
***********************

The data is from the ``Hospital Morbidity Database (HMDB)`` from the
`Canadian Institute for Health Information (CIHI) 
<https://www.cihi.ca/en/hospital-morbidity-database-hmdb-metadata>`_.

The hospitalization data was collected from patients presenting to a hospital in Canada
due to an asthma exacerbation. We will use this data to calibrate the exacerbation model.

The hospitalization rate in this table is the hospitalization rate per 100 000 people.
For example, in the category ``F_90+``, the value would be the number of people hospitalized who
are female and over 90 during the given year. This can be calculated:

.. math::
        
    \text{rate} = \dfrac{\text{count}}{N} \times 100000


.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - ``fiscalYear``
     - :code:`int`
     - the year the data was collected
   * - ``N``
     - :code:`float`
     - the hospitalization rate for all ages and sexes in that year.
   * - ``M``
     - :code:`float`
     - the rate for all ages who are male in that year.
   * - ``F``
     - :code:`float`
     - the rate for all ages who are female in that year.
   * - ``0``
     - :code:`float`
     - the rate for all sexes aged 0 in that year.
   * - ``1``
     - :code:`float`
     - the rate for all sexes aged 1 in that year.
   * - ``...``
     - :code:`...`
     - ...
   * - ``90``
     - :code:`float`
     - the rate for all sexes aged 90 in that year.
   * - ``90+``
     - :code:`float`
     - the rate for all sexes aged over 90 in that year.
   * - ``F_0``
     - :code:`float`
     - the rate for all females aged 0 in that year.
   * - ``...``
     - :code:`...`
     - ...
   * - ``F_90+``
     - :code:`float`
     - the rate for all females aged over 90 in that year.
   * - ``M_0``
     - :code:`float`
     - the rate for all males aged 0 in that year.
   * - ``...``
     - :code:`...`
     - ...
   * - ``M_90+``
     - :code:`float`
     - the rate for all males aged over 90 in that year.



Model
======

The number of exacerbations in a given year is modelled using a Poisson distribution. The formula is:

.. math::

    N_{\text{exacerbations}} \sim \text{Poisson}(\lambda) = \dfrac{\lambda^k e^{-\lambda}}{k!}


Here :math:`\lambda` is the expected number of exacerbations per year. To obtain :math:`\lambda`,
we must perform a Poisson regression. The Poisson regression assumes that the value we are
interested in can be approximated using the following formula:

.. math::

    \ln(\lambda) = \ln(\alpha) + \beta_0 + \beta_{a} a + \beta_{s} s + \sum_{i=1}^3 \beta_i c_i 


where:

.. list-table::
   :widths: 25 50
   :header-rows: 1

   * - Variable
     - Description
   * - :math:`\alpha`
     - the calibration multiplier that adjusts the model to match the hospitalization data
   * - :math:`\beta_0`
     - a constant randomly chosen from the normal distribution :math:`\mathcal{N}(0, 1)`
   * - :math:`a`
     - age
   * - :math:`\beta_a`
     - age constant
   * - :math:`s`
     - sex
   * - :math:`\beta_{s}`
     - sex constant
   * - :math:`c_i`
     - relative time spent in control level :math:`i`
   * - :math:`\beta_i`
     - control level constant calculated from the :ref:`Control Model <control-model>`


Calibration
******************

We are interested in calculating :math:`\alpha`. If we rewrite the equation, the meaning of
:math:`\alpha` becomes more apparent:

.. math::

    \lambda = \alpha \cdot e^{\beta_0} e^{\beta_{a} a} e^{\beta_{s} s} \prod_{i=1}^3 e^{\beta_i c_i} 


How do we obtain :math:`\alpha`? We again assume that the mean value has the same form as in a
Poisson regression, with the following formula:

.. math::

    \ln(\lambda_{C}) = \sum_{i=1}^3 \gamma_i c_i 


* :math:`\lambda_C`: the average number of exacerbations in a given year
* :math:`c_i`: relative time spent in control level :math:`i`
* :math:`\gamma_i`: control level constant (different from :math:`\beta_i` above)

Here, the :math:`\gamma_i` values were calculated from the
`Economic Burden of Asthma (EBA) study <https://bmjopen.bmj.com/content/3/9/e003360.long>`_
and are given by:

.. math::

    \gamma_1 &:= 0.1880058 \quad \text{rate(exacerbation | fully controlled)} \\
    \gamma_2 &:= 0.3760116 \quad \text{rate(exacerbation | partially controlled)} \\
    \gamma_3 &:= 0.5640174 \quad \text{rate(exacerbation | uncontrolled)}


The number of exacerbations predicted by the model is then:

.. math::

    N_{\text{exac}}^{\text{(pred)}} &= \lambda_C \cdot N_{\text{asthma}} \\
    N_{\text{asthma}} &= N \cdot \eta_{\text{prev}}

* :math:`N_{\text{asthma}}`: the number of people in a given year, age, sex with asthma
* :math:`N`: the number of people in a given year, age, and sex
* :math:`\eta_{\text{prev}}`: the prevalence of asthma in a given year, age, and sex, from
  :ref:`occurrence-model-1`

and number of hospitalizations is:

.. math::

    N_{\text{hosp}}^{\text{(pred)}} = N_{\text{exac}}^{\text{(pred)}} \cdot P(\text{hosp})


* :math:`N_{\text{exac}}^{\text{(pred)}}`: the predicted number of exacerbations (of any severity)
  for a given year, age, and sex
* :math:`P(\text{hosp})`: the probability of hospitalization due to asthma given the patient has an
  asthma exacerbation

Finally, :math:`\alpha` can be computed:

.. math::

    \alpha(a, s, y) = \dfrac{N_{\text{hosp}}(a, s, y)}{N_{\text{hosp}}^{\text{(pred)}}(a, s, y)}
