===========================
Mortality Model
===========================

Data
====

To obtain the mortality data for each year, we used one table from ``Statistics Canada``:

Past Data: 1996 - 2021
*************************

For past years, we used
`Table 13-10-00837-01 from StatCan <https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1310083701>`_.

The ``*.csv`` file can be downloaded from here:
`13100837-eng.zip <https://www150.statcan.gc.ca/n1/tbl/csv/13100837-eng.zip>`_
and is saved as:
`LEAP/leap/original_data/13100837.csv
<https://github.com/resplab/leap/blob/main/leap/original_data/13100837.csv>`_.

The relevant columns are:

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - ``REF_DATE``
     - :code:`int`
     - the calendar year
   * - ``AGE_GROUP``
     - :code:`str`
     - the age of the person in years
   * - ``GEO``
     - :code:`str`
     - the province or terriroty full name
   * - ``SEX``
     - :code:`str`
     - one of "Both sexes", "Females", or "Males"
   * - ``ELEMENT``
     - :code:`str`
     - describes what the variable of interest is; we want ``"Death probability between age x and x+1 (qx)"``
   * - ``VALUE``
     - :code:`int`
     - the probability of death between age ``x`` and ``x+1`` in that year, province, sex, and age group


Projected Data: 2021 - 2068
****************************

``Statistics Canada`` doesn't provide annual projections for death probabilities, but does
provide a projection for specific years (which we call calibration years) for the ``M3``
projection scenario only. For Canada, this is ``2068``, and for BC, ``2043``.

This data can be found:
`Table 5.2.1 <https://www150.statcan.gc.ca/n1/pub/91-620-x/91-620-x2025001-eng.htm>`_.

Model
========

``Statistics Canada`` doesn't provide annual projections for death probabilities, but does
provide a projection for specific years (which we call calibration years) for the ``M3``
projection scenario only. For Canada, this is ``2068``, and for BC, ``2043``.
The following equation can be used to obtain the probability of death in future years:

.. math::

    \sigma^{-1}(p(\text{sex}, \text{age}, \text{year})) = 
        \sigma^{-1}(p(\text{sex}, \text{age}, \text{year}_0)) -
        e^{\beta_{\text{sex}}(\text{year} - \text{year}_0)}


where :math:`p(\text{sex}, \text{age}, \text{year}_0)` is the probability of death for a person of
that age/sex in the year the collected data ends (in our case, ``2020``), and
:math:`p(\text{sex}, \text{age}, \text{year})` is the probability of death for a person of that
age/sex in a future year.

The parameter :math:`\beta_{\text{sex}}` is unknown, and so we first need to calculate it.
To do so, we set :math:`\text{year} = \text{year}_C`, the calibration year, and use the ``Brent``
root-finding algorithm to optimize :math:`\beta_{\text{sex}}` such that the life expectancy in the
calibration year (which is known) matches the predicted life expectancy.

Once we have found :math:`\beta_{\text{sex}}`, we can use this formula to find the projected death
probabilities.