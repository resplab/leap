.. _cost-model:

=====================
Cost Model
=====================

Data
====

.. _cost-data-exacerbations:

Cost Due to Asthma Exacerbations
**************************************

The data used in the cost model comes from the study :cite:`yaghoubi`, in
`Table I <https://www.sciencedirect.com/science/article/pii/S0091674919316343#tbl1>`_.

.. list-table::
   :widths: 70 30
   :header-rows: 1

   * - Variable
     - Value
   * - Direct costs of mild exacerbation
     - $130 USD
   * - Direct costs of moderate exacerbation
     - $594 USD
   * - Direct costs of severe exacerbation
     - $2425 USD (see :ref:`note <cost-severe-exacerbation-note>` below)
   * - Direct costs of very severe exacerbation
     - $9900 USD

.. _cost-severe-exacerbation-note:

.. note::

    In :cite:`yaghoubi`, ``severe`` exacerbations are equivalent to our definition of ``very severe``
    exacerbations. Thus, we are missing the cost of ``severe`` exacerbations. To account for this,
    we defined the cost of a ``severe`` exacerbation as the geometric mean of the costs of
    ``moderate`` and ``very severe`` exacerbations, i.e. the average of the two costs on the log-scale:

    .. math::

        \text{cost}_E(\text{severe}) &= \exp\left(\dfrac{
            \text{log}(\text{cost}_E(\text{moderate})) +
            \text{log}(\text{cost}_E(\text{very severe}))
        }{2}\right) \\
        &= \$2425 \text{ USD}


.. _cost-data-control:

Cost Due to Asthma Control Levels
**********************************

The data used for background cost by asthma control level also comes from the same study :cite:`yaghoubi`, in
`Table I <https://www.sciencedirect.com/science/article/pii/S0091674919316343#tbl1>`_:

.. list-table::
   :widths: 75 25
   :header-rows: 1

   * - Variable
     - Value
   * - Annual direct costs of well-controlled asthma
     - $2372 USD
   * - Annual direct costs of not-well-controlled asthma
     - $2965 USD
   * - Annual direct costs of uncontrolled asthma
     - $3127 USD


.. _cost-data-currency-conversion:

Currency Conversion
**************************************

The costs reported in :cite:`yaghoubi` above, for both exacerbations and asthma control levels,
are in USD, reported in September 2018 price year. To match the target price year of our model, all costs
are converted to 2023 CAD in two steps:

#. **Inflation adjustment**: 2018 USD are converted to 2023 USD using the Consumer Price
   Index (CPI) Inflation Calculator from the `U.S. Bureau of Labor Statistics
   <https://www.bls.gov/data/inflation_calculator.htm>`_:

   .. math::

       1 \text{ USD (Sept 2018)} = 1.22 \text{ USD (Sept 2023)}

#. **Exchange rate conversion**: 2023 USD are converted to 2023 CAD using the
   `Bank of Canada's annual average exchange rate
   <https://www.bankofcanada.ca/rates/exchange/annual-average-exchange-rates/>`_:

   .. math::

       1 \text{ USD (2023)} = 1.36 \text{ CAD (2023)}

Combining these two factors gives the overall USD-to-CAD conversion rate used in the model:

.. math::

    \text{exchange_rate_usd_cad} = 1.22 \times 1.36 \approx 1.66

The conversion rate is applied to obtain the 2023 CAD
costs used in the model:

.. list-table:: Direct costs (2023 CAD)
   :widths: 70 30
   :header-rows: 1

   * - Variable
     - Value
   * - Mild exacerbation
     - $216 CAD
   * - Moderate exacerbation
     - $986 CAD
   * - Severe exacerbation
     - $4026 CAD
   * - Very severe exacerbation
     - $16434 CAD
   * - Well-controlled asthma (annual)
     - $3938 CAD
   * - Not-well-controlled asthma (annual)
     - $4922 CAD
   * - Uncontrolled asthma (annual)
     - $5191 CAD


Model
======

Exacerbation costs are applied per exacerbation event, i.e. an agent incurs this cost
each time they have an exacerbation of the corresponding severity. Control level costs are weighted by :math:`P(y^{(i)} = k)`, which per the :ref:`control-model` represents
the proportion of the time interval agent :math:`i` spends at control level :math:`k`:

.. math::

  \text{cost}^{(i)} = \sum_{S=1}^4 n_E^{(i)}(S) \cdot \text{cost}_E(S) +
    \sum_{k=1}^3 P(y^{(i)} = k) \cdot \text{cost}_C(k)


where:

* :math:`n_E^{(i)}(S)` is the number of exacerbations at severity level :math:`S`
* :math:`\text{cost}_E(S)` is the cost of an exacerbation at severity level :math:`S`
* :math:`P(y^{(i)} = k)` is the probability of agent :math:`i` being at asthma control level
  :math:`k`
* :math:`\text{cost}_C(k)` is the cost of being at asthma control level :math:`k`