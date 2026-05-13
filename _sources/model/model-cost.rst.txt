.. _cost-model:

=====================
Asthma Cost Model
=====================

Data
====

.. _cost-data-exacerbations:

Cost Due to Asthma Exacerbations
**************************************

The data used in the cost model comes from the study :cite:`yaghoubi`, in
`Table I <https://www.sciencedirect.com/science/article/pii/S0091674919316343#tbl1>`_:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Variable
     - Value
   * - Direct costs of exacerbation by severity: mild
     - $130 USD
   * - Direct costs of exacerbation by severity: moderate
     - $594 USD
   * - Direct costs of exacerbation by severity: (very) severe
     - $9900 USD

.. note::

    In the paper, ``severe`` exacerbations are equivalent to our definition of ``very severe``
    exacerbations. Thus, we are missing the cost of ``severe`` exacerbations. To account for this,
    we defined the cost of a ``severe`` exacerbation as:

    .. math::

        \text{cost}_E(\text{severe}) &= \exp\left(\dfrac{
            \text{log}(\text{cost}_E(\text{moderate})) +
            \text{log}(\text{cost}_E(\text{very severe}))
        }{2}\right) \\
        &= \$2425 \text{ USD}


.. _cost-data-control:

Cost Due to Asthma Control Levels
**********************************

The data used for the cost by asthma control level also comes from the study :cite:`yaghoubi`, in
`Table I <https://www.sciencedirect.com/science/article/pii/S0091674919316343#tbl1>`_:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Variable
     - Value
   * - Annual direct costs of well-controlled asthma
     - $2372 USD
   * - Annual direct costs of not-well-controlled asthma
     - $2965 USD
   * - Annual direct costs of uncontrolled asthma
     - $3127 USD


Model
======

.. math::

  \text{cost} = \sum_{S=1}^4 n_E(S) \cdot \text{cost}_E(S) + 
    \sum_{L=1}^3 P(L) \cdot \text{cost}_C(L)


where:

* :math:`n_E(S)` is the number of exacerbations at severity level :math:`S`
* :math:`\text{cost}_E(S)` is the cost of an exacerbation at severity level :math:`S`
* :math:`P(L)` is the probability of being at asthma control level :math:`L`
* :math:`\text{cost}_C(L)` is the cost of being at asthma control level :math:`L`