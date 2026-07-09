.. _utility-model:

=================
Utility Model
=================

Datasets
=================

EQ5D
*******

The `EQ5D <https://euroqol.org/information-and-support/euroqol-instruments/eq-5d-5l/>`_ is a
standardized instrument used to measure health-related quality of life. It consists of
five dimensions: mobility, self-care, usual activities, pain/discomfort, and anxiety/depression. Each dimension has five levels of severity, ranging from no problems to extreme problems.

EQ-5D-5L data for the general population was obtained from
`Table 3 <https://link.springer.com/article/10.1007/s10198-023-01570-1/tables/3>`_ in the paper
`Canada population norms for the EQ-5D-5L <https://doi.org/10.1007/s10198-023-01570-1>`_
:cite:`eq5d2024`.

After processing the EQ5D data,
`leap/processed_data/eq5d_canada.csv <https://github.com/resplab/leap/blob/main/leap/processed_data/eq5d_canada.csv>`_,
is formatted as follows:

.. raw:: html

  <table class="table">
    <thead>
      <tr>
          <th>Column</th>
          <th>Type</th>
          <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code class="notranslate">age</code></td>
        <td>
          <code class="notranslate">int</code>
        </td>
        <td>
          A person's age in years, range <code class="notranslate">[0, 110]</code>
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">sex</code></td>
        <td>
          <code class="notranslate">str</code>
        </td>
        <td>
            F = Female, M = Male
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">eq5d</code></td>
        <td>
          <code class="notranslate">float</code>
        </td>
        <td>
          The health state utility for a person of a given age and sex.
          Range <code class="notranslate">[0, 1]</code>.
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">sd</code></td>
        <td>
          <code class="notranslate">float</code>
        </td>
        <td>
            The standard deviation of the utility value, used to account for uncertainty in the
            utility value.
        </td>
      </tr>
    </tbody>
    </table>

Since the EQ-5D-5L data described above only covers individuals aged 18 and older, the utility
values for ages 0 to 17 are interpolated. We assume that utility increases linearly from a value
of 1 (perfect health) at age 0 to the observed EQ-5D value at age 18, :math:`u_{18}`, separately
for each sex:

.. math::

    u_{\text{age}} = 1 - \dfrac{1 - u_{18}}{18} \times \text{age}, \quad \text{age} \in [0, 17]

Since these values are derived rather than directly observed, the standard deviation ``sd`` for
ages :math:`< 18` is set to ``0``, as noted in the table above.

.. _utility-data-exacerbations:

Disutility Due to Asthma Exacerbations
*****************************************

The health state utility values stratified by exacerbation severity come from
:cite:`yaghoubi`, in
`Table I <https://www.sciencedirect.com/science/article/pii/S0091674919316343#tbl1>`_, originally
sourced from :cite:`lloyd2007` and :cite:`campbell2010`:

.. list-table::
   :widths: 70 30
   :header-rows: 1

   * - Variable
     - Value
   * - Utility of mild exacerbation
     - 0.57
   * - Utility of moderate exacerbation
     - 0.45
   * - Utility of severe exacerbation
     - 0.39 (see :ref:`note <utility-severe-exacerbation-note>` below)
   * - Utility of very severe exacerbation
     - 0.33

.. _utility-severe-exacerbation-note:

.. note::

    In :cite:`yaghoubi`, ``severe`` exacerbations are equivalent to our definition of ``very severe``
    exacerbations. Thus, we are missing the utility of ``severe`` exacerbations. To account for
    this, we defined the utility of a ``severe`` exacerbation as the arithmetic mean of the
    utilities of ``moderate`` and ``very severe`` exacerbations:

    .. math::

        u_E(\text{severe}) &= \dfrac{
            u_E(\text{moderate}) + u_E(\text{very severe})
        }{2} \\
        &= \dfrac{0.45 + 0.33}{2} = 0.39

We need to convert these health state utilities to disutilities due to exacerbations. According to
:cite:`lloyd2007`, the mean EQ-5D utility for asthma patients with no exacerbation was ``0.89``; we
use this as the baseline. Thus, the disutility due to asthma exacerbations is given by:

.. math::

    d_E(S) = 0.89 - u_E(S)

where:

* :math:`d_E(S)` is the disutility due to an asthma exacerbation of severity level :math:`S`
* :math:`u_E(S)` is the utility due to an asthma exacerbation of severity level :math:`S`
* :math:`S \in \{\text{mild}, \text{moderate}, \text{severe}, \text{very severe}\}` is the asthma
  exacerbation severity level


.. list-table::
   :header-rows: 1

   * - Exacerbation Severity
     - Utility
     - Disutility
   * - Mild
     - 0.57
     - 0.32
   * - Moderate
     - 0.45
     - 0.44
   * - Severe
     - 0.39
     - 0.50
   * - Very Severe
     - 0.33
     - 0.56

Now, the values listed in this table are the disutility for having an asthma exacerbation of a given
severity for an entire year. We assume that a mild asthma exacerbation lasts for 7 days, while
all the other severity levels last for 14 days :cite:`aldington2007`. To convert these values we
have the weekly disutility:

.. math::

  d_{\text{weekly}} = d_{\text{annual}} \times \dfrac{1}{52}


.. list-table::
   :widths: 15 15 15 15 15
   :header-rows: 1

   * - Exacerbation Severity
     - Annual Disutility
     - Exacerbation Duration
     - Weekly Disutility
     - Disutility per Exacerbation
   * - Mild
     - 0.32
     - 7 days
     - 0.00615
     - 0.00615
   * - Moderate
     - 0.44
     - 14 days
     - 0.00846
     - 0.01692
   * - Severe
     - 0.50
     - 14 days
     - 0.00962
     - 0.01923
   * - (Very) Severe
     - 0.56
     - 14 days
     - 0.01077
     - 0.02154


.. _utility-data-control:
Disutility Due to Asthma Control Levels
**********************************************

The health state utility values stratified by asthma control level come from a discrete choice
experiment of 157 adult patients with asthma in Canada :cite:`mctaggartcowan2008`, as reported in
`Table 3 <https://www.tandfonline.com/doi/pdf/10.3111/13696998.2015.1025793#page=6.08>`_ of
:cite:`einarson`:

.. list-table::
   :header-rows: 1

   * - Control
     - Author
     - Instrument
     - Baseline (Mean)
     - SD
   * - Well controlled
     - McTaggart-Cowan et al.
     - EQ-5D
     - 0.840
     - 0.200
   * - Adequate
     - McTaggart-Cowan et al.
     - EQ-5D
     - 0.810
     - 0.220
   * - Not controlled
     - McTaggart-Cowan et al.
     - EQ-5D
     - 0.800
     - 0.210

As with the exacerbation severity, we want to convert these utility values to disutility values.
We use age 15 as the baseline age, consistent with the minimum age (15 years) of the target population in :cite:`yaghoubi`. According to the
`EQ5D data <https://github.com/resplab/leap/blob/main/leap/processed_data/eq5d_canada.csv>`_,
the baseline utility for a 15-year-old is ``0.9``. Thus, the disutility stratified by asthma
control is given by:

.. math::

    d_C(k) = 0.9 - u_C(k)

where:

* :math:`d_C(k)` is the disutility due to asthma control level :math:`k`
* :math:`u_C(k)` is the utility due to asthma control level :math:`k`
* :math:`k \in \{\text{well-controlled}, \text{partially-controlled}, \text{uncontrolled}\}` is
  the asthma control level


.. list-table::
   :header-rows: 1

   * - Asthma Control Level
     - Utility
     - Disutility
   * - Well-Controlled
     - 0.84
     - 0.06
   * - Partially-Controlled
     - 0.81
     - 0.09
   * - Uncontrolled
     - 0.80
     - 0.10


Model: Calculating Utility
===========================

Exacerbation disutility is applied per exacerbation event, i.e. it is incurred each time there is
an exacerbation of the corresponding severity. Control level disutility is weighted by
:math:`P(y = k)`, which per the :ref:`control-model` represents the proportion of the time interval
spent at control level :math:`k`. The net utility is given by the formula:

.. math::

    u =
    \begin{cases}
        u_{\text{age}, \text{sex}} & \text{if the person does not have asthma} \\[6pt]
        \max\left(0,\ u_{\text{age}, \text{sex}} - \left(
          \sum_{S=1}^{4} n_{\text{Exac}}(S) \cdot d_E(S) + \sum_{k=1}^{3} P(y = k) \cdot d_C(k)
        \right)\right) & \text{if the person has asthma}
    \end{cases}

where:

* :math:`u_{\text{age}, \text{sex}}` is the baseline utility for a person of the given age and
  sex (without asthma)
* :math:`n_{\text{Exac}}(S)` is the number of exacerbations at severity level :math:`S` in a
  time interval
* :math:`d_E(S)` is the disutility due to an asthma exacerbation of severity level :math:`S`
* :math:`S \in \{\text{mild}, \text{moderate}, \text{severe}, \text{very severe}\}` is the asthma
  exacerbation severity level
* :math:`P(y = k)` is the probability of being at asthma control level :math:`k`
* :math:`d_C(k)` is the disutility due to being at asthma control level :math:`k`
* :math:`k \in \{\text{well-controlled}, \text{partially-controlled}, \text{uncontrolled}\}` is
  the asthma control level

