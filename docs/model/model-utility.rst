.. _utility-model:

=================
Utility Model
=================

In economics, the ``utility`` is a number that represents how satisfied a person is with a certain
situation. In the context of the ``LEAP`` model, it is used to quantify the health-related quality
of life of individuals with asthma. 

Datasets
=================

EQ5D
*******

The `EQ5D <https://euroqol.org/information-and-support/euroqol-instruments/eq-5d-5l/>`_ is a
standardized instrument used to measure health-related quality of life. It consists of
five dimensions: mobility, self-care, usual activities, pain/discomfort, and anxiety/depression.
The EQ5D is widely used in health economics and clinical trials to assess the impact of diseases
and treatments on quality of life.


We use the EQ5D values to calculate the baseline utility for an individual of a given age and sex.
For example, suppose we have a 30-year-old female with asthma. We can use the EQ5D values to
calculate her baseline utility (her utility if she didn't have asthma), and then we can compute
her net utility by subtracting the utility loss due to asthma.

The EQ5D data was obtained from
`Table 3 <https://link.springer.com/article/10.1007/s10198-023-01570-1/tables/3>`_ in the paper
`Canada population norms for the EQ-5D-5L <https://doi.org/10.1007/s10198-023-01570-1>`_
:cite:`eq5d2024`.

After processing the EQ5D data, our dataset is formatted as follows:

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
          The baseline utility for a person of a given age and sex.
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
            utility value. Standard deviation for <code class="notranslate">age &lt 18</code> is
            set to 0 since those EQ5D values were interpolated.
        </td>
      </tr>
    </tbody>
    </table>

.. _utility-data-exacerbations:

Disutility Due to Asthma Exacerbations
*****************************************

As in the :ref:`control-model`, we used the study :cite:`yaghoubi`, in
`Table I <https://www.sciencedirect.com/science/article/pii/S0091674919316343#tbl1>`_:

.. list-table::
   :header-rows: 1

   * - Variable
     - Value
   * - Utility of exacerbation by severity: mild
     - 0.57
   * - Utility of exacerbation by severity: moderate
     - 0.45
   * - Utility of exacerbation by severity: (very) severe
     - 0.33


.. note::

    In the paper, ``severe`` exacerbations are equivalent to our definition of ``very severe``
    exacerbations. Thus, we are missing the utility of ``severe`` exacerbations. To account for this,
    we defined the utility of a ``severe`` exacerbation as:

    .. math::

        \text{utility}(\text{severe}) &= \dfrac{
            \text{utility}(\text{moderate}) +
            \text{utility}(\text{very severe})
        }{2} \\
        &= \dfrac{0.45 + 0.33}{2} = 0.39


Now, these are utility values, but we want *disutility*. Since the study :cite:`yaghoubi` starts at
age 15, we set that as the baseline age. According to the EQ5D data, the baseline utility for a
15-year-old is ``0.9``. Thus, the disutility due to asthma exacerbations is given by:

.. math::

    d_E(S) = 0.9 - u_E(S)

where:

* :math:`d_E(S)` is the disutility due to an asthma exacerbation of severity level :math:`S`
* :math:`u_E(S)` is the utility due to an asthma exacerbation of severity level :math:`S`
* :math:`S \in \{1, 2, 3, 4\}` is the asthma exacerbation severity level (1 = mild, 2 =
  moderate, 3 = severe, 4 = very severe)


.. list-table::
   :header-rows: 1

   * - Exacerbation Severity
     - Utility
     - Disutility
   * - Mild
     - 0.57
     - 0.33
   * - Moderate
     - 0.45
     - 0.45
   * - Severe
     - 0.39
     - 0.51
   * - Very Severe
     - 0.33
     - 0.57

Now, the values listed in this table are the disutility for having an asthma exacerbation of a given
severity for an entire year. We assume that a mild asthma exacerbation lasts for 7 days, while
all the other severity levels last for 14 days. To convert these values we have the weekly
utility:

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
     - 0.33
     - 7 days
     - 0.00633
     - 0.00633
   * - Moderate
     - 0.45
     - 14 days
     - 0.00865
     - 0.01731
   * - Severe
     - 0.51
     - 14 days
     - 0.00981
     - 0.01962
   * - (Very) Severe
     - 0.57
     - 14 days
     - 0.01096
     - 0.02192


.. _utility-data-control:
Disutility Due to Asthma Control Levels
**********************************************

We used
`Table 3 <https://www.tandfonline.com/doi/pdf/10.3111/13696998.2015.1025793#page=6.08>`_
in the study :cite:`einarson` to obtain the utility values stratified by asthma control level:

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
We use the baseline utility for a 15-year-old of ``0.9``. Thus, the disutility stratified by
asthma control is given by:

.. math::

    d_C(L) = 0.9 - u_C(L)

where:

* :math:`d_C(L)` is the disutility due to asthma control level :math:`L`
* :math:`u_C(L)` is the utility due to asthma control level :math:`L`
* :math:`L \in \{1, 2, 3\}` is the asthma control level (1 = well-controlled, 2 =
  partially-controlled, 3 = uncontrolled)


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

The net utility is given by the formula:

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

