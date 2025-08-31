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


Calculating Utility
=========================

The net utility is given by the formula:

.. math::

    u := u_{\text{baseline}} - A \cdot (d_{\text{exacerbation}} - d_{\text{control}})

where:

* :math:`u_{\text{baseline}}` is the baseline utility for a person of a given age and sex
  (without asthma)
* :math:`d_{\text{exacerbation}}` is the disutility due to asthma exacerbations
* :math:`d_{\text{control}}` is the disutility due to asthma control levels
* :math:`A` is a boolean indicating whether the person has asthma

