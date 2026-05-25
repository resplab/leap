.. _contingency-tables:

==================
Contingency Tables
==================

A contingency table (also called a cross-tabulation or crosstab) displays the joint frequency
distribution of two categorical variables. They are commonly used in statistics to examine
the relationship between two variables, and to calculate odds ratios and other measures of
association.

Two-by-Two Contingency Table
==============================

The simplest form is a 2×2 table, which has two rows and two columns. Consider two binary
variables — an exposure and an outcome — for a population of :math:`n` individuals:

.. raw:: html

  <table class="table">
      <thead>
      <tr>
          <th></th>
          <th>outcome +</th>
          <th>outcome -</th>
          <th>total</th>
      </tr>
      </thead>
      <tbody>
      <tr>
          <td>exposure +</td>
          <td><code class="notranslate">a</code></td>
          <td><code class="notranslate">b</code></td>
          <td><code class="notranslate">n₁</code></td>
      </tr>
      <tr>
          <td>exposure -</td>
          <td><code class="notranslate">c</code></td>
          <td><code class="notranslate">d</code></td>
          <td><code class="notranslate">n₀</code></td>
      </tr>
      <tr>
          <td>total</td>
          <td><code class="notranslate">n₂</code></td>
          <td></td>
          <td><code class="notranslate">n</code></td>
      </tr>
      </tbody>
  </table>

where the marginal totals are:

.. math::

    n_1 &= a + b \quad \text{(total with exposure)} \\
    n_0 &= c + d \quad \text{(total without exposure)} \\
    n_2 &= a + c \quad \text{(total with outcome)} \\
    n   &= a + b + c + d \quad \text{(total population)}

The **odds ratio** measures the strength of association between the exposure and the outcome:

.. math::

    \omega = \frac{a \cdot d}{b \cdot c}

An odds ratio of 1 indicates no association; values greater than 1 indicate that exposure is
associated with increased odds of the outcome.

Example: Smoking and Lung Cancer
----------------------------------

Suppose we observe ``n = 300`` patients and record whether they smoke and whether they have
lung cancer:

.. raw:: html

    <table class="table">
        <thead>
        <tr>
            <th></th>
            <th>lung cancer</th>
            <th>no lung cancer</th>
            <th>total</th>
        </tr>
        </thead>
        <tbody>
        <tr>
            <td>smoker</td>
            <td><code class="notranslate">a = 100</code></td>
            <td><code class="notranslate">b = 50</code></td>
            <td><code class="notranslate">n₁ = 150</code></td>
        </tr>
        <tr>
            <td>non-smoker</td>
            <td><code class="notranslate">c = 25</code></td>
            <td><code class="notranslate">d = 125</code></td>
            <td><code class="notranslate">n₀ = 150</code></td>
        </tr>
        <tr>
            <td>total</td>
            <td><code class="notranslate">n₂ = 125</code></td>
            <td></td>
            <td><code class="notranslate">n = 300</code></td>
        </tr>
        </tbody>
    </table>

The odds ratio for this table is:

.. math::

    \omega = \frac{100 \times 125}{50 \times 25} = 10

This means that smokers have 10 times the odds of developing lung cancer compared to
non-smokers in this population.

Usage in LEAP
===============

Contingency tables are used in LEAP to optimize the age-dependent beta parameters for the
incidence equation in :ref:`occurrence-model-2`. See the
:ref:`optimizing-beta-parameters` section of the Asthma Occurrence Model for details.
