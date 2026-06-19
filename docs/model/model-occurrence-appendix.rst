.. _occurrence-model-appendix:

===========================================================
Technical Appendix: Asthma Occurrence Model Mathematics
===========================================================

This appendix contains the detailed mathematical derivations
for the calibration and optimisation steps described in the
:ref:`occurrence-model`.

Calibration Model Formula
--------------------------

Before we begin, let us define some terms. We have two risk factors we are interested in:
family history and antibiotic use. There are :math:`2 * 4 = 8` possible combinations of these two
risk factors:


.. raw:: html

  <table class="table">
    <thead>
      <tr>
          <th>&lambda;</th>
          <th>Family History</th>
          <th>Antibiotic Dose</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><code class="notranslate">0</code></td>
        <td><code class="notranslate">0</code></td>
        <td><code class="notranslate">0</code></td>
      </tr>
      <tr>
        <td><code class="notranslate">1</code></td>
        <td><code class="notranslate">1</code></td>
        <td><code class="notranslate">0</code></td>
      </tr>
      <tr>
        <td><code class="notranslate">2</code></td>
        <td><code class="notranslate">0</code></td>
        <td><code class="notranslate">1</code></td>
      </tr>
      <tr>
        <td><code class="notranslate">3</code></td>
        <td><code class="notranslate">1</code></td>
        <td><code class="notranslate">1</code></td>
      </tr>
      <tr>
        <td><code class="notranslate">4</code></td>
        <td><code class="notranslate">0</code></td>
        <td><code class="notranslate">2</code></td>
      </tr>
      <tr>
        <td><code class="notranslate">5</code></td>
        <td><code class="notranslate">1</code></td>
        <td><code class="notranslate">2</code></td>
      </tr>
      <tr>
        <td><code class="notranslate">6</code></td>
        <td><code class="notranslate">0</code></td>
        <td><code class="notranslate">3</code></td>
      </tr>
      <tr>
        <td><code class="notranslate">7</code></td>
        <td><code class="notranslate">1</code></td>
        <td><code class="notranslate">3</code></td>
      </tr>
    </tbody>
    </table>


We can represent each combination as a vector of the form:

.. math::

  \begin{bmatrix}
    f_{\lambda} \\
    d_{\lambda}
  \end{bmatrix}

where :math:`f_{\lambda}` is the family history and :math:`d_{\lambda}` is the antibiotic dose.

We next define the odds ratio for a given risk factor as:

.. math::

  \omega(r=k) = \dfrac{P(A = 1 \mid r = k)}{P(A = 1 \mid r = 0)}

where :math:`A` is the asthma incidence or prevalence and :math:`r` is the risk factor.
To combine odds ratios, we have:

.. math::

  \omega_{\lambda} &= \omega(f = f_{\lambda}, d = d_{\lambda}) \\
  &= \dfrac{P(A = 1 \mid f = f_{\lambda}, d = d_{\lambda})}{P(A = 1 \mid f = 0, d = 0)} \\
  &= \dfrac{P(A = 1 \mid f = f_{\lambda})}{P(A = 1 \mid f = 0)} \cdot
    \dfrac{P(A = 1 \mid d = d_{\lambda})}{P(A = 1 \mid d = 0)} \\
  &= \omega(f = f_{\lambda}) \cdot \omega(d = d_{\lambda})

Since these are multiplicative, the log of the odds ratios is additive:

.. math::

  \log(\omega_{\lambda}) = \log(\omega(f = f_{\lambda})) +
    \log(\omega(d = d_{\lambda}))

We can now define our formula for the calibration model:

.. math::

  \zeta_{\lambda}^{(i)} = \sigma\left(\beta_{\eta} + \log(\omega_{\lambda}^{(i)}) + \alpha\right)

where:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Variable
     - Description
   * - :math:`\beta_{\eta} = \sigma^{-1}(\eta^{(i)})`
     - determined by the output of the first model
   * - :math:`\eta^{(i)}`
     - the predicted incidence or prevalence from the first model
   * - :math:`\sigma(x)`
     - the logistic function
   * - :math:`\alpha = \sum_{\lambda=1}^{n} p(\lambda) \cdot \beta_{\lambda}`
     - the correction / calibration term for either the incidence or prevalence
   * - :math:`\zeta^{(i)} = \sum_{\lambda=0}^{n} p(\lambda) \zeta_{\lambda}^{(i)}`
     - predicted asthma prevalence / incidence for the model. We want this to be as close as
       possible to :math:`\eta^{(i)}`.
   * - :math:`\zeta_{\lambda}^{(i)}`
     - the predicted asthma incidence or prevalence from the model for the risk factor combination
       indexed by :math:`\lambda`
   * - :math:`p(\lambda)`
     - the probability of the risk factor combination indexed by :math:`\lambda`

Solving for the Correction Term: Prevalence
--------------------------------------------

.. math::

  \zeta_{\text{prev}} &= \sum_{\lambda=0}^{n} p(\lambda) \zeta_{\lambda} \\
  &= \sum_{\lambda=0}^{n} p(\lambda) \sigma(\beta_{\eta} + \log(\omega_{\lambda}) - \alpha) 


We want to find a correction term :math:`\alpha` such that the predicted asthma prevalence
:math:`\zeta` is as close as possible to the predicted asthma prevalence :math:`\eta`. To do this,
we use the ``Broyden-Fletcher-Goldfarb-Shanno (BFGS)`` algorithm to minimize the absolute
difference between :math:`\zeta` and :math:`\eta`.


Solving for the Correction Term: Incidence
--------------------------------------------

In our model, asthma incidence is defined as the number of new diagnoses between the previous
timepoint and the current timepoint, divided by the total population. To calibrate the incidence,
we first find the calibrated prevalence for the previous timepoint:

.. math::

  \zeta_{\text{prev}}(t-1) &= \sum_{\lambda=0}^{n} p(\lambda, t-1) \zeta_{\text{prev}, \lambda}(t-1) \\
  &= \sum_{\lambda=0}^{n} p(\lambda, t-1) \sigma(\beta_{\eta} + \log(\omega_{\lambda}) - \alpha)

Now, what we want to find is the joint probability of each risk factor combination,
:math:`p(\lambda, A = 0 \mid t-1)`, for the population without asthma.

.. math::

  P(\lambda, A = 0) = P(A = 0 \mid \lambda) \cdot P(\lambda)

Now, we must have:

.. math::

  P(A = 0 \mid \lambda) = 1 - P(A = 1 \mid \lambda) = 1 - \zeta_{\text{prev}, \lambda}(t-1)

So, we can rewrite the joint probability as:

.. math::

  p(\lambda, A = 0 \mid t-1) = (1 - \zeta_{\text{prev}, \lambda}(t-1)) \cdot p(\lambda, t-1)


Next, we find the calibrated asthma incidence for the current timepoint:

.. math::

  \zeta_{\text{inc}}(t) &= \sum_{\lambda=0}^{n} p(\lambda, A = 0 \mid t-1) \zeta_{\text{inc}, \lambda}(t) \\
  &= \sum_{\lambda=0}^{n} p(\lambda, A = 0 \mid t-1) \sigma(\beta_{\eta} + \log(\omega_{\lambda}) - \alpha)


where we recall that:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Variable
     - Description
   * - :math:`\beta_{\eta} = \sigma^{-1}(\eta^{(i)}(t))`
     - determined by the output of the first model
   * - :math:`\eta^{(i)}(t)`
     - defined above; the predicted incidence from the first model
   * - :math:`\alpha = \sum_{\lambda=1}^{n} p(\lambda, A = 0 \mid t-1) \cdot \beta_{\lambda}`
     - the correction / calibration term for the incidence
   * - :math:`\zeta^{(i)} = \sum_{\lambda=0}^{n} p(\lambda, A = 0 \mid t-1) \zeta_{\lambda}^{(i)}`
     - predicted asthma incidence for the model. We want this to be as close as
       possible to :math:`\eta^{(i)}`.
   * - :math:`\zeta_{\lambda}^{(i)}`
     - the predicted asthma incidence from the model for the risk factor combination
       indexed by :math:`\lambda`
   * - :math:`p(\lambda, A = 0 \mid t-1)`
     - the joint probability of the risk factor combination indexed by :math:`\lambda`, for a
       person who did not have asthma at time :math:`t-1`


We again want to find a correction term :math:`\alpha` such that the predicted asthma incidence
:math:`\zeta` is as close as possible to the asthma incidence from the first model, :math:`\eta`.
To do this, we use the ``Broyden-Fletcher-Goldfarb-Shanno (BFGS)`` algorithm to minimize the
absolute difference between :math:`\zeta` and :math:`\eta`.



Optimizing the Initial Beta Parameters for the Incidence Equation
------------------------------------------------------------------

For the incidence equation, we need to optimize two of the initial beta parameters:

* :math:`\beta_{\text{fhx}_\text{age}}`
* :math:`\beta_{\text{abx}_\text{age}}`


Past Contingency Table
^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

    <table class="table">
        <thead>
        <tr>
            <th></th>
            <th>variable 2, outcome +</th>
            <th>variable 2, outcome -</th>
            <th></th>
        </tr>
        </thead>
        <tbody>
        <tr>
            <td>variable 1, outcome +</td>
            <td><code class="notranslate">a0</code></td>
            <td><code class="notranslate">b0</code></td>
            <td><code class="notranslate">n1</code></td>
        </tr>
        <tr>
            <td>variable 1, outcome -</td>
            <td><code class="notranslate">c0</code></td>
            <td><code class="notranslate">d0</code></td>
            <td></td>
        </tr>
        <tr>
            <td></td>
            <td><code class="notranslate">n2</code></td>
            <td></td>
            <td><code class="notranslate">n</code></td>
        </tr>
        </tbody>
    </table>


We want to calculate :math:`a_0`, :math:`b_0`, :math:`c_0`, and :math:`d_0` using :math:`n_1`,
:math:`n_2`, :math:`n`, and :math:`\omega_{\lambda}`. Now, we have the probabilities of each of the
risk factor combinations, :math:`p(\lambda)`, but for the contingency table, we only want to
consider one risk factor combination at a time. To do this, we compute the conditional probability:

.. math::

    p(\Lambda = \lambda \mid \Lambda \in \{0, \lambda\}) = 
      \dfrac{p(\Lambda = \lambda)}{p(\Lambda = \lambda) + p(\Lambda = 0)}

To obtain :math:`n_1`, the number of people with risk factor combination :math:`\lambda` with or
without an asthma diagnosis, we multiply the conditional probability by the total population
:math:`n`:

.. math::
    n_1 = p(\Lambda = \lambda \mid \Lambda \in \{0, \lambda\}) \cdot n

To obtain :math:`n_2`, the number of people diagnosed with asthma with or without risk factor
combination :math:`\lambda`:

.. math::
    n_2 = (1 - p(\Lambda = \lambda \mid \Lambda \in \{0, \lambda\})) \cdot \zeta_{\text{prev}, 0}(t=0) \cdot n +
      p(\Lambda = \lambda \mid \Lambda \in \{0, \lambda\}) \cdot \zeta_{\text{prev}, \lambda}(t=0) \cdot n

From this, we can calculate the values for the contingency table:

.. math::

    b_0 &= n_1 - a_0 \\
    c_0 &= n_2 - a_0 \\
    d_0 &= n - n_1 - n_2 - a_0

To obtain :math:`a_0`, we follow the methods described in the paper :cite:`dipietrantonj2006`.
See :doc:`conv_2x2 <../dev/api/data_generation/leap.data_generation.utils>` for the Python
implementation of this method.

Current Contingency Table: Reassessment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

According to our model, an asthma diagnosis is not static; a patient may be diagnosed with asthma
and then later be reassessed as not having asthma. We would like to compute the updated contingency
table:

.. raw:: html

    <table class="table">
        <thead>
        <tr>
            <th></th>
            <th>asthma, outcome +</th>
            <th>asthma, outcome -</th>
            <th></th>
        </tr>
        </thead>
        <tbody>
        <tr>
            <td>risk factor <code class="notranslate">λ</code>, outcome +</td>
            <td><code class="notranslate">a1_ra</code></td>
            <td><code class="notranslate">b1_ra</code></td>
            <td><code class="notranslate">n1</code></td>
        </tr>
        <tr>
            <td>risk factor <code class="notranslate">λ</code>, outcome -</td>
            <td><code class="notranslate">c1_ra</code></td>
            <td><code class="notranslate">d1_ra</code></td>
            <td></td>
        </tr>
        <tr>
            <td></td>
            <td><code class="notranslate">n2</code></td>
            <td></td>
            <td><code class="notranslate">n</code></td>
        </tr>
        </tbody>
    </table>

To calculate the updated contingency table, we have:

.. math::
    a_{1, \text{ra}} &= a_0 \cdot \rho \\
    b_{1, \text{ra}} &= a_0 \cdot (1 - \rho) \\
    c_{1, \text{ra}} &= c_0 \cdot \rho \\
    d_{1, \text{ra}} &= c_0 \cdot (1 - \rho)

where:

.. raw:: html

    <table class="table">
        <thead>
        <tr>
            <th></th>
            <th>Risk Factors</th>
            <th>t=0</th>
            <th>t=1</th>
        </tr>
        </thead>
        <tbody>
        <tr>
            <td><code class="notranslate">a1_ra</code></td>
            <td><code class="notranslate">λ</code></td>
            <td>has asthma diagnosis</td>
            <td>reassessed: has asthma diagnosis</td>
        </tr>
        <tr>
            <td><code class="notranslate">b1_ra</code></td>
            <td><code class="notranslate">λ</code></td>
            <td>has asthma diagnosis</td>
            <td>reassessed: no asthma diagnosis</td>
        </tr>
        <tr>
            <td><code class="notranslate">c1_ra</code></td>
            <td>None</td>
            <td>has asthma diagnosis</td>
            <td>reassessed: has asthma diagnosis</td>
        </tr>
        <tr>
            <td><code class="notranslate">d1_ra</code></td>
            <td>None</td>
            <td>has asthma diagnosis</td>
            <td>reassessed: no asthma diagnosis</td>
        </tr>
        </tbody>
    </table>

* :math:`a_{1, \text{ra}}` is the proportion of the population with risk factor combination :math:`\lambda`
  who had an asthma diagnosis at :math:`t=0` and still have it at :math:`t=1`
* :math:`b_{1, \text{ra}}` is the proportion of the population with risk factor combination :math:`\lambda`
  who had an asthma diagnosis at :math:`t=0` but no longer have it at :math:`t=1`
* :math:`c_{1, \text{ra}}` is the proportion of the population with no risk factors (:math:`\lambda = 0`)
  who had an asthma diagnosis at :math:`t=0` and still have it at :math:`t=1`
* :math:`d_{1, \text{ra}}` is the proportion of the population with no risk factors (:math:`\lambda = 0`)
  who had an asthma diagnosis at :math:`t=0` but no longer have it at :math:`t=1`
* :math:`\rho` is the probability that a person would be reassessed as having an asthma diagnosis
  at :math:`t=1` given that they had an asthma diagnosis at :math:`t=0`


Current Contingency Table: New Diagnosis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the reassessment table, we considered only the patients who were diagnosed with asthma.
We will now consider those who were not diagnosed with asthma:

.. raw:: html

    <table class="table">
        <thead>
        <tr>
            <th></th>
            <th>asthma, outcome +</th>
            <th>asthma, outcome -</th>
            <th></th>
        </tr>
        </thead>
        <tbody>
        <tr>
            <td>risk factor <code class="notranslate">λ</code>, outcome +</td>
            <td><code class="notranslate">a1_dx</code></td>
            <td><code class="notranslate">b1_dx</code></td>
            <td><code class="notranslate">n1</code></td>
        </tr>
        <tr>
            <td>risk factor <code class="notranslate">λ</code>, outcome -</td>
            <td><code class="notranslate">c1_dx</code></td>
            <td><code class="notranslate">d1_dx</code></td>
            <td></td>
        </tr>
        <tr>
            <td></td>
            <td><code class="notranslate">n2</code></td>
            <td></td>
            <td><code class="notranslate">n</code></td>
        </tr>
        </tbody>
    </table>


To calculate the updated contingency table, we have:

.. math::
    a_{1, \text{dx}} &= b_0 \cdot \zeta_{\text{inc}, \lambda}(t=1) \\
    b_{1, \text{dx}} &= b_0 \cdot (1 - \zeta_{\text{inc}, \lambda}(t=1)) \\
    c_{1, \text{dx}} &= d_0 \cdot \zeta_{\text{inc}, 0}(t=1) \\
    d_{1, \text{dx}} &= d_0 \cdot (1 - \zeta_{\text{inc}, 0}(t=1))

where:

.. raw:: html

    <table class="table">
        <thead>
        <tr>
            <th></th>
            <th>Risk Factors</th>
            <th>t=0</th>
            <th colspan="3">t=1</th>
        </tr>
        <tr>
            <th></th>
            <th></th>
            <th></th>
            <th>incidence</th>
            <th>net</th>
        </thead>
        <tbody>
        <tr>
            <td><code class="notranslate">a1_dx</code></td>
            <td><code class="notranslate">λ</code></td>
            <td>no asthma diagnosis</td>
            <td>new asthma diagnosis</td>
            <td>asthma</td>
        </tr>
        <tr>
            <td><code class="notranslate">b1_dx</code></td>
            <td><code class="notranslate">λ</code></td>
            <td>no asthma diagnosis</td>
            <td>no new asthma diagnosis</td>
            <td>no asthma</td>
        </tr>
        <tr>
            <td><code class="notranslate">c1_dx</code></td>
            <td>None</td>
            <td>no asthma diagnosis</td>
            <td>new asthma diagnosis</td>
            <td>asthma</td>
        </tr>
        <tr>
            <td><code class="notranslate">d1_dx</code></td>
            <td>None</td>
            <td>no asthma diagnosis</td>
            <td>no new asthma diagnosis</td>
            <td>no asthma</td>
        </tr>
        </tbody>
    </table>

* :math:`a_{1, \text{dx}}` is the proportion of the population with risk factor combination :math:`\lambda`
  who didn't have an asthma diagnosis at :math:`t=0` and were diagnosed at :math:`t=1`
  :math:`\rightarrow` have asthma at :math:`t=1`
* :math:`b_{1, \text{dx}}` is the proportion of the population with risk factor combination :math:`\lambda`
  who didn't have an asthma diagnosis at :math:`t=0` and were not diagnosed with asthma at :math:`t=1`,
  :math:`\rightarrow` don't have asthma at :math:`t=1`
* :math:`c_{1, \text{dx}}` is the proportion of the population with no risk factors (:math:`\lambda = 0`)
  who didn't have an asthma diagnosis at :math:`t=0` and were diagnosed at :math:`t=1`
  :math:`\rightarrow` have asthma at :math:`t=1`
* :math:`d_{1, \text{dx}}` is the proportion of the population with no risk factors (:math:`\lambda = 0`)
  who didn't have an asthma diagnosis at :math:`t=0` and were not diagnosed with asthma at
  :math:`t=1`, :math:`\rightarrow` don't have asthma at :math:`t=1`


Current Contingency Table
^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, we can compute the contingency table for the current timepoint, :math:`t=1`:

.. raw:: html

    <table class="table">
        <thead>
        <tr>
            <th></th>
            <th>asthma, outcome +</th>
            <th>asthma, outcome -</th>
            <th></th>
        </tr>
        </thead>
        <tbody>
        <tr>
            <td>risk factor <code class="notranslate">λ</code>, outcome +</td>
            <td><code class="notranslate">a1</code></td>
            <td><code class="notranslate">b1</code></td>
            <td><code class="notranslate">n1</code></td>
        </tr>
        <tr>
            <td>risk factor <code class="notranslate">λ</code>, outcome -</td>
            <td><code class="notranslate">c1</code></td>
            <td><code class="notranslate">d1</code></td>
            <td></td>
        </tr>
        <tr>
            <td></td>
            <td><code class="notranslate">n2</code></td>
            <td></td>
            <td><code class="notranslate">n</code></td>
        </tr>
        </tbody>
    </table>


where:

.. math::
    a_1 &= a_{1, \text{ra}} + a_{1, \text{dx}} \\
    b_1 &= b_{1, \text{ra}} + b_{1, \text{dx}} \\
    c_1 &= c_{1, \text{ra}} + c_{1, \text{dx}} \\
    d_1 &= d_{1, \text{ra}} + d_{1, \text{dx}}

From these values, we can compute the odds ratio:

.. math::
    \Omega = \dfrac{a_1 \cdot d_1}{b_1 \cdot c_1}


Optimization
^^^^^^^^^^^^^^^^^

We want to find the beta parameters that minimize the difference between the predicted odds
ratio :math:`\Omega` and the observed odds ratio :math:`\omega_{\lambda}`.

.. math::
    \sum_{i=1}^{N}\sum_{\lambda=1}^{n}
      \dfrac{\left| \log(\Omega^{(i)}) - \log(\omega_{\lambda}^{(i)}) \right|}{N}
