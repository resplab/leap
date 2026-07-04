.. _control-model:

=====================
Asthma Control Model
=====================

Data
====

Economic Burden of Asthma (EBA) Study
***********

``EBA`` was a prospective observational study of 618 participants (aged 1–85; 74% adults)
with self-reported, physician-diagnosed asthma from BC, with measurements taken every 3 months
for a year. Only 6% of the 613 patients were lost to follow-up, and asthma control level
changed during follow-up for 79% of patients.

We followed the 
`2020 GINA guidelines <https://ginasthma.org/wp-content/uploads/2020/04/GINA-2020-full-report_-final-_wms.pdf>`_
to define asthma control level by using the sum of the
four indicator variables (0 if no and 1 if yes) in the last 4 weeks before each measurement:

1. **daytime symptoms**: Daytime symptoms more than twice per week?
2. **nocturnal symptoms**: Any night waking due to asthma symptoms?
3. **inhaler use**: Reliever medication needed more than twice per week?
4. **limited activities**: Any activity limitation due to asthma?

Responses of *do not know* were treated as *no*. Five patients were excluded due to missing or
implausible asthma diagnosis dates (two with diagnosis dates predating birth, three with no
diagnosis date), for a final count of 613 patients.

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
        <td><code class="notranslate">studyId</code></td>
        <td>
        <code class="notranslate">int</code>
        </td>
        <td>
        8-digit patient ID
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">visit</code></td>
        <td>
        <code class="notranslate">int</code>
        </td>
        <td>
        The visit number. Visits were scheduled every 3 months for a year. A value in
        <code class="notranslate">[1, 5]</code>.
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">daytimeSymptoms</code></td>
        <td>
        <code class="notranslate">int</code>
        </td>
        <td>
        1 = yes, 2 = no, 3 = unknown
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">nocturnalSymptoms</code></td>
        <td>
        <code class="notranslate">int</code>
        </td>
        <td>
        1 = yes, 2 = no, 3 = unknown
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">inhalerUse</code></td>
        <td>
        <code class="notranslate">int</code>
        </td>
        <td>
        1 = yes, 2 = no, 3 = unknown
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">limitedActivities</code></td>
        <td>
        <code class="notranslate">int</code>
        </td>
        <td>
        1 = yes, 2 = no, 3 = unknown
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">exacerbations</code></td>
        <td>
        <code class="notranslate">int</code>
        </td>
        <td>
        Has there been an asthma exacerbation in the last 3 months? <br>
        1 = yes, 2 = no, 3 = unknown
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">sex</code></td>
        <td>
        <code class="notranslate">int</code>
        </td>
        <td>
            0 = female, 1 = male
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">age</code></td>
        <td>
        <code class="notranslate">float</code>
        </td>
        <td>
        Age in years
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">ageAtAsthmaDx</code></td>
        <td>
        <code class="notranslate">float</code>
        </td>
        <td>
        Age at asthma diagnosis
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">time_since_Dx</code></td>
        <td>
        <code class="notranslate">float</code>
        </td>
        <td>
        Time since asthma diagnosis in years
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">time_since_Dx_cat</code></td>
        <td>
        <code class="notranslate">int</code>
        </td>
        <td>
        1 = 0 - 4.93 years <br>
        2 = 5.00 - 19.95 years <br>
        3 = 20.06 - 74.20 years
        </td>
      </tr>
    </tbody>
  </table>


Processed Data
***************

Columns were converted to snake case and ``studyId`` was renamed to ``patient_id``. The four
GINA indicator variables were binarized (``1 = True``, ``0 = False``).

We also needed to compute the asthma control level from the four indicator variables. We first
computed the ``control_score``, defined as:

.. math::

  \text{control_score} = \text{daytime_symptoms} + \text{nocturnal_symptoms} +
  \text{inhaler_use} + \text{limited_activities}

which has a minimum value of ``0`` (maximum control) and a maximum value of ``4`` (minimum control).

Then we defined the asthma control level as follows:

.. math::

  \text{control_level} = \begin{cases}
    1 & \quad \text{control_score} = 0 \\
    2 &  \quad 0 ~ < \text{control_score} < 3 \\
    3 & \quad \text{control_score} \geq 3
  \end{cases}


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
        <td><code class="notranslate">patient_id</code></td>
        <td>
        <code class="notranslate">int</code>
        </td>
        <td>
        8-digit patient ID
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">visit</code></td>
        <td>
        <code class="notranslate">int</code>
        </td>
        <td>
        The visit number. Visits were scheduled every 3 months for a year. A value in
        <code class="notranslate">[1, 5]</code>.
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">daytime_symptoms</code></td>
        <td>
        <code class="notranslate">int</code>
        </td>
        <td>
          <code class="notranslate">1 = True</code>, 
          <code class="notranslate">0 = False</code>
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">nocturnal_symptoms</code></td>
        <td>
        <code class="notranslate">int</code>
        </td>
        <td>
          <code class="notranslate">1 = True</code>, 
          <code class="notranslate">0 = False</code>
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">inhaler_use</code></td>
        <td>
        <code class="notranslate">int</code>
        </td>
        <td>
          <code class="notranslate">1 = True</code>, 
          <code class="notranslate">0 = False</code>
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">limited_activities</code></td>
        <td>
        <code class="notranslate">int</code>
        </td>
        <td>
          <code class="notranslate">1 = True</code>, 
          <code class="notranslate">0 = False</code>
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">exacerbations</code></td>
        <td>
        <code class="notranslate">int</code>
        </td>
        <td>
          <code class="notranslate">1 = True</code>, 
          <code class="notranslate">0 = False</code>
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">sex</code></td>
        <td>
        <code class="notranslate">int</code>
        </td>
        <td>
            0 = female, 1 = male
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">age</code></td>
        <td>
        <code class="notranslate">float</code>
        </td>
        <td>
        Age in years
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">age_at_asthma_dx</code></td>
        <td>
        <code class="notranslate">float</code>
        </td>
        <td>
        Age at asthma diagnosis
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">time_since_dx</code></td>
        <td>
        <code class="notranslate">float</code>
        </td>
        <td>
        Time since asthma diagnosis in years
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">time_since_dx_cat</code></td>
        <td>
        <code class="notranslate">int</code>
        </td>
        <td>
        1 = 0 - 4.93 years <br>
        2 = 5.00 - 19.95 years <br>
        3 = 20.06 - 74.20 years
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">control_score</code></td>
        <td>
        <code class="notranslate">int</code>
        </td>
        <td>
        0 = maximum control, 4 = minimum control
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">control_level</code></td>
        <td>
        <code class="notranslate">int</code>
        </td>
        <td>
        Asthma control level:
        <ul>
          <li>1 = fully-controlled</li>
          <li>2 = partially-controlled</li>
          <li>3 = uncontrolled</li>
        </ul>
        </td>
      </tr>
    </tbody>
  </table>

Model: Ordinal Regression with Random Effects
=============================================

We use an ``ordinal regression`` model with a patient-specific ``random effect`` to predict
asthma control level from age and sex. See :ref:`ordinal-regression` for background
on ordinal regression and random effects.

The model is:

.. math::

  \text{logit}(P(y^{(i)} \leq k)) = \theta_k
    + \beta_{\text{age}} \cdot a^{(i)}
    + \beta_{\text{sex}} \cdot s^{(i)}
    + \beta_{\text{age}^2} \cdot {a^{(i)}}^2
    + \beta_{\text{age,sex}} \cdot a^{(i)} \cdot s^{(i)}
    + \beta_{\text{age}^2\text{,sex}} \cdot {a^{(i)}}^2 \cdot s^{(i)}
    + \beta_0^{(i)}

where:

.. list-table::
   :widths: 20 20 20 40
   :header-rows: 1

   * - Coefficient
     - Indices
     - Term
     - Description
   * - :math:`\theta_k`
     - :math:`k \in \{1, 2\}`
     -
     - level-specific threshold (2 thresholds for 3 control levels)
   * - :math:`\beta_{\text{age}}`
     -
     - :math:`a^{(i)}`
     - age main effect
   * - :math:`\beta_{\text{sex}}`
     -
     - :math:`s^{(i)}`
     - sex main effect
   * - :math:`\beta_{\text{age}^2}`
     -
     - :math:`{a^{(i)}}^2`
     - age quadratic term
   * - :math:`\beta_{\text{age,sex}}`
     -
     - :math:`a^{(i)} \cdot s^{(i)}`
     - :math:`\text{age} \times \text{sex}` interaction
   * - :math:`\beta_{\text{age}^2\text{,sex}}`
     -
     - :math:`{a^{(i)}}^2 \cdot s^{(i)}`
     - :math:`\text{age}^2 \times \text{sex}` interaction
   * - :math:`\beta_0^{(i)}`
     -
     -
     - patient-specific random effect; :math:`\beta_0^{(i)} \sim \mathcal{N}(0, \sigma^2)`

and :math:`y^{(i)}` is the observed control level, :math:`k \in \{1, 2, 3\}` is the control level
index, :math:`a^{(i)}` is the age, and :math:`s^{(i)}` is the sex of patient :math:`i`.

The probability of being in a specific control level is:

.. math::

  P(y^{(i)} = k) = P(y^{(i)} \leq k) - P(y^{(i)} \leq k-1)

Fitting the Model with EBA Data
*******************************

The model was fit on 3-month measurement intervals. We make the following assumptions to
apply its predictions to the simulation:

1. The probability of being in a control level is equivalent to the proportion of time
   spent in that control level.
2. Predictions generalise from the 3-month EBA measurement period to the simulation's
   time interval (1 month or 1 year).
3. Control level probabilities do not vary with calendar year.
4. Control level probabilities do not depend on past control history.
5. Control level probabilities do not depend on past exacerbation history.

For each agent with asthma, an individual-specific intercept is sampled from the estimated
random-effects distribution and held fixed for their simulated lifetime. At each time interval,
this intercept is used with the fitted model to generate the proportion of time spent in each
control level.


Predictions
==================

Once the ordinal regression model has been fit on the ``EBA`` dataset, the coefficients are
saved to the ``leap/processed_data/config.json`` file. During the simulation, these coefficients
are used to determine the probability of being in each of the control levels for each agent
labelled as having asthma.
