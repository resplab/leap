=====================
Asthma Control Model
=====================

Data
====

Raw Data
***********

``EBA``	was	a	prospective representative observational study of ``618`` participants aged ``1-85``
years (74% were >= 18 years old) with self-reported, physician-diagnosed asthma from BC.
The measurements were taken every 3 months for a year. Among ``613`` patients, only 6% were lost
during the one-year follow-up. There were at least ``500`` cases for each asthma control level.
More females were in the cohort as expected (adult asthma is more prevalent among females).
Asthma control level changed during the follow-up for ``79%`` of the patients.

We followed the 2020 GINA guidelines to define asthma control level by using the sum of the
four indicator variables (0 if no and 1 if yes):

1. daily symptoms
2. nocturnal symptoms
3. inhaler use
4. limited activities

in the last 3 months before each measurement. If the sum is zero, then the asthma control level is
controlled. If it is less than 3, then it is partially controlled. Otherwise, it is uncontrolled.
For responses with *do not know* to the indicator variables, we treated them as a *no*.
In this analysis, we did not consider treatment nor whether a patient experienced an exacerbation
in the last 3 months before the visit. We excluded two patients whose asthma diagnosis dates were
earlier than they were born and three patients who had no asthma diagnosis dates, for a final
count of 613 patients.

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
        1 = yes, 2 = no
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">nocturnalSymptoms</code></td>
        <td>
        <code class="notranslate">int</code>
        </td>
        <td>
        1 = yes, 2 = no
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">inhalerUse</code></td>
        <td>
        <code class="notranslate">int</code>
        </td>
        <td>
        1 = yes, 2 = no
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">limitedActivities</code></td>
        <td>
        <code class="notranslate">int</code>
        </td>
        <td>
        1 = yes, 2 = no
        </td>
      </tr>
      <tr>
        <td><code class="notranslate">exacerbations</code></td>
        <td>
        <code class="notranslate">int</code>
        </td>
        <td>
        TODO
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
        1 = TODO, 2 = TODO, 3 = TODO
        </td>
      </tr>
    </tbody>
  </table>


Processed Data
***************

In keeping with ``Python`` conventions, the columns were converted to snake case. In addition,
``studyId`` was renamed to ``patient_id``, as ``studyId`` indicates that the ID is for a given
study, when in fact the ID was for an individual patient.

The variables ``daytimeSymptoms``, ``nocturnalSymptoms``, ``inhalerUse``, and ``limitedActivities``
were converted to binary variables, where ``1 = True`` and ``0 = False``.

We also needed to compute the asthma control level from the four indicator variables. We first
computed the ``control_score``, defined as:

.. math::

  \text{control_score} = \text{daytime_symptoms} + \text{nocturnal_symptoms} +
  \text{inhaler_use} + \text{limited_activities}

which has a minimum value of ``0`` (maximum control) and a maximum value of ``4`` (minimum control).

Then we defined the asthma control level as follows:

.. math::

  \text{control_level} = \begin{cases}
    1 & \text{control_score} = 0 \\
    1 &  0 ~ < \text{control_score} < 3 \\
    3 & \text{control_score} \geq 3
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
        TODO
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
        1 = TODO, 2 = TODO, 3 = TODO
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

Model
=====

Our goal is to fit a model for generating the proportion of time that an individual labelled as
``asthmatic`` spends in each control level.

Ordinal Regression
******************

``Ordinal regression`` is a type of regression analysis that is used when the response variable
(in our case, the control level) is ordered, but the intervals between the levels are
arbitrary. In our case, the order of the control levels matters
(``controlled`` < ``partially-controlled`` < ``uncontrolled``), but the numbers assigned to them
and the distance between those numbers are arbitrary.

To begin, we define our variables:

* :math:`i`: the patient index
* :math:`k`: the asthma control level, where :math:`k \in \{1,2,3\}`
* :math:`y^{(i)}`: the asthma control level for patient :math:`i`, where :math:`y^{(i)} \in \{1,2,3\}`
* :math:`\theta_k`: the threshold parameter for the :math:`k^{th}` control level
* :math:`x_n^{(i)}`: the :math:`n^{th}` covariate for patient :math:`i`
* :math:`\beta_n`: the coefficient for the :math:`n^{th}` covariate

Then the model is:

.. math::

  \begin{align}
    P(y^{(i)} \leq k) = \sigma(\theta_k + \sum_{n=1}^{N} \beta_n x_n^{(i)})
  \end{align}

where :math:`\sigma` is the logistic function:

.. math::

  \begin{align}
    \sigma(x) = \dfrac{1}{1 + e^{-x}}
  \end{align}

and the covariates are:

.. math::

  \sum_{n=1}^{N} \beta_n x_n := 
    \beta_{\text{age}} \cdot \text{age} +
    \beta_{\text{sex}} \cdot \text{sex} +
    \beta_{\text{age2}} \cdot \text{age}^2 +
    \beta_{\text{sexage}} \cdot \text{sex} \cdot \text{age} +
    \beta_{\text{sexage2}} \cdot \text{sex} \cdot \text{age}^2

To obtain the probability that a patient is in a specific control level, we use the following:

.. math::

  \begin{align}
    P(y^{(i)} = k) = P(y^{(i)} \leq k) - P(y^{(i)} \leq k-1)
  \end{align}


Random Effects
*****************

In our model, we also include a random effect to account for the correlation between
measurements from the same patient. This is important because the measurements are taken
repeatedly over time, and we expect that the measurements from the same patient will be more
similar to each other than to measurements from different patients. The random effect is
assumed to be normally distributed with mean zero and variance :math:`\sigma^2`.
The model with random effects is:

.. math::

  \begin{align}
    P(y^{(i)} \leq k) = \sigma(\theta_k + \sum_{n=1}^{N} \beta_n x_n^{(i)} + \beta_0^{(i)})
  \end{align}

where :math:`\beta_0^{(i)}` is the random effect for patient :math:`i`.

Fitting the Model with EBA Data
*******************************

The predictions from this model are the probabilities of being in each of the
control levels during the 3-month period, but we make the following assumptions to allow us to
apply these predictions to our simulation:

1. We assume that the probability of being in each of the control levels is equivalent to the
   proportion of time spent in each of the control levels.
2. We assume that we may extend these predictions from a 3-month period to a 1-year period
   (this is the time cycle of the simulation).
3. We assume that the probability of being in a control level does not depend on time.
4. We assume that the probability of being in a control level does not depend on the past history
   of asthma control.
5. We assume that the probability of being in a control level does not depend on the past history
   of exacerbations.

In short, for each virtual individual (agent) labelled as asthmatic, we sampled an
individual-specific intercept from the estimated distribution of the random effects, and with that
intercept in the asthma control prediction model, we simulated the proportion of time spent in each
of the control levels in each time cycle.


Predictions
==================

Once the ordinal regression model has been fit on the ``EBA`` dataset, the coefficients are
saved to the ``leap/processed_data/config.json`` file. During the simulation, these coefficients
are used to determine the probability of being in each of the control levels for each agent
labelled as ``asthmatic``. 

.. math::

  \begin{align}
    P(y^{(i)} \leq k) = \sigma(\theta_k + \sum_{n=1}^{N} \beta_n x_n^{(i)} + \beta_0^{(i)})
  \end{align}

where :math:`\beta_0^{(i)}` is assigned to each agent at the beginning of the simulation,
sampled randomly from a normal distribution with :math:`\mu = 0` and :math:`\sigma` as
calculated when the model was fit.
