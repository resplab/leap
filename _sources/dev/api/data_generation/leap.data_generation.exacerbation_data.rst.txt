Exacerbation Calibration Data
=============================

The number of exacerbations in a given year is modelled using a Poisson distribution.
The formula is:

.. math::

   \begin{align}
   N_{\text{exacerbations}} &\sim \text{Poisson}(\lambda) = \dfrac{\lambda^k e^{-\lambda}}{k!}
   \end{align}

Here :math:`\lambda` is the expected number of exacerbations per year. To obtain :math:`\lambda`,
we must perform a Poisson regression. The Poisson regression assumes that the value we are
interested in can be approximated using the following formula:

.. math::

   \begin{align}
   \ln(\lambda) &= \ln(\alpha) + \beta_0 + \beta_{a} a + \beta_{s} s + \sum_{i=1}^3 \beta_i c_i 
   \end{align}

where:

* :math:`\alpha`: **calibration multiplier**
* :math:`a`: age
* :math:`\beta_a`: age constant
* :math:`s`: sex
* :math:`\beta_s`: sex constant
* :math:`c_i`: relative time spent in control level :math:`i`
* :math:`\beta_i`: control level constant

In the ``exacerbation_data.py`` file, we are interested in calculating :math:`\alpha`. If we
rewrite the equation, the meaning of :math:`\alpha` becomes more apparent:

.. math::

   \begin{align}
   \lambda &= \alpha \cdot e^{\beta_0} e^{\beta_{a} a} e^{\beta_{s} s} \prod_{i=1}^3 e^{\beta_i c_i} 
   \end{align}


How do we obtain :math:`\alpha`? We again assume that the mean value has the same form as in a
Poisson regression, with the following formula:

.. math::

   \begin{align}
   \ln(\lambda_{C}) &= \sum_{i=1}^3 \gamma_i c_i 
   \end{align}

* :math:`\lambda_C`: the average number of exacerbations in a given year
* :math:`c_i`: relative time spent in control level :math:`i`
* :math:`\gamma_i`: control level constant (different from :math:`\beta_i` above)

Here, the :math:`\gamma_i` values were calculated from the
`Economic Burden of Asthma (EBA) study <https://bmjopen.bmj.com/content/3/9/e003360.long>`_
and are given by:

.. math::

   \begin{align}
   \gamma_1 &:= 0.1880058 & \text{rate(exacerbation | fully controlled)}\\
   \gamma_2 &:= 0.3760116 & \text{rate(exacerbation | partially controlled)}\\
   \gamma_3 &:= 0.5640174 & \text{rate(exacerbation | uncontrolled)}
   \end{align}

The number of exacerbations predicted by the model is then:

.. math::

   \begin{align}
   N_{\text{exac}}^{\text{(pred)}} &= \lambda_C \cdot N_{\text{asthma}}
   \end{align}

* :math:`N_{\text{asthma}}`: the number of people in a given
  year, age, and sex

and number of hospitalizations is:

.. math::

   \begin{align}
   N_{\text{hosp}}^{\text{(pred)}} &= N_{\text{exac}}^{\text{(pred)}} \cdot P(\text{hosp})
   \end{align}

* :math:`N_{\text{hosp}}^{\text{(pred)}}`: the predicted number of hospitalizations for a given
  year, age, and sex
* :math:`P(\text{hosp})`: the probability of hospitalization due to asthma given the patient has an
  asthma exacerbation

Finally, :math:`\alpha` can be computed:

.. math::

   \begin{align}
   \alpha(a, s, y) &= \dfrac{N_{\text{hosp}}(a, s, y)}{N_{\text{hosp}}^{\text{(pred)}}(a, s, y)}
   \end{align}


To run the data generation for the exacerbation data:

.. code-block:: bash

   cd LEAP
   python3 leap/data_generation/exacerbation_data.py


leap.data\_generation.exacerbation\_data module
************************************************

.. automodule:: leap.data_generation.exacerbation_data
   :members:
   :undoc-members:
   :show-inheritance:
