================
``config.json``
================

The ``config.json`` file contains the parameters used in the simulation.

.. warning::

    Users should NOT edit this file, with the exception of the ``simulation`` section. All the
    other parameters have been computed from fitting statistical models or from other papers.

.. code-block:: json

    {
        "simulation": {
            "min_year": 2024,
            "time_horizon": 13,
            "province": "BC",
            "population_growth_type": "LG",
            "num_births_initial": 100,
            "max_age": 111
        },
        "antibiotic_exposure": {
            "parameters": {
                "β0": 110.000442,
                "βyear": -0.055100,
                "β2005": 55.033675,
                "βsex": 0.249033,
                "θ": 727.383,
                "β2005_year": -0.027437,
                "fixyear": null,
                "βfloor": 0.05
            }
        },
        "census_table": {
            "year": 2021
        },
        "control": {
            "hyperparameters": {
                "β0_μ": 0.0,
                "β0_σ": 1.678728
            },
            "parameters": {
                "βage": 3.5430381,
                "βage2": -3.4980710,
                "βsexage": -0.8161495,
                "βsexage2": -1.1654264,
                "βsex": 0.2347807,
                "θ": [-0.3950, 2.754]
            }
        },
        "cost": {
            "parameters": {
                "control": [2372, 2965, 3127],
                "exac": [130, 594, 2425, 9900]
            },
            "exchange_rate_usd_cad": 1.66
        },
        "death": {
            "parameters": {}
        },
        "exacerbation": {
            "hyperparameters": {
                "β0_μ": 0.0,
                "β0_σ": 0.0000001
            },
            "parameters": {
                "βcontrol_C": -1.6712824655642424,
                "βcontrol_PC": -0.978135285004297,
                "βcontrol_UC": -0.5726701768961326
            },
            "initial_rate": 0.347
        },
        "exacerbation_severity": {
            "hyperparameters": {
                "α": [0.495, 0.195, 0.283, 0.026],
                "k": 100
            },
            "parameters": {
                "p": [0.25, 0.25, 0.25, 0.25],
                "βprev_hosp_ped": 1.79,
                "βprev_hosp_adult": 2.88
            }
        },
        "family_history": {
            "parameters": {
            "p": 0.2927242
            }
        },
        "incidence": {
            "hyperparameters": {
                "β0_μ": 0,
                "β0_σ": 0.00000001
            },
            "parameters": {
                "β0": 34.63398846,
                "βsex": -9.52017810,
                "βage": [-6.64423331, 7.73720625, -5.63121394, 3.90920803, -1.39497027],
                "βyear": -0.01967344,
                "βsexage": [-4.45607619, 4.70483885, -2.61760564, 0.79555703, 0.95476291],
                "βsexyear": 0.00461397,
                "βfam_hist": [0.12221763272424911, 0.3619942],
                "βabx_exp": [1.826, -0.2920745, 0.053]
            },
            "max_age": 63
        },
        "pollution": {
            "SSP": "SSP1_2.6"
        },
        "prevalence": {
            "hyperparameters": {
                "β0_μ": 0,
                "β0_σ": 0.00000001
            },
            "parameters": {
                "β0": -2.28093577,
                "βsex": -0.10755806,
                "βage": [
                    1.79932480805632, -2.17989374225804, 3.64152189395539,
                    -2.91796538427475, 1.43423653685647
                ],
                "βyear": [2.83586405, -1.18097542],
                "βsexage": [
                    -7.69209530818354, 2.68306716462003, 0.865308192929771,
                    -0.656000992252807, -0.0270826201453694
                ],
                "βsexyear": [1.29279956487906, 0.036861276364171],
                "βyearage": [
                    50.610032709273, 6.51236955045884, -39.4569160874519,
                    3.69176099747937, 15.9637932343298, -4.79271775804693,
                    -7.14281869955998, 4.18656498490802, -4.88274672641455, -3.3603262281752
                ],
                "βsexyearage": [
                    -3.19896302105009, 7.24422362459046, -25.7979736592919, 0.253623898303176,
                    11.3848773603672, -2.57625491419054, 7.61284030050534, 4.17111534541718,
                    -15.2128066205219, 3.70514542334455
                ],
                "βfam_hist": [0.12221763272424911, 0.37662555231482536],
                "βabx_exp": [1.826, -0.225, 0.053]
                },
                "max_age": 63
        },
        "utility": {
            "parameters": {
                "βcontrol": [0.06, 0.09, 0.10],
                "βexac_sev_hist": [
                    0.006153846153846154, 0.016923076923076923,
                    0.019230769230769232, 0.02153846153846154
                ]
            }
        }
    }


.. list-table::
    :header-rows: 1

    * - Module
      - Parameter
      - Description
      - Source
      - Documentation
    * - Antibiotic Exposure
      - ``β0``
      - Intercept for antibiotic exposure model
      - Fitted using a GLM
      - :ref:`antibiotic_exposure_model`
    * - Antibiotic Exposure
      - ``βyear``
      - Effect of year on antibiotic exposure model
      - Fitted using a GLM
      - :ref:`antibiotic_exposure_model`
    * - Antibiotic Exposure
      - ``β2005``
      - Effect of year 2005 on antibiotic exposure model
      - Fitted using a GLM
      - :ref:`antibiotic_exposure_model`
    * - Antibiotic Exposure
      - ``βsex``
      - Effect of sex 
      - Fitted using a GLM
      - :ref:`antibiotic_exposure_model`
    * - Antibiotic Exposure
      - ``θ``
      - Overdispersion parameter for antibiotic exposure model
      - Fitted using a GLM
      - :ref:`antibiotic_exposure_model`
    * - Antibiotic Exposure
      - ``β2005_year``
      - Interaction effect of year and year 2005 on antibiotic exposure model
      - Fitted using a GLM
      - :ref:`antibiotic_exposure_model`
    * - Antibiotic Exposure
      - ``fixyear``
      - Fixed year for antibiotic exposure model (if null, no fixed year)
      - Fitted using a GLM
      - :ref:`antibiotic_exposure_model`
    * - Antibiotic Exposure
      - ``βfloor``
      - Floor for antibiotic exposure probability
      - Fitted using a GLM
      - :ref:`antibiotic_exposure_model`
    * - Control
      - ``β0_μ``
      - Mean of the intercept for asthma control model
      - Fitted using an ordinal regression model
      - :ref:`control-params`
    * - Control
      - ``β0_σ``
      - Standard deviation of the intercept for asthma control model
      - Fitted using an ordinal regression model
      - :ref:`control-model`
    * - Control
      - ``βage``
      - :math:`\beta` coefficient for the :math:`\text{age}` term of the asthma control model
      - Fitted using an ordinal regression model
      - :ref:`control-model`
    * - Control
      - ``βage2``
      - :math:`\beta` coefficient for the :math:`\text{age}^2` term of the asthma control model
      - Fitted using an ordinal regression model
      - :ref:`control-model`
    * - Control
      - ``βsexage``
      - :math:`\beta` coefficient for the :math:`\text{sex}*\text{age}` term of the asthma control model
      - Fitted using an ordinal regression model
      - :ref:`control-model`
    * - Control
      - ``βsexage2``
      - :math:`\beta` coefficient for the :math:`\text{sex}*\text{age}^2` term of the asthma control model
      - Fitted using an ordinal regression model
      - :ref:`control-model`
    * - Control
      - ``βsex``
      - :math:`\beta` coefficient for the :math:`\text{sex}` term of the asthma control model
      - Fitted using an ordinal regression model
      - :ref:`control-model`
    * - Control
      - ``θ``
      - Thresholds for the asthma control model
      - Fitted using an ordinal regression model
      - :ref:`control-model`
    * - Cost
      - ``control``
      - Direct cost of asthma due to asthma control levels (1 = well-controlled, 2 = partially-controlled, 3 = uncontrolled)
        in USD
      - :cite:`yaghoubi`
      - :ref:`cost-data-control`
    * - Cost
      - ``exac``
      - Direct cost of asthma due to asthma exacerbation severity levels (1 = mild, 2 = moderate, 3 = severe, 4 = very severe)
        in USD
      - :cite:`yaghoubi`
      - :ref:`cost-data-exacerbations`
    * - Exacerbation
      - ``β0_μ``
      - Mean of the intercept for asthma exacerbation model
      - Fitted using a Poisson regression model
      - :ref:`exacerbation-model`
    * - Exacerbation
      - ``β0_σ``
      - Standard deviation of the intercept for asthma exacerbation model
      - Fitted using a Poisson regression model
      - :ref:`exacerbation-model`
    * - Exacerbation
      - ``βcontrol_C``
      - :math:`\beta` coefficient for the :math:`\text{asthma control level = well-controlled}` term of the asthma exacerbation model
      - `Economic Burden of Asthma (EBA) study <https://bmjopen.bmj.com/content/3/9/e003360.long>`_
      - :ref:`exacerbation-model`
    * - Exacerbation
      - ``βcontrol_PC``
      - :math:`\beta` coefficient for the :math:`\text{asthma control level = partially-controlled}` term of the asthma exacerbation model
      - `Economic Burden of Asthma (EBA) study <https://bmjopen.bmj.com/content/3/9/e003360.long>`_
      - :ref:`exacerbation-model`
    * - Exacerbation
      - ``βcontrol_UC``
      - :math:`\beta` coefficient for the :math:`\text{asthma control level = uncontrolled}` term of the asthma exacerbation model
      - `Economic Burden of Asthma (EBA) study <https://bmjopen.bmj.com/content/3/9/e003360.long>`_
      - :ref:`exacerbation-model`
    * - Exacerbation
      - ``initial_rate``
      - Initial asthma exacerbation rate for newly incident asthma cases
      - Fitted using a Poisson regression model
      - :ref:`exacerbation-model`
    * - Exacerbation Severity
      - ``α``
      - Dirichlet prior parameters for asthma exacerbation severity model
      - Fitted using a Bayesian model
      - :ref:`exacerbation_severity_model`
    * - Exacerbation Severity
      - ``k``
      - Concentration parameter for asthma exacerbation severity model
      - Fitted using a Bayesian model
      - :ref:`exacerbation_severity_model`
    * - Exacerbation Severity
      - ``p``
      - Probability of asthma exacerbation severity levels (1 = mild, 2 = moderate, 3 = severe, 4 = very severe)
      - Fitted using a Bayesian model
      - :ref:`exacerbation_severity_model`
    * - Exacerbation Severity
      - ``βprev_hosp_ped``
      - Effect of previous pediatric hospitalization on asthma exacerbation severity
      - Fitted using a Bayesian model
      - :ref:`exacerbation_severity_model`
    * - Exacerbation Severity
      - ``βprev_hosp_adult``
      - Effect of previous adult hospitalization on asthma exacerbation severity
      - Fitted using a Bayesian model
      - :ref:`exacerbation_severity_model`
    * - Family History
      - ``p``
      - Probability of having a family history of asthma
      - Fitted using the CHMS data
      - :ref:`family-history-model`
    * - Incidence
      - ``β0_μ``
      - Mean of the intercept for asthma incidence model
      - Fitted using a logistic regression model
      - :ref:`incidence-model`
    * - Utility
      - ``βcontrol``
      - Disutility due to asthma control levels (1 = well-controlled, 2 = partially-controlled, 3 = uncontrolled)
      - :cite:`yaghoubi`
      - :ref:`utility-data-control`
    * - Utility
      - ``βexac_sev_hist``
      - Disutility due to asthma exacerbation severity levels (1 = mild, 2 = moderate, 3 = severe, 4 = very severe)
      - :cite:`einarson`
      - :ref:`utility-data-exacerbations`
