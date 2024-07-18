import pytest
import pathlib
import json
import numpy as np
from leap.agent import Agent
from leap.family_history import FamilyHistory
from leap.antibiotic_exposure import AntibioticExposure
from leap.control import Control
from leap.exacerbation import Exacerbation
from leap.severity import ExacerbationSeverity
from tests.utils import __test_dir__


@pytest.fixture(scope="function")
def config():
    with open(pathlib.Path(__test_dir__, "data/config.json"), "r") as file:
        config = json.load(file)
    return config


@pytest.mark.parametrize(
    "hyperparameters, parameters, p",
    [
        (
            {
                "α": [1000, 0.00001, 0.00001, 0.00001],
                "k": 100
            },
            {
                "βprev_hosp_ped": 1.79,
                "βprev_hosp_adult": 2.88
            },
            [1.0, 0.0, 0.0, 0.0]
        ),
    ]
)
def test_exacerbation_severity_constructor(hyperparameters, parameters, p):
    exacerbation_severity = ExacerbationSeverity(
        hyperparameters=hyperparameters,
        parameters=parameters
    )
    assert exacerbation_severity.hyperparameters["α"] == hyperparameters["α"]
    assert exacerbation_severity.hyperparameters["k"] == hyperparameters["k"]
    np.testing.assert_array_equal(exacerbation_severity.parameters["p"], p)
    assert exacerbation_severity.parameters["βprev_hosp_ped"] == parameters["βprev_hosp_ped"]
    assert exacerbation_severity.parameters["βprev_hosp_adult"] == parameters["βprev_hosp_adult"]


@pytest.mark.parametrize(
    (
        "hyperparameters, parameters, sex, age, year, year_index, province, asthma_age,"
        "expected_prob"
    ),
    [
        (
            {
                "α": [0.00001, 0.00001, 0.00001, 1000],
                "k": 100
            },
            {
                "βprev_hosp_ped": 1.79,
                "βprev_hosp_adult": 2.88
            },
            True,
            90,
            2024,
            0,
            "BC",
            85,
            True
        ),
    ]
)
def test_exacerbation_severity_compute_hospitalization_prob(
    config, hyperparameters, parameters, sex, age, year, year_index, province, asthma_age,
    expected_prob
):
    """Test the ``compute_hospitalization_prob`` method of the ``ExacerbationSeverity`` class.

    Setting the exacerbation hyperparameter ``β0_μ=20.0`` ensures that every agent aged 4+ has an
    asthma exacerbation.

    Setting the exacerbation_severty hyperparameter ``α=[0.00001, 0.00001, 0.00001, 1000]`` ensures
    that probability of having an exacerbation with high severity is 1.
    """

    config["exacerbation"]["hyperparameters"]["β0_μ"] = 20.0
    exacerbation_severity = ExacerbationSeverity(
        hyperparameters=hyperparameters,
        parameters=parameters
    )
    agent = Agent(
        sex=sex,
        age=age,
        year=year,
        year_index=year_index,
        family_history=FamilyHistory(config=config["family_history"]),
        antibiotic_exposure=AntibioticExposure(config=config["antibiotic_exposure"]),
        province=province,
        month=1,
        ssp=config["pollution"]["SSP"],
        has_asthma=True,
        asthma_age=asthma_age
    )
    exacerbation = Exacerbation(config=config["exacerbation"], province=province)
    control = Control(config=config["control"])
    prob = exacerbation_severity.compute_hospitalization_prob(
        agent=agent,
        control=control,
        exacerbation=exacerbation
    )
    assert prob == expected_prob
