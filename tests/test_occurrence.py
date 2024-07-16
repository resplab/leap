import pytest
import pathlib
import json
import numpy as np
from leap.agent import Agent
from leap.family_history import FamilyHistory
from leap.antibiotic_exposure import AntibioticExposure
from leap.occurrence import Incidence, Prevalence, agent_has_asthma
from tests.utils import __test_dir__


@pytest.fixture(scope="function")
def config():
    with open(pathlib.Path(__test_dir__, "data/config.json"), "r") as file:
        config = json.load(file)
    return config


@pytest.mark.parametrize(
    "hyperparameters, parameters, max_age",
    [
        (
            {
                "β0_μ": 0.0,
                "β0_σ": 0.00000001
            },
            {
                "β0": 34.6,
                "βsex": -9.5,
                "βage": [-6.6, 7.7, -5.6, 3.9, -1.3],
                "βyear": -0.019,
                "βsexage": [-4.4, 4.7, -2.6, 0.79, 0.95],
                "βsexyear": 0.0046,
                "βfam_hist": [0.12, 0.36],
                "βabx_exp": [1.8, -0.29, 0.053]
            },
            60
        ),
    ]
)
def test_incidence_constructor(hyperparameters, parameters, max_age):
    incidence = Incidence(hyperparameters=hyperparameters, parameters=parameters, max_age=max_age)
    assert incidence.hyperparameters["β0_μ"] == hyperparameters["β0_μ"]
    assert incidence.hyperparameters["β0_σ"] == hyperparameters["β0_σ"]
    assert incidence.parameters["β0"] == parameters["β0"]
    assert incidence.parameters["βsex"] == parameters["βsex"]
    np.testing.assert_array_equal(incidence.parameters["βage"], parameters["βage"])
    assert incidence.parameters["βyear"] == parameters["βyear"]
    np.testing.assert_array_equal(incidence.parameters["βsexage"], parameters["βsexage"])
    assert incidence.parameters["βsexyear"] == parameters["βsexyear"]
    np.testing.assert_array_equal(incidence.parameters["βfam_hist"], parameters["βfam_hist"])
    np.testing.assert_array_equal(incidence.parameters["βabx_exp"], parameters["βabx_exp"])
    assert incidence.max_age == max_age


@pytest.mark.parametrize(
    (
        "hyperparameters, incidence_parameters, family_history_parameters, max_age,"
        "age, sex, starting_year, year, province, has_asthma"
    ),
    [
        (
            {
                "β0_μ": 0.0,
                "β0_σ": 0.00000001
            },
            {
                "β0": 34.6,
                "βsex": -9.5,
                "βage": [-6.6, 7.7, -5.6, 3.9, -1.3],
                "βyear": -0.019,
                "βsexage": [-4.4, 4.7, -2.6, 0.79, 0.95],
                "βsexyear": 0.0046,
                "βfam_hist": [100, 0],
                "βabx_exp": [1.8, -0.29, 0.053]
            },
            {
                "p": 1.0
            },
            63,
            24,
            False,
            2024,
            2025,
            "CA",
            True
        ),
    ]
)
def test_agent_has_asthma(
    config, hyperparameters, incidence_parameters, family_history_parameters, max_age,
    age, sex, starting_year, year, province, has_asthma
):
    """
    Setting the incidence parameter ``βfam_hist = [100, 0]`` and the family history parameter
    ``p = 1.0`` ensures that the probability of an agent being diagnosed with asthma is 1. The
    minimum age for an asthma diagnosis is 3, so setting the agent age to 24 ensures that the agent
    is diagnosed with asthma.
    """
    year_index = year - starting_year + 1
    incidence = Incidence(
        hyperparameters=hyperparameters, parameters=incidence_parameters, max_age=max_age
    )
    prevalence = Prevalence(config=config["prevalence"])
    agent = Agent(
        sex=sex,
        age=age,
        year=year,
        year_index=year_index,
        family_history=FamilyHistory(parameters=family_history_parameters),
        antibiotic_exposure=AntibioticExposure(config=config["antibiotic_exposure"]),
        province=province,
        month=1,
        ssp=config["pollution"]["SSP"]
    )

    assert agent.has_asthma is False
    assert agent_has_asthma(
        agent=agent, incidence=incidence, prevalence=prevalence, occurrence_type="inc"
    ) == has_asthma
