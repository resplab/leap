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
    "hyperparameters, parameters, max_age",
    [
        (
            {
                "β0_μ": 0.0,
                "β0_σ": 0.00000001
            },
            {
                "β0": -2.28,
                "βsex": -0.11,
                "βage": [1.7, -2.1, 3.6, -2.9, 1.4],
                "βyear": [2.8, -1.1],
                "βsexage": [-7.69, 2.68, 0.86, -0.656, -0.027],
                "βsexyear": [1.29, 0.036],
                "βyearage": [50.6, 6.5, -39.4, 3.6, 15.9, -4.7, -7.1, 4.1, -4.8, -3.3],
                "βsexyearage": [-3.1, 7.2, -25.7, 0.2, 11.3, -2.5, 7.6, 4.1, -15.2, 3.7],
                "βfam_hist": [0.122, 0.376],
                "βabx_exp": [1.826, -0.225, 0.053]
            },
            60
        ),
    ]
)
def test_prevalence_constructor(hyperparameters, parameters, max_age):
    prevalence = Prevalence(hyperparameters=hyperparameters, parameters=parameters, max_age=max_age)
    assert prevalence.hyperparameters["β0_μ"] == hyperparameters["β0_μ"]
    assert prevalence.hyperparameters["β0_σ"] == hyperparameters["β0_σ"]
    assert prevalence.parameters["β0"] == parameters["β0"]
    assert prevalence.parameters["βsex"] == parameters["βsex"]
    np.testing.assert_array_equal(prevalence.parameters["βage"], parameters["βage"])
    np.testing.assert_array_equal(prevalence.parameters["βyear"], parameters["βyear"])
    np.testing.assert_array_equal(prevalence.parameters["βsexage"], parameters["βsexage"])
    np.testing.assert_array_equal(prevalence.parameters["βsexyear"], parameters["βsexyear"])
    np.testing.assert_array_equal(prevalence.parameters["βyearage"], parameters["βyearage"])
    np.testing.assert_array_equal(prevalence.parameters["βsexyearage"], parameters["βsexyearage"])
    np.testing.assert_array_equal(prevalence.parameters["βfam_hist"], parameters["βfam_hist"])
    np.testing.assert_array_equal(prevalence.parameters["βabx_exp"], parameters["βabx_exp"])
    assert prevalence.max_age == max_age


@pytest.mark.parametrize(
    (
        "hyperparameters, incidence_parameters, prevalence_parameters, family_history_parameters,"
        "max_age, age, sex, starting_year, year, province, occurrence_type, has_asthma"
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
                "β0": -2.28,
                "βsex": -0.11,
                "βage": [1.7, -2.1, 3.6, -2.9, 1.4],
                "βyear": [2.8, -1.1],
                "βsexage": [-7.69, 2.68, 0.86, -0.656, -0.027],
                "βsexyear": [1.29, 0.036],
                "βyearage": [50.6, 6.5, -39.4, 3.6, 15.9, -4.7, -7.1, 4.1, -4.8, -3.3],
                "βsexyearage": [-3.1, 7.2, -25.7, 0.2, 11.3, -2.5, 7.6, 4.1, -15.2, 3.7],
                "βfam_hist": [0.122, 0.376],
                "βabx_exp": [1.826, -0.225, 0.053]
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
            "inc",
            True
        ),
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
                "β0": -2.28,
                "βsex": -0.11,
                "βage": [1.7, -2.1, 3.6, -2.9, 1.4],
                "βyear": [2.8, -1.1],
                "βsexage": [-7.69, 2.68, 0.86, -0.656, -0.027],
                "βsexyear": [1.29, 0.036],
                "βyearage": [50.6, 6.5, -39.4, 3.6, 15.9, -4.7, -7.1, 4.1, -4.8, -3.3],
                "βsexyearage": [-3.1, 7.2, -25.7, 0.2, 11.3, -2.5, 7.6, 4.1, -15.2, 3.7],
                "βfam_hist": [100, 0],
                "βabx_exp": [1.826, -0.225, 0.053]
            },
            {
                "p": 1.0
            },
            111,
            24,
            False,
            2024,
            2025,
            "CA",
            "prev",
            True
        ),
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
                "β0": -2.28,
                "βsex": -0.11,
                "βage": [1.7, -2.1, 3.6, -2.9, 1.4],
                "βyear": [2.8, -1.1],
                "βsexage": [-7.69, 2.68, 0.86, -0.656, -0.027],
                "βsexyear": [1.29, 0.036],
                "βyearage": [50.6, 6.5, -39.4, 3.6, 15.9, -4.7, -7.1, 4.1, -4.8, -3.3],
                "βsexyearage": [-3.1, 7.2, -25.7, 0.2, 11.3, -2.5, 7.6, 4.1, -15.2, 3.7],
                "βfam_hist": [0.122, 0.376],
                "βabx_exp": [1.826, -0.225, 0.053]
            },
            {
                "p": 0.2927242
            },
            111,
            0,
            False,
            2024,
            2025,
            "CA",
            "prev",
            False
        ),
    ]
)
def test_agent_has_asthma(
    config, hyperparameters, incidence_parameters, prevalence_parameters, family_history_parameters,
    max_age, age, sex, starting_year, year, province, occurrence_type, has_asthma
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
    prevalence = Prevalence(
        hyperparameters=hyperparameters, parameters=prevalence_parameters, max_age=max_age
    )
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
        agent=agent, incidence=incidence, prevalence=prevalence, occurrence_type=occurrence_type
    ) == has_asthma
