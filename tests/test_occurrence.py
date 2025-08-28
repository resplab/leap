import pytest
import pathlib
import json
import numpy as np
from leap.agent import Agent
from leap.family_history import FamilyHistory
from leap.antibiotic_exposure import AntibioticExposure
from leap.occurrence import Incidence, Prevalence, agent_has_asthma
from tests.utils import __test_dir__


POLY_PARAMETERS_INCIDENCE = {
    "alpha_age": [
        32.76923076923077,
        32.90289472637344,
        33.11652518493872,
        33.339224035842506,
        33.13305341556257
    ],
    "norm2_age": [
        519.9999999999999,
        178492.30769230757,
        49586442.54438886,
        12799523974.241816,
        3212003819462.7515,
        712446143502445.5
    ]
}

POLY_PARAMETERS_PREVALENCE = {
    "alpha_age": [
        32.76923076923077,
        32.90289472637344,
        33.11652518493872,
        33.339224035842506,
        33.13305341556257
    ],
    "norm2_age": [
        519.9999999999999,
        178492.30769230757,
        49586442.54438886,
        12799523974.241816,
        3212003819462.7515,
        712446143502445.5
    ],
    "alpha_year": [
        2009.5,
        2009.5
    ],
    "norm2_year": [
        519.9999999999999,
        17290.0,
        456456.00000000006
    ]
}


@pytest.fixture(scope="function")
def config():
    with open(pathlib.Path(__test_dir__, "data/config.json"), "r") as file:
        config = json.load(file)
    return config


@pytest.mark.parametrize(
    "parameters, poly_parameters, max_age",
    [
        (
            {
                "β0": 34.6,
                "βsex": -9.5,
                "βage": [-6.6, 7.7, -5.6, 3.9, -1.3],
                "βyear": -0.019,
                "βsexage": [-4.4, 4.7, -2.6, 0.79, 0.95],
                "βsexyear": 0.0046,
                "β_fam_hist": {
                    "β_fhx_0": 0.12,
                    "β_fhx_age": 0.36
                },
                "β_abx": [1.8, -0.29, 0.053]
            },
            POLY_PARAMETERS_INCIDENCE,
            60
        ),
    ]
)
def test_incidence_constructor(parameters, poly_parameters, max_age):
    incidence = Incidence(parameters=parameters, poly_parameters=poly_parameters, max_age=max_age)
    np.testing.assert_array_equal(
        incidence.poly_parameters["alpha_age"], poly_parameters["alpha_age"]
    )
    np.testing.assert_array_equal(
        incidence.poly_parameters["norm2_age"], poly_parameters["norm2_age"]
    )
    assert incidence.parameters["β0"] == parameters["β0"]
    assert incidence.parameters["βsex"] == parameters["βsex"]
    np.testing.assert_array_equal(incidence.parameters["βage"], parameters["βage"])
    assert incidence.parameters["βyear"] == parameters["βyear"]
    np.testing.assert_array_equal(incidence.parameters["βsexage"], parameters["βsexage"])
    assert incidence.parameters["βsexyear"] == parameters["βsexyear"]
    np.testing.assert_array_equal(incidence.parameters["β_fam_hist"], parameters["β_fam_hist"])
    np.testing.assert_array_equal(incidence.parameters["β_abx"], parameters["β_abx"])
    assert incidence.max_age == max_age



@pytest.mark.parametrize(
    "parameters, poly_parameters, max_age",
    [
        (
            {
                "β0": -2.28,
                "βsex": -0.11,
                "βage": [1.7, -2.1, 3.6, -2.9, 1.4],
                "βyear": [2.8, -1.1],
                "βsexage": [-7.69, 2.68, 0.86, -0.656, -0.027],
                "βsexyear": [1.29, 0.036],
                "βyearage": [50.6, 6.5, -39.4, 3.6, 15.9, -4.7, -7.1, 4.1, -4.8, -3.3],
                "βsexyearage": [-3.1, 7.2, -25.7, 0.2, 11.3, -2.5, 7.6, 4.1, -15.2, 3.7],
                "β_fam_hist": {
                    "β_fhx_0": 0.122,
                    "β_fhx_age": 0.376
                },
                "β_abx": [1.826, -0.225, 0.053]
            },
            POLY_PARAMETERS_PREVALENCE,
            60
        ),
    ]
)
def test_prevalence_constructor(parameters, poly_parameters, max_age):
    prevalence = Prevalence(parameters=parameters, poly_parameters=poly_parameters, max_age=max_age)
    np.testing.assert_array_equal(
        prevalence.poly_parameters["alpha_age"], poly_parameters["alpha_age"]
    )
    np.testing.assert_array_equal(
        prevalence.poly_parameters["norm2_age"], poly_parameters["norm2_age"]
    )
    np.testing.assert_array_equal(
        prevalence.poly_parameters["alpha_year"], poly_parameters["alpha_year"]
    )
    np.testing.assert_array_equal(
        prevalence.poly_parameters["norm2_year"], poly_parameters["norm2_year"]
    )
    assert prevalence.parameters["β0"] == parameters["β0"]
    assert prevalence.parameters["βsex"] == parameters["βsex"]
    np.testing.assert_array_equal(prevalence.parameters["βage"], parameters["βage"])
    np.testing.assert_array_equal(prevalence.parameters["βyear"], parameters["βyear"])
    np.testing.assert_array_equal(prevalence.parameters["βsexage"], parameters["βsexage"])
    np.testing.assert_array_equal(prevalence.parameters["βsexyear"], parameters["βsexyear"])
    np.testing.assert_array_equal(prevalence.parameters["βyearage"], parameters["βyearage"])
    np.testing.assert_array_equal(prevalence.parameters["βsexyearage"], parameters["βsexyearage"])
    np.testing.assert_array_equal(prevalence.parameters["β_fam_hist"], parameters["β_fam_hist"])
    np.testing.assert_array_equal(prevalence.parameters["β_abx"], parameters["β_abx"])
    assert prevalence.max_age == max_age


@pytest.mark.parametrize(
    (
        "poly_parameters_incidence, poly_parameters_prevalence, incidence_parameters,"
        "prevalence_parameters, family_history_parameters,"
        "max_age, age, sex, starting_year, year, province, occurrence_type, has_asthma"
    ),
    [
        (
            POLY_PARAMETERS_INCIDENCE,
            POLY_PARAMETERS_PREVALENCE,
            {
                "β0": 34.6,
                "βsex": -9.5,
                "βage": [-6.6, 7.7, -5.6, 3.9, -1.3],
                "βyear": -0.019,
                "βsexage": [-4.4, 4.7, -2.6, 0.79, 0.95],
                "βsexyear": 0.0046,
                "β_fam_hist": {
                    "β_fhx_0": 100,
                    "β_fhx_age": 0
                },
                "β_abx": {
                    "β_abx_0": 1.8,
                    "β_abx_age": -0.29,
                    "β_abx_dose": 0.053
                }
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
                "β_fam_hist": {
                    "β_fhx_0": 0.122,
                    "β_fhx_age": 0.376
                },
                "β_abx": {
                    "β_abx_0": 1.826,
                    "β_abx_age": -0.225,
                    "β_abx_dose": 0.053
                }
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
            "incidence",
            True
        ),
        (
            POLY_PARAMETERS_INCIDENCE,
            POLY_PARAMETERS_PREVALENCE,
            {
                "β0": 34.6,
                "βsex": -9.5,
                "βage": [-6.6, 7.7, -5.6, 3.9, -1.3],
                "βyear": -0.019,
                "βsexage": [-4.4, 4.7, -2.6, 0.79, 0.95],
                "βsexyear": 0.0046,
                "β_fam_hist": {
                    "β_fhx_0": 100,
                    "β_fhx_age": 0
                },
                "β_abx": {
                    "β_abx_0": 1.8,
                    "β_abx_age": -0.29,
                    "β_abx_dose": 0.053
                }
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
                "β_fam_hist": {
                    "β_fhx_0": 100,
                    "β_fhx_age": 0
                },
                "β_abx": {
                    "β_abx_0": 1.826,
                    "β_abx_age": -0.225,
                    "β_abx_dose": 0.053
                }
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
            "prevalence",
            True
        ),
        (
            POLY_PARAMETERS_INCIDENCE,
            POLY_PARAMETERS_PREVALENCE,
            {
                "β0": 34.6,
                "βsex": -9.5,
                "βage": [-6.6, 7.7, -5.6, 3.9, -1.3],
                "βyear": -0.019,
                "βsexage": [-4.4, 4.7, -2.6, 0.79, 0.95],
                "βsexyear": 0.0046,
                "β_fam_hist": {
                    "β_fhx_0": 100,
                    "β_fhx_age": 0
                },
                "β_abx": {
                    "β_abx_0": 1.8,
                    "β_abx_age": -0.29,
                    "β_abx_dose": 0.053
                }
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
                "β_fam_hist": {
                    "β_fhx_0": 0.122,
                    "β_fhx_age": 0.376
                },
                "β_abx": {
                    "β_abx_0": 1.826,
                    "β_abx_age": -0.225,
                    "β_abx_dose": 0.053
                }
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
            "prevalence",
            False
        )
    ]
)
def test_agent_has_asthma(
    config, poly_parameters_incidence, poly_parameters_prevalence, incidence_parameters,
    prevalence_parameters, family_history_parameters, max_age, age, sex, starting_year, year,
    province, occurrence_type, has_asthma
):
    """
    Setting the incidence parameter ``β_fam_hist = [100, 0]`` and the family history parameter
    ``p = 1.0`` ensures that the probability of an agent being diagnosed with asthma is 1. The
    minimum age for an asthma diagnosis is 3, so setting the agent age to 24 ensures that the agent
    is diagnosed with asthma.
    """
    year_index = year - starting_year + 1
    incidence = Incidence(
        parameters=incidence_parameters, poly_parameters=poly_parameters_incidence, max_age=max_age
    )
    prevalence = Prevalence(
       parameters=prevalence_parameters, poly_parameters=poly_parameters_prevalence, max_age=max_age
    )
    agent = Agent(
        sex=sex,
        age=age,
        year=year,
        year_index=year_index,
        family_history=FamilyHistory(probability=family_history_parameters["p"]),
        antibiotic_exposure=AntibioticExposure(config=config["antibiotic_exposure"]),
        province=province,
        month=1,
        ssp=config["pollution"]["SSP"]
    )

    assert agent.has_asthma is False
    assert agent_has_asthma(
        agent=agent, incidence=incidence, prevalence=prevalence, occurrence_type=occurrence_type
    ) == has_asthma
