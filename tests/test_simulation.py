import pytest
import pathlib
import json
import numpy as np
import pandas as pd
from leap.utility import Utility
from leap.agent import Agent
from leap.family_history import FamilyHistory
from leap.antibiotic_exposure import AntibioticExposure
from leap.control import ControlLevels
from leap.severity import ExacerbationSeverityHistory
from leap.simulation import Simulation
from leap.logger import get_logger
from tests.utils import __test_dir__

logger = get_logger(__name__)


@pytest.fixture(scope="function")
def config():
    with open(pathlib.Path(__test_dir__, "data/config.json"), "r") as file:
        config = json.load(file)
    return config


@pytest.mark.parametrize(
    (
        "min_year, time_horizon, province, population_growth_type, num_births_initial, max_age,"
        "antibiotic_exposure_parameters, prevalence_parameters, incidence_parameter_βfam_hist,"
        "family_history_parameters, exacerbation_hyperparameter_β0_μ, control_parameter_θ,"
        "sex, age, year_index,"
        "expected_has_asthma, expected_asthma_age, expected_asthma_status, expected_control_levels"
    ),
    [
        (
            2024,
            1,
            "CA",
            "M3",
            10,
            4,
            {
                "β0": -100000,
                "βyear": -0.01,
                "βsex": -1,
                "θ": 500,
                "fixyear": None,
                "βfloor": 0.0,
                "β2005": 1,
                "β2005_year": 1
            },
            {
                "β0": -20,
                "βsex": -20,
                "βage": [0.0, 0.0, 0.0, 0.0, 0.0],
                "βyear": [0.0, 0.0],
                "βsexage": [0.0, 0.0, 0.0, 0.0, 0.0],
                "βsexyear": [0.0, 0.0],
                "βyearage": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "βsexyearage": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "βfam_hist": [-100, 0],
                "βabx_exp": [0.0, 0.0, 0.0]
            },
            [100, 0],
            {"p": 1.0},
            5.0,
            [-0.3950, 2.754],
            True,
            4,
            1,
            False,
            None,
            False,
            None
        ),
        (
            2024,
            1,
            "CA",
            "M3",
            10,
            4,
            {
                "β0": -100000,
                "βyear": -0.01,
                "βsex": -1,
                "θ": 500,
                "fixyear": None,
                "βfloor": 0.0,
                "β2005": 1,
                "β2005_year": 1
            },
            {
                "β0": -0.01,
                "βsex": -0.01,
                "βage": [0.0, 0.0, 0.0, 0.0, 0.0],
                "βyear": [0.0, 0.0],
                "βsexage": [0.0, 0.0, 0.0, 0.0, 0.0],
                "βsexyear": [0.0, 0.0],
                "βyearage": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "βsexyearage": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "βfam_hist": [100, 0],
                "βabx_exp": [0.0, 0.0, 0.0]
            },
            [100, 0],
            {"p": 1.0},
            5.0,
            [-1 * 10**5, -1 * 10**5],
            True,
            4,
            1,
            True,
            3,
            True,
            [0.0, 0.0, 1.0]
        ),
    ]
)
def test_simulation_generate_initial_asthma(
    config, min_year, time_horizon, province, population_growth_type, num_births_initial, max_age,
    antibiotic_exposure_parameters, prevalence_parameters, incidence_parameter_βfam_hist,
    family_history_parameters, exacerbation_hyperparameter_β0_μ, control_parameter_θ,
    sex, age, year_index, expected_has_asthma, expected_asthma_age, expected_asthma_status,
    expected_control_levels
):
    """
    
    Setting the ``time_horizon = 1`` means that the agents are generated from the initial population
    table, and that no immigration happens.

    Setting the antibiotic exposure parameters below ensures that the antibiotic use is 0.

    Setting the ``num_births_initial = 10`` and starting in 2024 with growth type "M3", each of the
    age groups has 10 agents, for a total of 10 x 5 = 50 agents.

    Setting the incidence parameter ``βfam_hist = [100, 0]`` and the family history parameter
    ``p = 1.0`` ensures that the probability of an agent being diagnosed with asthma is 1. The
    maximum age is set to 4, and the minimum age required for an asthma diagnosis is 3. So all
    agents aged 4 should receive an asthma diagnosis.

    Setting the control parameter ``θ = [-1e5, -1e5]`` ensures that the ``control_levels`` are:
        FC: 0.0
        PC: 0.0
        UC: 1.0

    Setting the exacerbation parameter ``β0_μ = 5.0`` ensures that the number of exacerbations will
    be large.
    """

    config["simulation"] = {
        "min_year": min_year,
        "time_horizon": time_horizon,
        "province": province,
        "population_growth_type": population_growth_type,
        "num_births_initial": num_births_initial,
        "max_age": max_age
    }
    config["antibiotic_exposure"]["parameters"] = antibiotic_exposure_parameters
    config["prevalence"]["parameters"] = prevalence_parameters
    config["incidence"]["parameters"]["βfam_hist"] = incidence_parameter_βfam_hist
    config["family_history"]["parameters"] = family_history_parameters
    config["exacerbation"]["hyperparameters"]["β0_μ"] = exacerbation_hyperparameter_β0_μ
    config["control"]["parameters"]["θ"] = control_parameter_θ

    simulation = Simulation(config)
    agent = Agent(
        sex=sex,
        age=age,
        year=min_year,
        year_index=year_index,
        family_history=simulation.family_history,
        antibiotic_exposure=simulation.antibiotic_exposure,
        province=simulation.province,
        ssp=simulation.SSP
    )
    simulation.generate_initial_asthma(agent)
    assert agent.has_asthma == expected_has_asthma
    assert agent.asthma_age == expected_asthma_age
    assert agent.asthma_status == expected_asthma_status
    if expected_control_levels is None:
        assert agent.control_levels is None
    else:
        np.testing.assert_array_equal(agent.control_levels.as_array(), expected_control_levels)


