import pytest
import pathlib
import json
import numpy as np
import pandas as pd
from leap import simulation
from leap.utility import Utility
from leap.agent import Agent
from leap.family_history import FamilyHistory
from leap.antibiotic_exposure import AntibioticExposure
from leap.exacerbation import ExacerbationHistory
from leap.control import ControlLevels
from leap.severity import ExacerbationSeverityHistory
from leap.simulation import Simulation
from leap.outcome_matrix import OutcomeMatrix
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

@pytest.mark.parametrize(
    (
        "min_year, time_horizon, province, population_growth_type, num_births_initial, max_age,"
        "antibiotic_exposure_parameters, incidence_parameter_βfam_hist,"
        "family_history_parameters, exacerbation_hyperparameter_β0_μ, control_parameter_θ,"
        "sex, age, has_asthma, asthma_age, asthma_status, expected_control_levels,"
        "expected_exacerbation_history, expected_outcome_matrix_control,"
        "expected_outcome_matrix_exacerbation, expected_outcome_matrix_exacerbation_history"
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
            [100, 0],
            {"p": 1.0},
            5.0,
            [-1 * 10**5, -1 * 10**5],
            "F",
            4,
            True,
            4,
            True,
            np.array([0.0, 0.0, 1.0]),
            ExacerbationHistory(100, 0),
            pd.DataFrame(
                data={
                    "year": [2024] * 30,
                    "level": [0] * 10 + [1] * 10 + [2] * 10,
                    "age": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4] * 3,
                    "sex": ["F", "M"] * 15,
                    "prob": [0.0] * 20 + [1.0, 1.0] + [0.0] * 8
                }
            ),
            pd.DataFrame(
                data={
                    "year": [2024] * 10,
                    "age": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
                    "sex": ["F", "M"] * 5,
                    "n_exacerbations": [0] * 8 + [400, 0]
                }
            ),
            pd.DataFrame(
                data={
                    "year": [2024] * 10,
                    "severity": [0] * 10,
                    "age": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
                    "sex": ["F", "M"] * 5,
                    "p_exacerbations": [0.0] * 8 + [180.0] + [0.0]
                },
                index=range(0, 10)
            )
        ),
    ]
)              
def test_simulation_update_asthma_effects(
    config, min_year, time_horizon, province, population_growth_type, num_births_initial, max_age,
    antibiotic_exposure_parameters, incidence_parameter_βfam_hist, family_history_parameters,
    control_parameter_θ, exacerbation_hyperparameter_β0_μ, sex, age, has_asthma, asthma_age,
    asthma_status, expected_control_levels, expected_exacerbation_history,
    expected_outcome_matrix_control, expected_outcome_matrix_exacerbation,
    expected_outcome_matrix_exacerbation_history
):
    """
    Setting the ``time_horizon`` to 1 means that the agents are generated from the initial
    population table, and that the agents are generated from the initial population table, and
    that no immigration happens.

    Setting the antibiotic exposure parameters below ensures that the antibiotic use is 0.

    Setting the ``num_births_initial`` to 10 and starting in 2024 with growth type "M3", each of the
    age groups has 10 agents, for a total of 10 x 5 = 50 agents.

    Setting the incidence parameter ``βfam_hist`` to [100, 0] and the family history parameter ``p``
    to 1.0 ensures that the probability of an agent being diagnosed with asthma is 1. The maximum
    age is set to 4, and the minimum age required for an asthma diagnosis is 3. So all agents aged 4
    should receive an asthma diagnosis.

    Setting the control parameter ``θ`` to [-1e5, -1e5] ensures that the ``control_levels`` are:
        FC: 0.0
        PC: 0.0
        UC: 1.0

    Setting the exacerbation parameter ``β0_μ`` to 5.0 ensures that the number of exacerbations will
    be large.
    """
    max_year = min_year + time_horizon - 1
    config["simulation"] = {
        "min_year": min_year,
        "time_horizon": time_horizon,
        "province": province,
        "population_growth_type": population_growth_type,
        "num_births_initial": num_births_initial,
        "max_age": max_age
    }
    config["antibiotic_exposure"]["parameters"] = antibiotic_exposure_parameters
    config["incidence"]["parameters"]["βfam_hist"] = incidence_parameter_βfam_hist
    config["family_history"]["parameters"] = family_history_parameters
    config["control"]["parameters"]["θ"] = control_parameter_θ
    config["exacerbation"]["hyperparameters"]["β0_μ"] = exacerbation_hyperparameter_β0_μ
    outcome_matrix = OutcomeMatrix(
        until_all_die=False,
        min_year=min_year,
        max_year=max_year,
        max_age=max_age
    )
    simulation = Simulation(config)
    agent = Agent(
        sex=sex,
        age=age,
        year=min_year,
        year_index=0,
        family_history=simulation.family_history,
        antibiotic_exposure=simulation.antibiotic_exposure,
        province=simulation.province,
        ssp=simulation.SSP,
        has_asthma=has_asthma,
        asthma_age=asthma_age,
        asthma_status=asthma_status
    )
    simulation.update_asthma_effects(agent, outcome_matrix)
    np.testing.assert_array_equal(agent.control_levels.as_array(), expected_control_levels)
    assert agent.exacerbation_history.num_prev_year == expected_exacerbation_history.num_prev_year
    assert agent.exacerbation_history.num_current_year > expected_exacerbation_history.num_current_year
    pd.testing.assert_frame_equal(
        outcome_matrix.control.data,
        expected_outcome_matrix_control
    )
    pd.testing.assert_frame_equal(
        outcome_matrix.exacerbation.grouped_data.get_group((min_year)),
        expected_outcome_matrix_exacerbation,
        check_exact=False,
        rtol=0.15
    )
    pd.testing.assert_frame_equal(
        outcome_matrix.exacerbation_by_severity.grouped_data.get_group(
            (min_year, expected_outcome_matrix_exacerbation_history["severity"].iloc[0])
        ),
        expected_outcome_matrix_exacerbation_history,
        check_exact=False,
        rtol=0.2
    )


@pytest.mark.parametrize(
    (
        "min_year, time_horizon, province, population_growth_type, num_births_initial, max_age,"
        "year, n_new_agents, expected_immigrants"
    ),
    [
        (
            2024,
            3,
            "CA",
            "M3",
            10,
            111,
            2024,
            999,
            pd.Series([False] * 999, dtype=bool, name="immigrant")
        ),
        (
            2024,
            3,
            "CA",
            "M3",
            10,
            111,
            2025,
            21,
            pd.Series([True] * 10 + [False] * 11, dtype=bool, name="immigrant")
        )
    ]
)
def test_simulation_get_new_agents(
    config, min_year, time_horizon, province, population_growth_type, num_births_initial, max_age,
    year, n_new_agents, expected_immigrants
):

    config["simulation"] = {
        "min_year": min_year,
        "time_horizon": time_horizon,
        "province": province,
        "population_growth_type": population_growth_type,
        "num_births_initial": num_births_initial,
        "max_age": max_age
    }
    simulation = Simulation(config)
    new_agents_df = simulation.get_new_agents(year=year)
    assert new_agents_df.shape[0] == n_new_agents
    assert new_agents_df.shape[1] == time_horizon
    logger.info(new_agents_df.immigrant)
    pd.testing.assert_series_equal(
        new_agents_df.immigrant,
        expected_immigrants
    )



