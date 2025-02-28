import pytest
import pathlib
import json
import numpy as np
import pandas as pd
from leap.agent import Agent
from leap.exacerbation import ExacerbationHistory
from leap.simulation import Simulation
from leap.outcome_matrix import OutcomeMatrix
from leap.logger import get_logger
from tests.utils import __test_dir__

logger = get_logger(__name__, 20)


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
            [0.3333, 0.3333, 0.3333]
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
    np.testing.assert_array_equal(agent.control_levels.as_array(), expected_control_levels)


@pytest.mark.parametrize(
    (
        "min_year, time_horizon, province, population_growth_type, num_births_initial, max_age,"
        "antibiotic_exposure_parameters, incidence_parameter_βfam_hist,"
        "incidence_parameter_βabx_exp, family_history_parameters, sex, age, year_index,"
        "expected_agent_has_asthma, expected_asthma_age, expected_asthma_status,"
        "expected_asthma_incidence"
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
            [1.826, -0.2920745, 0.053],
            {"p": 1.0},
            "F",
            4,
            0,
            True,
            4,
            True,
            1
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
            [-100, 0],
            [0.0, 0.0, 0.0],
            {"p": 1.0},
            "F",
            4,
            0,
            False,
            None,
            False,
            0
        )
    ]
)
def test_check_if_agent_gets_new_asthma_diagnosis(
    config, min_year, time_horizon, province, population_growth_type, num_births_initial, max_age,
    antibiotic_exposure_parameters, incidence_parameter_βfam_hist, incidence_parameter_βabx_exp,
    family_history_parameters, sex, age, year_index, expected_agent_has_asthma, expected_asthma_age,
    expected_asthma_status, expected_asthma_incidence
):
    """
    Setting the ``time_horizon`` to 1 means that the agents are generated from the initial
    population table, and that the agents are generated from the initial population table, and
    that no immigration happens.

    Setting the antibiotic exposure parameters below ensures that the antibiotic use is 0.

    Setting the ``num_births_initial = 10`` and starting in 2024 with growth type "M3", each of the
    age groups has 10 agents, for a total of 10 x 5 = 50 agents.

    Test 1:
    Setting the incidence parameter ``βfam_hist = [100, 0]`` and the family history parameter
    ``p = 1.0`` ensures that the probability of an agent being diagnosed with asthma is 1. The
    maximum age is set to 4, and the minimum age required for an asthma diagnosis is 3. So all
    agents aged 4 should receive an asthma diagnosis.

    Test 2:
    Setting the incidence parameter ``βfam_hist = [-100, 0]`` and the family history parameter
    ``p = 1.0`` ensures that the probability of an agent being diagnosed with asthma is 0.
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
    config["incidence"]["parameters"]["βfam_hist"] = incidence_parameter_βfam_hist
    config["incidence"]["parameters"]["βabx_exp"] = incidence_parameter_βabx_exp
    config["family_history"]["parameters"] = family_history_parameters

    outcome_matrix = OutcomeMatrix(
        until_all_die=False,
        min_year=min_year,
        max_year=min_year,
        max_age=max_age
    )

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

    simulation.check_if_agent_gets_new_asthma_diagnosis(agent, outcome_matrix)
    assert outcome_matrix.asthma_incidence.get(
        columns="n_new_diagnoses", year=min_year + year_index, age=age, sex=sex
    ) == expected_asthma_incidence
    assert outcome_matrix.asthma_status.get(
        columns="status", year=min_year + year_index, age=age, sex=sex
    ) == expected_asthma_status

    assert agent.has_asthma == expected_agent_has_asthma
    assert agent.asthma_age == expected_asthma_age
    assert agent.asthma_status == expected_asthma_status


@pytest.mark.parametrize(
    (
        "min_year, time_horizon, province, population_growth_type, num_births_initial, max_age,"
        "antibiotic_exposure_parameters, incidence_parameter_βfam_hist,"
        "family_history_parameters, exacerbation_hyperparameter_β0_μ, control_parameter_θ,"
        "sex, age, has_asthma, asthma_age, asthma_status, expected_control_levels,"
        "expected_exacerbation_history, expected_outcome_matrix_control,"
        "expected_outcome_matrix_exacerbation"
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
                    "prob": [0.0] * 28 + [1.0, 0.0]
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
        ),
    ]
)
def test_simulation_update_asthma_effects(
    config, min_year, time_horizon, province, population_growth_type, num_births_initial, max_age,
    antibiotic_exposure_parameters, incidence_parameter_βfam_hist, family_history_parameters,
    control_parameter_θ, exacerbation_hyperparameter_β0_μ, sex, age, has_asthma, asthma_age,
    asthma_status, expected_control_levels, expected_exacerbation_history,
    expected_outcome_matrix_control, expected_outcome_matrix_exacerbation,
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
    logger.info(outcome_matrix.control.data)
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


@pytest.mark.parametrize(
    (
        "min_year, time_horizon, province, population_growth_type, num_births_initial, max_age,"
        "antibiotic_exposure_parameters, incidence_parameter_βfam_hist,"
        "family_history_parameters, exacerbation_hyperparameter_β0_μ, control_parameter_θ,"
        "sex, age, has_asthma, asthma_age, asthma_status, exacerbation_history, year_index,"
        "expected_control_levels, expected_exacerbation_history"
    ),
    [
        (
            2024,
            1,
            "BC",
            "M3",
            10,
            100,
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
            10.0,
            [-1 * 10**5, -1 * 10**5],
            "M",
            53,
            True,
            4,
            True,
            ExacerbationHistory(20, 0),
            0,
            [0.0, 0.0, 1.0],
            ExacerbationHistory(100, 20)
        ),
    ]
)
def test_reassess_asthma_diagnosis(
    config, min_year, time_horizon, province, population_growth_type, num_births_initial, max_age,
    antibiotic_exposure_parameters, incidence_parameter_βfam_hist, family_history_parameters,
    exacerbation_hyperparameter_β0_μ, control_parameter_θ, sex, age, has_asthma, asthma_age,
    asthma_status, exacerbation_history, year_index, expected_control_levels,
    expected_exacerbation_history
):
    """
    Setting ``time_horizon=1`` means that the agents are generated from the initial population
    table, and that no immigration happens.

    Setting the antibiotic exposure parameters below ensures that the antibiotic use is 0.

    Setting the ``num_births_initial`` to 10 and starting in 2024 with growth type "M3", each of the
    age groups has 10 agents, for a total of 10 x 5 = 50 agents.

    Setting the incidence parameter ``βfam_hist=[100, 0]`` and the family history parameter ``p=1.0``
    ensures that the probability of an agent being diagnosed with asthma is 1. The maximum
    age is set to 4, and the minimum age required for an asthma diagnosis is 3. So all agents aged 4
    should receive an asthma diagnosis.

    Setting the control parameter ``θ=[-1e5, -1e5]`` ensures that the ``control_levels`` are:
        FC: 0.0
        PC: 0.0
        UC: 1.0

    Setting the exacerbation parameter ``β0_μ=5.0`` ensures that the number of exacerbations will
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
        year_index=year_index,
        family_history=simulation.family_history,
        antibiotic_exposure=simulation.antibiotic_exposure,
        province=simulation.province,
        ssp=simulation.SSP,
        has_asthma=has_asthma,
        asthma_age=asthma_age,
        asthma_status=asthma_status,
        exacerbation_history=exacerbation_history
    )

    simulation.reassess_asthma_diagnosis(agent, outcome_matrix)
    assert agent.has_asthma == has_asthma
    assert agent.asthma_age == asthma_age
    assert agent.asthma_status == asthma_status
    np.testing.assert_array_equal(agent.control_levels.as_array(), expected_control_levels)
    logger.info(outcome_matrix.control.data)
    assert outcome_matrix.control.get(
        columns="prob", year=min_year + year_index, age=age, sex=sex, level=0
    ) == expected_control_levels[0]
    assert outcome_matrix.control.get(
        columns="prob", year=min_year + year_index, age=age, sex=sex, level=1
    ) == expected_control_levels[1]
    assert outcome_matrix.control.get(
        columns="prob", year=min_year + year_index, age=age, sex=sex, level=2
    ) == expected_control_levels[2]
    assert agent.exacerbation_history.num_prev_year == expected_exacerbation_history.num_prev_year
    assert agent.exacerbation_history.num_current_year > expected_exacerbation_history.num_current_year
    assert outcome_matrix.exacerbation.get(
        columns="n_exacerbations", year=min_year + year_index, age=age, sex=sex
    ) > expected_exacerbation_history.num_current_year


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


@pytest.mark.parametrize(
    (
        "min_year, time_horizon, province, population_growth_type, num_births_initial, max_age,"
        "antibiotic_exposure_parameters, incidence_parameter_βfam_hist,"
        "family_history_parameters, exacerbation_hyperparameter_β0_μ, control_parameter_θ,"
        "death_parameters, prevalence_parameters, utility_parameters, cost_parameters,"
        "year_index, expected_asthma_incidence_total, expected_asthma_status_total,"
        "expected_asthma_cost, expected_death, expected_emigration, expected_exacerbation_total,"
        "expected_family_history, expected_immigration, expected_utility"

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
            {
                "β0": -1,
                "β1": -1,
                "β2": -1
            },
            {
                "β0": -20,
                "βsex": -20,
                "βage": [0.0, 0.0, 0.0, 0.0, 0.0],
                "βyear": [0.0, 0.0],
                "βsexage": [0.0, 0.0, 0.0, 0.0, 0.0],
                "βsexyear": [0.0, 0.0],
                "βyearage": [0.0] * 10,
                "βsexyearage": [0.0] * 10,
                "βfam_hist": [-100, 0],
                "βabx_exp": [0.0, 0.0, 0.0]
            },
            {
                "βcontrol": [0.0, 0.0, 0.10],
                "βexac_sev_hist": [0.0, 0.0, 0.0, 0.0]
            },
            {
                "control": [0.0, 0.0, 100.0],
                "exac": [0.0, 0.0, 0.0, 0.0]
            },
            0,
            10,
            10,
            pd.DataFrame(
                data={
                    "year": [2024] * 10,
                    "age": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
                    "sex": ["F", "M"] * 5,
                    "cost": [0.0] * 8 + [664, 996]
                }
            ),
            pd.DataFrame(
                data={
                    "year": [2024] * 10,
                    "age": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
                    "sex": ["F", "M"] * 5,
                    "n_deaths": [0] * 10
                }
            ),
            pd.DataFrame(
                data={
                    "year": [2024] * 10,
                    "age": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
                    "sex": ["F", "M"] * 5,
                    "n_emigrants": [0] * 10
                }
            ),
            1000,
            pd.DataFrame(
                data={
                    "year": [2024] * 10,
                    "age": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
                    "sex": ["F", "M"] * 5,
                    "has_family_history": [2, 8, 4, 6, 4, 6, 3, 7, 6, 4]
                }
            ),
            pd.DataFrame(
                data={
                    "year": [2024] * 10,
                    "age": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
                    "sex": ["F", "M"] * 5,
                    "n_immigrants": [0] * 10
                }
            ),
            pd.DataFrame(
                data={
                    "year": [2024] * 10,
                    "age": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
                    "sex": ["F", "M"] * 5,
                    "utility": [2.0, 8.0, 3.97, 5.93, 5.96, 3.94, 5.88, 3.92, 3.47, 5.23]
                }
            ),
        )
    ]
)
def test_run_simulation_one_year(
    config, min_year, time_horizon, province, population_growth_type, num_births_initial, max_age,
    antibiotic_exposure_parameters, incidence_parameter_βfam_hist, family_history_parameters,
    exacerbation_hyperparameter_β0_μ, control_parameter_θ, death_parameters,
    prevalence_parameters, utility_parameters, cost_parameters, year_index,
    expected_asthma_incidence_total, expected_asthma_status_total, expected_asthma_cost,
    expected_death, expected_emigration, expected_exacerbation_total, expected_family_history,
    expected_immigration, expected_utility
):
    """
    Setting ``num_births_initial = 10`` and starting in 2024 with growth type "M3", each of the
    age groups has 10 agents, for a total of 10 x 5 = 50 agents.

    Setting the antibiotic exposure parameters below ensures that the antibiotic use is 0.

    Setting the control parameter ``θ = [-1e5, -1e5]`` ensures that the ``control_levels`` are:
        FC: 0.0
        PC: 0.0
        UC: 1.0

    Setting the cost parameters to:
        {
            "control": [0.0, 0.0, 100.0],
            "exac": [0.0, 0.0, 0.0, 0.0]
        }

    together with the control parameter ``θ = [-1e5, -1e5]`` ensures that the cost for each agent
    diagnosed with asthma is 100.

    For the year 2024, province "CA", growth type "M3", ages 0 - 4, the probability of emigration
    is 0. See ``processed_data/migration/emigration_table.csv``.

    Setting the exacerbation hyperparameter ``β0_μ = 20.0`` ensures that every agent aged 4 has an
    asthma exacerbation.

    Setting the ``family_history`` parameter ``p = 1.0`` ensures that every agent has a
    family history of asthma.

    Setting ``time_horizon=1`` means that the agents are generated from the initial population
    table, and that no immigration happens.

    Setting the incidence parameter ``βfam_hist=[100, 0]`` and the family history parameter
    ``p=1.0`` ensures that the probability of an agent being diagnosed with asthma is 1. The
    maximum age is set to 4, and the minimum age required for an asthma diagnosis is 3. So all
    agents aged 4 should receive an asthma diagnosis.

    Setting the ``prevalence`` parameters below ensures that the prevalence is 0.

    Setting the ``utility`` parameters to:
        {
            "βcontrol": [0.0, 0.0, 0.10],
            "βexac_sev_hist": [0.0, 0.0, 0.0, 0.0]
        }

    ensures that the utility for each agent with asthma is either 0.87222 (male) or 0.87356 (female).
    For each agent without asthma, it is just the baseline from the EQ-5D table.
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
    config["incidence"]["parameters"]["βfam_hist"] = incidence_parameter_βfam_hist
    config["family_history"]["parameters"] = family_history_parameters
    config["exacerbation"]["hyperparameters"]["β0_μ"] = exacerbation_hyperparameter_β0_μ
    config["control"]["parameters"]["θ"] = control_parameter_θ
    config["death"]["parameters"] = death_parameters
    config["prevalence"]["parameters"] = prevalence_parameters
    config["utility"]["parameters"] = utility_parameters
    config["cost"]["parameters"] = cost_parameters

    simulation = Simulation(config)
    outcome_matrix = simulation.run(
        seed=1,
        until_all_die=False
    )
    logger.info(outcome_matrix.utility.data)
    assert outcome_matrix.antibiotic_exposure.data.shape == (2 * 1 * (max_age + 1), 4)
    np.testing.assert_array_equal(
        outcome_matrix.antibiotic_exposure.data["n_antibiotic_exposure"],
        [0] * 2 * 1 * (max_age + 1)
    )
    assert outcome_matrix.asthma_incidence.data.shape == (2 * 1 * (max_age + 1), 4)
    assert outcome_matrix.asthma_incidence.data["n_new_diagnoses"].sum() \
        == expected_asthma_incidence_total
    assert outcome_matrix.asthma_status.data.shape == (2 * 1 * (max_age + 1), 4)
    assert outcome_matrix.asthma_status.data["status"].sum() == expected_asthma_status_total

    pd.testing.assert_frame_equal(
        outcome_matrix.cost.data,
        expected_asthma_cost,
        rtol=0.15
    )
    assert outcome_matrix.cost.data["cost"].sum() == expected_asthma_cost["cost"].sum()

    pd.testing.assert_frame_equal(
        outcome_matrix.death.data,
        expected_death
    )

    pd.testing.assert_frame_equal(
        outcome_matrix.emigration.data,
        expected_emigration
    )

    assert outcome_matrix.exacerbation.data.shape == (2 * 1 * (max_age + 1), 4)
    assert outcome_matrix.exacerbation.data["n_exacerbations"].sum() > expected_exacerbation_total

    pd.testing.assert_frame_equal(
        outcome_matrix.immigration.data,
        expected_immigration
    )

    assert outcome_matrix.family_history.data.shape == (2 * 1 * (max_age + 1), 4)
    for age in range(0, max_age + 1):
        # Test family history status total for a given age
        has_family_history_age = outcome_matrix.family_history.get(
            columns="has_family_history", age=age
        ).sum()
        expected_has_family_history_age = expected_family_history.loc[
            expected_family_history["age"] == age]["has_family_history"].sum()
        assert has_family_history_age == expected_has_family_history_age
    
        # Test utility total for a given age
        utility_age = round(
            outcome_matrix.utility.get(columns="utility", age=age).sum(),
            ndigits=1
        )
        expected_utility_age = round(
            expected_utility.loc[expected_utility["age"] == age]["utility"].sum(),
            ndigits=1
        )
        assert utility_age == expected_utility_age


@pytest.mark.parametrize(
    (
        "min_year, time_horizon, province, population_growth_type, num_births_initial, max_age,"
        "antibiotic_exposure_parameters, incidence_parameter_βfam_hist,"
        "family_history_parameters, exacerbation_hyperparameter_β0_μ, control_parameter_θ,"
        "death_parameters, prevalence_parameters, cost_parameters,"
        "expected_alive, expected_antibiotic_exposure,"
        "expected_asthma_cost, expected_death, expected_emigration, expected_exacerbation_total,"
        "expected_family_history, expected_immigration_total"

    ),
    [
        (
            2024,
            2,
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
            {
                "β0": -1,
                "β1": -1,
                "β2": -1
            },
            {
                "β0": -20,
                "βsex": -20,
                "βage": [0.0, 0.0, 0.0, 0.0, 0.0],
                "βyear": [0.0, 0.0],
                "βsexage": [0.0, 0.0, 0.0, 0.0, 0.0],
                "βsexyear": [0.0, 0.0],
                "βyearage": [0.0] * 10,
                "βsexyearage": [0.0] * 10,
                "βfam_hist": [-100, 0],
                "βabx_exp": [0.0, 0.0, 0.0]
            },
            {
                "control": [0.0, 0.0, 100.0],
                "exac": [0.0, 0.0, 0.0, 0.0]
            },
            pd.DataFrame(
                data={
                    "year": [2024] * 10 + [2025] * 10,
                    "age": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4] * 2,
                    "sex": ["F", "M"] * 10,
                    "n_alive": [6, 4] * 5 + [4, 7] + [2, 8] * 3 + [3, 8]
                }
            ),
            pd.DataFrame(
                data={
                    "year": [2024] * 10 + [2025] * 10,
                    "age": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4] * 2,
                    "sex": ["F", "M"] * 10,
                    "n_antibiotic_exposure": [0] * 20
                }
            ),
            pd.DataFrame(
                data={
                    "year": [2024] * 10 + [2025] * 10,
                    "age": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4] * 2,
                    "sex": ["F", "M"] * 10,
                    "cost": [0.0] * 8 + [664, 996] + [0.0] * 8 + [724, 1096]
                }
            ),
            pd.DataFrame(
                data={
                    "year": [2024] * 10 + [2025] * 10,
                    "age": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4] * 2,
                    "sex": ["F", "M"] * 10,
                    "n_deaths": [0] * 20
                }
            ),
            pd.DataFrame(
                data={
                    "year": [2024] * 10 + [2025] * 10,
                    "age": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4] * 2,
                    "sex": ["F", "M"] * 10,
                    "n_emigrants": [0] * 20
                }
            ),
            1000,
            pd.DataFrame(
                data={
                    "year": [2024] * 10,
                    "age": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
                    "sex": ["F", "M"] * 5,
                    "has_family_history": [2, 8, 4, 6, 4, 6, 3, 7, 6, 4]
                }
            ),
            1,
        )
    ]
)
def test_run_simulation_two_years(
    config, min_year, time_horizon, province, population_growth_type, num_births_initial, max_age,
    antibiotic_exposure_parameters, incidence_parameter_βfam_hist, family_history_parameters,
    exacerbation_hyperparameter_β0_μ, control_parameter_θ, death_parameters, prevalence_parameters,
    cost_parameters, expected_alive, expected_antibiotic_exposure, expected_asthma_cost,
    expected_death, expected_emigration, expected_exacerbation_total, expected_family_history,
    expected_immigration_total
):
    """
    Setting the incidence parameter ``βfam_hist=[100, 0]`` and the family history parameter
    ``p=1.0`` ensures that the probability of an agent being diagnosed with asthma is 1. The
    maximum age is set to 4, and the minimum age required for an asthma diagnosis is 3. So all
    agents aged 4 should receive an asthma diagnosis.

    Setting the antibiotic exposure parameters below ensures that the antibiotic use is 0.

    Setting ``num_births_initial=10`` and starting in 2024 with growth type "M3", each of the
    age groups has 10 agents, for a total of 10 x 5 = 50 agents.

    Setting the control parameter ``θ = [-1e5, -1e5]`` ensures that the ``control_levels`` are:
        FC: 0.0
        PC: 0.0
        UC: 1.0

    Setting the exacerbation hyperparameter ``β0_μ = 20.0`` ensures that every agent aged 4 has an
    asthma exacerbation.

    Setting the ``time_horizon=2`` means that in the first year there should be 0 immigrants,
    and in the second year there should be 1 immigrant.

    For the years 2024 and 2025, province "CA", growth type "M3", ages 0 - 4, the probability of
    emigration is 0. See ``processed_data/migration/emigration_table.csv``.

    Setting the ``prevalence`` parameters below ensures that the prevalence is 0.
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
    config["incidence"]["parameters"]["βfam_hist"] = incidence_parameter_βfam_hist
    config["family_history"]["parameters"] = family_history_parameters
    config["exacerbation"]["hyperparameters"]["β0_μ"] = exacerbation_hyperparameter_β0_μ
    config["control"]["parameters"]["θ"] = control_parameter_θ
    config["death"]["parameters"] = death_parameters
    config["prevalence"]["parameters"] = prevalence_parameters
    config["cost"]["parameters"] = cost_parameters

    simulation = Simulation(config)
    outcome_matrix = simulation.run(
        seed=1,
        until_all_die=False
    )

    for year, age in zip(range(min_year, min_year + time_horizon), range(max_age + 1)):
        assert outcome_matrix.alive.get(
            columns="n_alive", year=year, age=age
        ).sum() == expected_alive.loc[
            (expected_alive["age"] == age) & (expected_alive["year"] == year)
        ]["n_alive"].sum()

    pd.testing.assert_frame_equal(
        outcome_matrix.antibiotic_exposure.data,
        expected_antibiotic_exposure
    )
    pd.testing.assert_frame_equal(
        outcome_matrix.death.data,
        expected_death
    )
    pd.testing.assert_frame_equal(
        outcome_matrix.emigration.data,
        expected_emigration
    )
    pd.testing.assert_frame_equal(
        outcome_matrix.immigration.get(
            columns=["year", "sex", "n_immigrants"], year=min_year
        ),
        pd.DataFrame(
            data={
                "year": [min_year] * (max_age + 1) * 2,
                "sex": ["F", "M"] * (max_age + 1),
                "n_immigrants": [0] * (max_age + 1) * 2
            }
        )
    )

    assert int(outcome_matrix.immigration.get(
        columns=["n_immigrants"]
    ).sum()) == expected_immigration_total

    assert outcome_matrix.family_history.data.shape == (2 * time_horizon * (max_age + 1), 4)
    for age in range(0, max_age + 1):
        assert outcome_matrix.family_history.get(
            columns="has_family_history", age=age, year=min_year
        ).sum() == expected_family_history.loc[
            (expected_family_history["age"] == age) &
            (expected_family_history["year"] == min_year)
        ]["has_family_history"].sum()
        assert outcome_matrix.cost.get(
            columns="cost", age=age, year=min_year
        ).sum() == expected_asthma_cost.loc[
            (expected_asthma_cost["age"] == age) &
            (expected_asthma_cost["year"] == min_year)
        ]["cost"].sum()

    assert outcome_matrix.exacerbation.data.shape == (2 * time_horizon * (max_age + 1), 4)
    assert outcome_matrix.exacerbation.data["n_exacerbations"].sum() > expected_exacerbation_total
    # assert outcome_matrix.cost.get(
    #     columns="cost", year=min_year + 1, age=max_age
    # ).sum() <= expected_asthma_cost.loc[
    #     (expected_asthma_cost["age"] == max_age) &
    #     (expected_asthma_cost["year"] == min_year + 1)
    # ]["cost"].sum()
    # assert outcome_matrix.cost.get(
    #     columns="cost", year=min_year + 1, age=max_age
    # ).sum() >= expected_asthma_cost.loc[
    #     (expected_asthma_cost["age"] == max_age) &
    #     (expected_asthma_cost["year"] == min_year)
    # ]["cost"].sum()


@pytest.mark.parametrize(
    (
        "min_year, time_horizon, province, population_growth_type, num_births_initial, max_age,"
    ),
    [
        (
            2024,
            3,
            "CA",
            "M3",
            10,
            111,
        )
    ]
)
def test_run_simulation_full(
    config, min_year, time_horizon, province, population_growth_type, num_births_initial, max_age
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
    outcome_matrix = simulation.run(
        seed=1,
        until_all_die=False
    )

    assert outcome_matrix.immigration.data.shape == (2 * time_horizon * (max_age + 1), 4)
    pd.testing.assert_frame_equal(
        outcome_matrix.immigration.get(columns=["year", "sex", "n_immigrants"], year=2024),
        pd.DataFrame(
            data={
                "year": [2024] * (max_age + 1) * 2,
                "sex": ["F", "M"] * (max_age + 1),
                "n_immigrants": [0] * (max_age + 1) * 2
            }
        )
    )
