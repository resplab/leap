import pytest
import pathlib
import json
import numpy as np
import datetime as dt
from leap.agent import Agent
from leap.antibiotic_exposure import AntibioticExposure
from leap.cost import AsthmaCost
from leap.control import ControlLevels
from leap.exacerbation import ExacerbationHistory
from leap.family_history import FamilyHistory
from tests.utils import __test_dir__


@pytest.fixture(scope="function")
def config():
    with open(pathlib.Path(__test_dir__, "data/config.json"), "r") as file:
        config = json.load(file)
    return config


@pytest.mark.parametrize(
    "parameters, exchange_rate_usd_cad, expected_parameters",
    [
        (
            {
            "control": [2372, 2965, 3127],
            "exac": [130, 594, 2425, 9900]
            },
            1.66,
            {
                "control": np.array([2372, 2965, 3127]) * 1.66,
                "exac": np.array([130, 594, 2425, 9900]) * 1.66
            }
        )
    ]
)
def test_asthma_cost_constructor(parameters, exchange_rate_usd_cad, expected_parameters):
    asthma_cost = AsthmaCost(
        exac=parameters["exac"],
        control_probs=parameters["control"],
        exchange_rate_usd_cad=exchange_rate_usd_cad
    )
    assert np.array_equal(asthma_cost.control_probs, expected_parameters["control"])
    assert np.array_equal(asthma_cost.exac, expected_parameters["exac"])


@pytest.mark.parametrize(
    (
        "parameters, exchange_rate_usd_cad, age, sex, timepoint, timepoint_index, province,"
        "control_levels, exacerbation_history, has_asthma, expected_cost"
    ),
    [
        (
            {
            "control": [2372, 2965, 3127],
            "exac": [130, 594, 2425, 9900]
            },
            1.66,
            20,
            "F",
            dt.datetime(2024, 1, 1),
            0,
            "BC",
            ControlLevels(0.2, 0.75, 0.05),
            ExacerbationHistory(num_current_timepoint=1, num_prev_timepoint=0),
            True,
            {">": 0.0}
        ),
        (
            {
            "control": [2372, 2965, 3127],
            "exac": [130, 594, 2425, 9900]
            },
            1.66,
            20,
            "F",
            dt.datetime(2024, 1, 1),
            0,
            "BC",
            ControlLevels(0.2, 0.75, 0.05),
            ExacerbationHistory(num_current_timepoint=1, num_prev_timepoint=0),
            False,
            {"=": 0.0}
        )
    ]
)
def test_compute_cost(
    config, parameters, exchange_rate_usd_cad, age, sex, timepoint, timepoint_index, province,
    control_levels, exacerbation_history, has_asthma, expected_cost
):
    agent = Agent(
        sex=sex,
        age=age,
        timepoint=timepoint,
        timepoint_index=timepoint_index,
        family_history=FamilyHistory(config=config["family_history"]),
        antibiotic_exposure=AntibioticExposure(config=config["antibiotic_exposure"]),
        exacerbation_history=exacerbation_history,
        province=province,
        ssp=config["pollution"]["SSP"],
        control_levels=control_levels,
        has_asthma=has_asthma
    )
    asthma_cost = AsthmaCost(
        control_probs=parameters["control"],
        exac=parameters["exac"],
        exchange_rate_usd_cad=exchange_rate_usd_cad
    )
    cost = asthma_cost.compute_cost(agent)
    if ">" in expected_cost.keys():
        assert cost > expected_cost[">"]
    elif "=" in expected_cost.keys():
        assert cost == expected_cost["="]


