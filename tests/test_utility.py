import pytest
import pathlib
import json
import numpy as np
from leap.utility import Utility
from leap.agent import Agent
from leap.family_history import FamilyHistory
from leap.antibiotic_exposure import AntibioticExposure
from leap.control import ControlLevels
from leap.severity import ExacerbationSeverityHistory
from tests.utils import __test_dir__


@pytest.fixture(scope="function")
def config():
    with open(pathlib.Path(__test_dir__, "data/config.json"), "r") as file:
        config = json.load(file)
    return config


@pytest.mark.parametrize(
    "parameters, age, sex, eq5d",
    [
        (
            {
                "βcontrol": [0.06, 0.09, 0.10],
                "βexac_sev_hist": [0.00615, 0.0169, 0.0192, 0.0215]
            },
            9,
            0,
            0.9405
        ),
    ]
)
def test_utility_constructor(parameters, age, sex, eq5d):
    utility = Utility(parameters=parameters)
    np.testing.assert_array_equal(utility.parameters["βcontrol"], parameters["βcontrol"])
    np.testing.assert_array_equal(utility.parameters["βexac_sev_hist"], parameters["βexac_sev_hist"])
    assert utility.table.get_group((age, sex))["eq5d"].values[0] == eq5d


@pytest.mark.parametrize(
    (
        "parameters, age, sex, year, year_index, province, has_asthma, asthma_age, control_levels,"
        "exacerbation_severity_history, value"
    ),
    [
        (
            {
                "βcontrol": [0.06, 0.09, 0.10],
                "βexac_sev_hist": [0.00615, 0.0169, 0.0192, 0.0215]
            },
            9,
            False,
            2024,
            0,
            "BC",
            False,
            None,
            None,
            ExacerbationSeverityHistory(np.zeros(4), np.zeros(4)),
            0.9405
        ),
        (
            {
                "βcontrol": [0.06, 0.09, 0.0405],
                "βexac_sev_hist": [0.0, 0.02, 0.0, 0.0]
            },
            9,
            False,
            2024,
            0,
            "BC",
            True,
            7,
            ControlLevels(fully_controlled=0.0, partially_controlled=0.0, uncontrolled=1.0),
            ExacerbationSeverityHistory(np.array([1, 5, 1, 0]), np.zeros(4)),
            0.8
        ),
    ]
)
def test_utility_compute_utility(
    config, parameters, age, sex, year, year_index, province, has_asthma, asthma_age,
    control_levels, exacerbation_severity_history, value
):
    utility = Utility(parameters=parameters)
    agent = Agent(
        sex=sex,
        age=age,
        year=year,
        year_index=year_index,
        family_history=FamilyHistory(config=config["family_history"]),
        antibiotic_exposure=AntibioticExposure(config=config["antibiotic_exposure"]),
        exacerbation_severity_history=exacerbation_severity_history,
        province=province,
        month=1,
        ssp=config["pollution"]["SSP"],
        has_asthma=has_asthma,
        asthma_age=asthma_age,
        control_levels=control_levels
    )
    assert utility.compute_utility(agent) == value
