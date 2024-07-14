import pytest
import json
import pathlib
from leap.agent import Agent
from leap.antibiotic_exposure import AntibioticExposure
from leap.family_history import FamilyHistory
from tests.utils import __test_dir__


@pytest.fixture(scope="function")
def config():
    with open(pathlib.Path(__test_dir__, "data/config.json"), "r") as file:
        config = json.load(file)
    return config


@pytest.mark.parametrize(
    (
        "sex, age, year, year_index, antibiotic_exposure_parameters, family_history_parameters,"
        "num_antibiotic_use, has_family_history"
    ),
    [
        (
            False,
            23,
            2024,
            1,
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
                "p": 0.0
            },
            0,
            False
        ),
        (
            False,
            23,
            2024,
            1,
            None,
            None,
            None,
            None
        )
    ]
)
def test_agent_constructor(
    config, sex, age, year, year_index, antibiotic_exposure_parameters, family_history_parameters,
    num_antibiotic_use, has_family_history
):

    if antibiotic_exposure_parameters is None:
        antibiotic_exposure = None
    else:
        config["antibiotic_exposure"]["parameters"] = antibiotic_exposure_parameters
        antibiotic_exposure = AntibioticExposure(config["antibiotic_exposure"])

    if family_history_parameters is None:
        family_history = None
    else:
        config["family_history"]["parameters"]["p"] = family_history_parameters["p"]
        family_history = FamilyHistory(config["family_history"])

    agent = Agent(
        sex=sex, age=age, year=year, year_index=year_index, antibiotic_exposure=antibiotic_exposure,
        family_history=family_history
    )

    assert agent.sex == sex
    assert agent.age == age
    assert agent.year == year
    assert agent.year_index == year_index
    assert agent.has_family_history == has_family_history
    if antibiotic_exposure is not None:
        assert round(agent.num_antibiotic_use, 1) == num_antibiotic_use
