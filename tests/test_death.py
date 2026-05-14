import pytest
import pathlib
import json
import datetime as dt
from leap.death import Death
from leap.agent import Agent
from leap.family_history import FamilyHistory
from leap.antibiotic_exposure import AntibioticExposure
from tests.utils import __test_dir__


@pytest.fixture(scope="function")
def config():
    with open(pathlib.Path(__test_dir__, "data/config.json"), "r") as file:
        config = json.load(file)
    return config


@pytest.mark.parametrize(
    "province, min_timepoint",
    [
        (
            "BC",
            dt.datetime(2024, 1, 1)
        ),
    ]
)
def test_death_constructor(config, province, min_timepoint):
    death = Death(province=province, min_timepoint=min_timepoint)
    assert death.life_table is not None


@pytest.mark.parametrize(
    "province, min_timepoint, timepoint, timepoint_index, sex, age, is_dead",
    [
        (
            "BC",
            dt.datetime(2024, 1, 1),
            dt.datetime(2024, 1, 1),
            0,
            True,
            110,
            True
        ),
        (
            "BC",
            dt.datetime(2024, 1, 1),
            dt.datetime(2025, 1, 1),
            1,
            True,
            7,
            False
        ),
    ]
)
def test_death_agent_dies(
    config, province, min_timepoint, timepoint, timepoint_index, sex, age, is_dead
):
    death = Death(province=province, min_timepoint=min_timepoint)
    agent = Agent(
        sex=sex,
        age=age,
        timepoint=timepoint,
        timepoint_index=timepoint_index,
        family_history=FamilyHistory(config=config["family_history"]),
        antibiotic_exposure=AntibioticExposure(config=config["antibiotic_exposure"]),
        province=province,
        ssp=config["pollution"]["SSP"]
    )
    assert death.agent_dies(agent) == is_dead
