import pytest
import pathlib
import json
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
    "province, starting_year",
    [
        (
            "BC",
            2024
        ),
    ]
)
def test_death_constructor(config, province, starting_year):
    death = Death(province=province, starting_year=starting_year)
    assert death.life_table is not None


@pytest.mark.parametrize(
    "province, starting_year, year, year_index, sex, age, is_dead",
    [
        (
            "BC",
            2024,
            2024,
            0,
            True,
            110,
            True
        ),
        (
            "BC",
            2024,
            2025,
            1,
            True,
            7,
            False
        ),
    ]
)
def test_death_agent_dies(
    config, province, starting_year, year, year_index, sex, age, is_dead
):
    death = Death(province=province, starting_year=starting_year)
    agent = Agent(
        sex=sex,
        age=age,
        year=year,
        year_index=year_index,
        family_history=FamilyHistory(config=config["family_history"]),
        antibiotic_exposure=AntibioticExposure(config=config["antibiotic_exposure"]),
        province=province,
        ssp=config["pollution"]["SSP"]
    )
    assert death.agent_dies(agent) == is_dead
