import pytest
import pathlib
import json
from leap.agent import Agent
from leap.family_history import FamilyHistory
from leap.antibiotic_exposure import AntibioticExposure
from leap.reassessment import Reassessment
from tests.utils import __test_dir__


@pytest.fixture(scope="function")
def config():
    with open(pathlib.Path(__test_dir__, "data/config.json"), "r") as file:
        config = json.load(file)
    return config


@pytest.mark.parametrize(
    "starting_year, province",
    [
        (
            2020,
            "BC"
        ),
    ]
)
def test_reassessment_constructor(starting_year, province):
    reassessment = Reassessment(starting_year, province)
    df = reassessment.table.get_group((starting_year))
    assert df.iloc[0]["age"] == 4


@pytest.mark.parametrize(
    "starting_year, province, sex, age, year, has_asthma",
    [
        (
            2020,
            "BC",
            True,
            53,
            2024,
            True
        ),
    ]
)
def test_reassessment_agent_has_asthma(config, starting_year, province, sex, age, year, has_asthma):
    """
    By the `master_asthma_reassessment.csv` table, the probability of still having asthma
    is 1 for a male aged 53 in 2024 in BC.
    """
    reassessment = Reassessment(starting_year, province)
    year_index = year - starting_year
    agent = Agent(
        sex=sex,
        age=age,
        year=year,
        year_index=year_index,
        family_history=FamilyHistory(config=config["family_history"]),
        antibiotic_exposure=AntibioticExposure(config=config["antibiotic_exposure"]),
        province=province,
        month=1,
        ssp=config["pollution"]["SSP"],
        has_asthma=True
    )
    assert agent.has_asthma is True
    assert reassessment.agent_has_asthma(agent) == has_asthma
