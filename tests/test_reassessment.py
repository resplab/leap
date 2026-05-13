import pytest
import pathlib
import json
import datetime as dt
from leap.agent import Agent
from leap.family_history import FamilyHistory
from leap.antibiotic_exposure import AntibioticExposure
from leap.reassessment import Reassessment
from leap.utils import TimeDelta
from tests.utils import __test_dir__


@pytest.fixture(scope="function")
def config():
    with open(pathlib.Path(__test_dir__, "data/config.json"), "r") as file:
        config = json.load(file)
    return config


@pytest.mark.parametrize(
    "min_timepoint, province",
    [
        (
            dt.datetime(2020, 1, 1),
            "BC"
        ),
    ]
)
def test_reassessment_constructor(min_timepoint, province):
    reassessment = Reassessment(min_timepoint, province)
    df = reassessment.table.get_group((min_timepoint))
    assert df.iloc[0]["age"] == 4


@pytest.mark.parametrize(
    "min_timepoint, province, sex, age, timepoint, time_delta, has_asthma",
    [
        (
            dt.datetime(2020, 1, 1),
            "BC",
            True,
            53,
            dt.datetime(2024, 1, 1),
            TimeDelta(years=1),
            True
        ),
    ]
)
def test_reassessment_agent_has_asthma(
    config, min_timepoint, province, sex, age, timepoint, time_delta, has_asthma
):
    """
    By the ``asthma_reassessment.csv`` table, the probability of still having asthma
    is 1 for a male aged 53 in 2024 in BC.
    """
    reassessment = Reassessment(min_timepoint, province)
    timepoint_index = TimeDelta(td=timepoint - min_timepoint) // time_delta
    agent = Agent(
        sex=sex,
        age=age,
        timepoint=timepoint,
        timepoint_index=timepoint_index,
        family_history=FamilyHistory(config=config["family_history"]),
        antibiotic_exposure=AntibioticExposure(config=config["antibiotic_exposure"]),
        province=province,
        month=1,
        ssp=config["pollution"]["SSP"],
        has_asthma=True
    )
    assert agent.has_asthma is True
    assert reassessment.agent_has_asthma(agent) == has_asthma
