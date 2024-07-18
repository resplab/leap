import pytest
from leap.pollution import PollutionTable, Pollution


def test_pollution_table_constructor():
    pollution_table = PollutionTable()
    grouped_df = pollution_table.data
    assert grouped_df.keys == "SSP"
    assert list(grouped_df.groups.keys()) == ["SSP1_2.6", "SSP2_4.5", "SSP3_7.0", "SSP5_8.5"]
    assert grouped_df.get_group("SSP1_2.6").iloc[0]["month"] == 1
    assert grouped_df.get_group("SSP1_2.6").iloc[9]["CDUID"] == 1001


@pytest.mark.parametrize(
    "cduid, year, month, ssp, wildfire_pm25_scaled, total_pm25",
    [
        (5915, 2028, 2, "SSP2_4.5", 0.09851602637, 3.168516026)
    ]
)
def test_pollution_constructor(cduid, year, month, ssp, wildfire_pm25_scaled, total_pm25):
    pollution = Pollution(cduid, year, month, ssp)
    assert pollution.cduid == cduid
    assert pollution.year == year
    assert pollution.month == month
    assert pollution.SSP == ssp
    assert pollution.wildfire_pm25_scaled == wildfire_pm25_scaled
    assert pollution.total_pm25 == total_pm25
