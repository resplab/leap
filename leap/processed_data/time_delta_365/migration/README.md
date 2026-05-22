# Immigration and Emigration in Canada

## Population Data

The data for the population was generated from the StatsCan website. For the years 2001-2019, we
used Table 17-10-0005-01, and for the years 2020-2068, we used Table 17-10-0057-01:

2001-2019: https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1710000501
2020-2068: https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1710005701

Note: The projection table goes to 2068 for Canada (CA), but only to 2043 for British Columbia
(BC). No other provinces are currently supported.

Since StatsCan does not provide a breakdown of population projections/estimates by emigration
or immigration, we calculated the net migration for each age, sex, timepoint, province, and
projection scenario from the population and mortality data. See
`leap/data_generation/migration_data.py` for details, and `docs/model/model-migration.rst` for
the mathematical description.

## Migration Table

Both immigration and emigration data are stored in a single file:

`migration_table.csv`

Each row corresponds to a unique combination of `timepoint`, `province`, `age`, `sex`, and
`projection_scenario`. The columns are:

| column | type | description |
|--------|------|-------------|
| `timepoint` | `datetime` | The starting date / time for the entry. Data applies to the interval `[timepoint, timepoint + time_delta]`. Range is 2001-2068 for CA and 2001-2043 for BC. |
| `province` | `str` | Province abbreviation. One of `CA` (all of Canada) or `BC` (British Columbia). |
| `age` | `int` | Age in years, in the range 1-100. Age 0 (newborns) is excluded because births are handled separately by the birth model. |
| `sex` | `str` | `"F"` = female, `"M"` = male. |
| `projection_scenario` | `str` | Population growth type, one of: `LG` (low growth), `HG` (high growth), `M1`-`M5` (medium growth variants), `FA` (fast aging), `SA` (slow aging). Note: `M6` is a placeholder scenario present in the StatCan data but not yet implemented. See [StatCan Projection Scenarios](https://www150.statcan.gc.ca/n1/pub/91-520-x/91-520-x2022001-eng.htm). |
| `delta_n` | `float` | The signed net migration for this age, sex, timepoint, province, and projection scenario. Positive values indicate net immigration; negative values indicate net emigration. |
| `prop_migrants_birth` | `float` | `delta_n` divided by the total number of births in that timepoint, province, and projection scenario. Signed — positive for immigration cells, negative for emigration cells. Used to compute the total number of immigrant agents to create each timepoint, scaled to the simulation's birth count. |
| `prop_immigrants_year` | `float` | For cells where `delta_n > 0`, the proportion of immigrants for this age and sex relative to the total number of immigrants in that timepoint. Zero for emigration cells. Sums to 1.0 across all immigration cells for a given timepoint, province, and projection scenario. Used to sample the age and sex of new immigrant agents. |
| `prop_emigrants_year` | `float` | For cells where `delta_n < 0`, the proportion of emigrants for this age and sex relative to the total number of emigrants in that timepoint. Zero for immigration cells. Sums to 1.0 across all emigration cells for a given timepoint, province, and projection scenario. |
| `prob_emigration` | `float` | For cells where `delta_n < 0`, the per-person probability of emigrating during this timepoint, computed as `\|delta_n\| / N` where `N` is the population for that age, sex, province, and projection scenario. Zero for immigration cells. Applied as a Bernoulli trial to each existing agent each timepoint. |
