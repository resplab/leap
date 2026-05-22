# Archived Migration Data (Annual)

This directory contains archived annual immigration and emigration tables from the previous
two-file schema. These files are superseded by `migration_table.csv` in the parent
`migration/` directory, which consolidates both into a single file.

## Emigration Table (`emigration_table.csv`)

| column | type | description |
|--------|------|-------------|
| `timepoint` | `datetime` | The starting date / time for the entry. Range 2001-2065. |
| `age` | `int` | Age in years. |
| `sex` | `str` | `"F"` = female, `"M"` = male. |
| `province` | `str` | Two-letter province abbreviation, e.g. `"BC"`. For all of Canada, use `"CA"`. |
| `n_emigrants` | `int` | The number of emigrants for a given timepoint, sex, age, province, and projection scenario. |
| `prop_emigrants_birth` | `float` | The proportion of emigrants relative to the number of births for a given timepoint and projection scenario. |
| `prop_emigrants_timepoint` | `float` | The proportion of emigrants for a given age and sex relative to the total number of emigrants for a given timepoint and projection scenario. |
| `projection_scenario` | `str` | Population growth type. See [StatCan Projection Scenarios](https://www150.statcan.gc.ca/n1/pub/91-520-x/91-520-x2022001-eng.htm). |

## Immigration Table (`immigration_table.csv`)

| column | type | description |
|--------|------|-------------|
| `timepoint` | `datetime` | The starting date / time for the entry. Range 2001-2065. |
| `age` | `int` | Age in years. |
| `sex` | `str` | `"F"` = female, `"M"` = male. |
| `province` | `str` | Two-letter province abbreviation, e.g. `"BC"`. For all of Canada, use `"CA"`. |
| `n_immigrants` | `int` | The number of immigrants for a given timepoint, sex, age, province, and projection scenario. |
| `prop_immigrants_birth` | `float` | The proportion of immigrants relative to the number of births for a given timepoint and projection scenario. |
| `prop_immigrants_timepoint` | `float` | The proportion of immigrants for a given age and sex relative to the total number of immigrants for a given timepoint and projection scenario. |
| `projection_scenario` | `str` | Population growth type. See [StatCan Projection Scenarios](https://www150.statcan.gc.ca/n1/pub/91-520-x/91-520-x2022001-eng.htm). |
