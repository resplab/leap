# Immigration and Emigration in Canada

## Population Data

The data for the population was generated from the StatsCan website. For the years 2001-2019, we
used Table 17-10-0005-01, and for the years 2020-2065, we used Table 17-10-0057-01:

2001-2019: https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1710000501
2020-2065: https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1710005701

Note: For the latter table, the projection goes to 2065 for Canada, but only to 2043 for the
provinces.

Since StatsCan does not provide a breakdown of population projections/estimates by emigration
or immigration, we calculated the average number of individuals required to emigrate/immigrate to
match the projected demographics. These results are saved in:

`emigration_table.csv`
`immigration_table.csv`

### Emigration Table

The emigration table has the following columns:

`year`: Integer year in the range 2001-2065.
`age`: Integer age.
`sex`: String, "F" = female, "M" = male.
`province`: A string indicating the province abbreviation, e.g. "BC". For all of Canada,
  set province to "CA".
`n_emigrants`: The number of immigrants for a given year, sex, age, province, and projection
    scenario.
`prop_emigrants_birth`: The proportion of emigrants for a given age and sex relative to the total
  number of births for a given year and projection scenario. To compute the number of emigrants
  for a given year, projection scenario, age, and sex, multiply the number of births by
  `prop_emigrants_birth`.
`prop_emigrants_year`: The proportion of emigrants for a given age and sex relative to the total
  number of emigrants for a given year and projection scenario. To compute the number of emigrants
  for a given year, projection scenario, age, and sex, multiply the total number of emigrants for
  that year and projection scenario by `prop_emigrants_year`.
`projection_scenario`: Population growth type, one of:
  ["past", "LG", "HG", "M1", "M2", "M3", "M4", "M5", "M6", FA", "SA"].
  See [Stats Canada](https://www150.statcan.gc.ca/n1/pub/91-520-x/91-520-x2022001-eng.htm).


### Immigration Table

The immigration table has the following columns:

`year`: Integer year in the range 2001-2065.
`age`: Integer age.
`sex`: String, "F" = female, "M" = male.
`province`: A string indicating the province abbreviation, e.g. "BC". For all of Canada,
  set province to "CA".
`n_immigrants`: The number of immigrants for a given year, sex, age, province, and projection
    scenario.
`prop_immigrants_birth`: The proportion of immigrants for a given age and sex relative to the total
  number of births for a given year and projection scenario. To compute the number of immigrants
  for a given year, projection scenario, age, and sex, multiply the number of births by
  `prop_immigrants_birth`.
`prop_immigrants_year`: The proportion of immigrants for a given age and sex relative to the total
  number of immigrants for a given year and projection scenario. To compute the number of immigrants
  for a given year, projection scenario, age, and sex, multiply the total number of immigrants for
  that year and projection scenario by `prop_immigrants_year`.
`projection_scenario`: Population growth type, one of:
  ["past", "LG", "HG", "M1", "M2", "M3", "M4", "M5", "M6", FA", "SA"].
  See [Stats Canada](https://www150.statcan.gc.ca/n1/pub/91-520-x/91-520-x2022001-eng.htm).
