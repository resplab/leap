# Data Generation

In this module, we generate the datasets used for the `LEAP` model. This only needs to be
run once, and in general, users may ignore this folder. It is mainly for developers to reference
and update if new data needs to be added.

## Birth Data

To obtain the population data for each year, we used two tables from `StatCan`:

1. 1999 - 2021

For past years, we used 
[Table 17-10-00005-01 from StatCan](https://www150.statcan.gc.ca/t1/tbl1/en/cv.action?pid=1710000501).

The `*.csv` file can be downloaded from here:
[17100005-eng.zip](https://www150.statcan.gc.ca/n1/tbl/csv/17100005-eng.zip)

and is saved as:
`LEAP/leap/original_data/17100005.csv`

2. 2021 - 2065:

For future years, we used
[Table 17-10-0057-01 from StatCan](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1710005701).

The `*.csv` file can be downloaded from here:
[17100057-eng.zip](https://www150.statcan.gc.ca/n1/tbl/csv/17100057-eng.zip).

and is saved as:
`LEAP/leap/original_data/17100057.csv`


To run the data processing for the population data:

```sh
cd LEAP
python3 leap/data_generation/birth_data.py
```

This will update the following data files: 

1. `leap/processed_data/master_birth_estimate.csv`
2. `leap/processed_data/master_initial_pop_distribution_prop.csv`

