# Canadian Census Divisions

## Population Data

The data for the census divisions was generated from the StatsCan website:

https://www150.statcan.gc.ca/t1/tbl1/en/cv.action?pid=9810000701

Under `Population and dwelling counts`, the following columns were selected:

1. Population, 2021
2. Land area in square kilometres, 2021
3. Population density per square kilometre, 2021

Under `Geographic name`, under `Select specific levels only`, only the third checkbox was ticked
so as to only take data for Canadian census divisions and not include the total for the provinces
or the country.

To download, two files were selected:

1. `CSV: Download as displayed (excluding accompanying symbols)`
2. `CSV: Download selected data (for database loading)`

The database loading data includes the `DGUID`, which was needed to generate the `CDUID` column.
This data was reformatted into the `master_census_data_2021.csv` file, with the following columns:

1. `year`: the year that the census data was collected.
2. `census_division_name`: the name of the census division.
3. `DGUID`: the dissemination geography unique identifier, in the format:

  2021 | A | 0003 | CDUID

  2021 - year the data was collected
  A - administrative (not important, a StatsCan identifier)
  0003 - schema indicating census division
  CDUID - the census division identifier

  See https://www150.statcan.gc.ca/n1/pub/92f0138m/92f0138m2019001-eng.htm for more details.
4. `CDUID`: the census division identifier.
5. `geographic_area_type`: the type of region.
6. `province`: the province or territory 2-letter id.
7. `population`: the number of people living in that census division.
8. `area_km2`: the area of the census division in square kilometres.
9. `population_density_per_km2`: the number of people per square kilometre.

## Boundary Files

The digital boundary files delineate the boundaries of the census divisions. These were downloaded
from:

https://www12.statcan.gc.ca/census-recensement/2021/geo/sip-pis/boundary-limites/index2021-eng.cfm?year=21

with the following selections:

1. `Type`: `Cartographic Boundary Files (CBF)`
2. `Administrative boundaries`: `Census divisions`
3. `Format`: `Shapefile (.shp)`

and saved under the folder `census_division_boundaries`.
