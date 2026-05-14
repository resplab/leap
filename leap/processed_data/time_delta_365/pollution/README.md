# PM2.5 Projections for IPCC Scenarios

The data in this folder details the projections for the years 2024-2036 based on the
Shared Socioeconomic Pathways climate change scenarios.

SSP1: Sustainability
SSP2: Middle of the Road
SSP3: Regional Rivalry
SSP4: Inequality
SSP5: Fossil-fueled Development

Within each of these scenarios, there are different categories based on global temperature increase
and greenhouse gas emissions. We have chosen to include the following in our app:

SSP1-2.6:
  Very likely range of warming: 1.3 - 2.4 degrees Celsius by 2081-2100
  Low greenhouse gas emissions
  CO2 emissions cut to net zero by 2075

SSP2-4.5:
  Very likely range of warming: 2.1 - 3.5 degrees Celsius by 2081-2100
  Intermediate greenhouse gas emissions
  CO2 emissions around current levels until 2050, then falling but not reaching net zero by 2100

SSP3-7.0:
  Very likely range of warming: 2.8 - 4.6 degrees Celsius by 2081-2100
  High greenhouse gas emissions
  CO2 emissions double by 2100

SSP5-8.5:
  Very likely range of warming: 3.3 - 5.7 degrees Celsius by 2081-2100
  Very high greenhouse gas emissions
  CO2 emissions triple by 2075

## Data Columns

The baseline projections are obtained from the ECCC using their GEM-MACH model, which gives us a
prediction for background PM2.5 levels and wildfire PM2.5 levels for the years 2026, 2031, and 2036.
All other years are calculated using linear interpolation:

`background_pm25`: the average background PM2.5 levels for a given month.
`wildfire_pm25`: the average PM2.5 levels due to wildfires for a given month.

To compute the added effect of different IPCC climate change scenarios, we use a climate scaling
factor to predict the increase in wildfires:

`factor`: the future climate scaling factor.
`wildfire_pm25_scaled`: `wildfire_pm25` * `factor`.

Finally, the total PM2.5 levels are calculated:

`total_pm25`: the total average PM2.5 levels for a given month:
  `wildfire_pm25_scaled` + `background_pm25`

The last column indicates which SSP scenario is used:

`SSP`: one of `SSP1_2.6`, `SSP2_4.5`, `SSP3_7.0`, `SSP5_8.5`
