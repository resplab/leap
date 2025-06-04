import pandas as pd
import numpy as np
import itertools
import re
from leap.utils import get_data_path
from leap.logger import get_logger
from leap.data_generation.utils import parse_age_group

pd.options.mode.copy_on_write = True

logger = get_logger(__name__, 20)

MAX_AGE = 111


def load_eq5d_data(max_age: int = MAX_AGE) -> pd.DataFrame:
    """Load EQ-5D data from the original Excel file and process it into a DataFrame.

    The original data file is formatted as follows:

    .. raw:: html

        <table class="table">
            <thead>
            <tr>
                <th>Variable</th>
                <th>Total<br>(N = 1207)</th>
                <th>p value*</th>
                <th>Female<br>(N = 662)</th>
                <th>Male<br>(N = 545)</th>
            </tr>
            </thead>
            <tbody>
            <tr>
                <td>All</td>
                <td>eq5d (sd)</td>
                <td></td>
                <td>eq5d (sd)</td>
                <td>eq5d (sd)</td>
            </tr>
            <tr>
                <td>Age group</td>
                <td></td>
                <td></td>
                <td></td>
                <td></td>
            </tr>
            <tr>
                <td>18–24</td>
                <td>eq5d (sd)</td>
                <td></td>
                <td>eq5d (sd)</td>
                <td>eq5d (sd)</td>
            </tr>
            <tr>
                <td>25–34</td>
                <td>eq5d (sd)</td>
                <td></td>
                <td>eq5d (sd)</td>
                <td>eq5d (sd)</td>
            </tr>
            <tr>
                <td>35–44</td>
                <td>eq5d (sd)</td>
                <td></td>
                <td>eq5d (sd)</td>
                <td>eq5d (sd)</td>
            </tr>
            <tr>
                <td>45–54</td>
                <td>eq5d (sd)</td>
                <td></td>
                <td>eq5d (sd)</td>
                <td>eq5d (sd)</td>
            </tr>
            <tr>
                <td>55–64</td>
                <td>eq5d (sd)</td>
                <td></td>
                <td>eq5d (sd)</td>
                <td>eq5d (sd)</td>
            </tr>
            <tr>
                <td>65–74</td>
                <td>eq5d (sd)</td>
                <td></td>
                <td>eq5d (sd)</td>
                <td>eq5d (sd)</td>
            </tr>
            <tr>
                <td>75 +</td>
                <td>eq5d (sd)</td>
                <td>&lt; 0.01</td>
                <td>eq5d (sd)</td>
                <td>eq5d (sd)</td>
            </tr>
            <tr>
                <td>Education</td>
                <td></td>
                <td></td>
                <td></td>
                <td></td>
            </tr>
            <tr>
                <td>...</td>
                <td>...</td>
                <td>...</td>
                <td>...</td>
                <td>...</td>
            </tr>
        </tbody>
        </table>

			
    Args:
        max_age (int): The maximum age to consider in the data.
    
    Returns:
        A DataFrame containing EQ-5D utility values by age and sex.
        Columns:

        * ``age (int)``: Age of the individual.
        * ``sex (str)``: One of ``"F"`` = female or ``"M"`` = male.
        * ``eq5d (float)``: EQ-5D utility value.
        * ``sd (float)``: Standard deviation of the EQ-5D utility value.
    """

    # Load the original EQ5D data from the Excel file
    df = pd.read_excel(
        get_data_path("original_data/Table_3_Mean_(SD)_EQ-5D-5L_utilities_by_socio-demographic_characteristics.xlsx")
    )

    # Replace "en dash" (u2013) with "hyphen minus" (Python's "-")
    df["Variable"] = df["Variable"].str.replace(u"\u2013", "-")

    # Filter only the age groups
    df = df.loc[df["Variable"].isin(["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75 +"])]

    # Rename "Variable" column to "age_group"
    df.rename(columns={"Variable": "age_group"}, inplace=True)

    # Find column containing EQ5D values for females
    regex = re.compile("^(Female)")
    index = np.where([regex.match(x) is not None for x in df.columns])[0][0]
    df.rename(columns={df.columns[index]: "F"}, inplace=True)

    # Find column containing EQ5D values for males
    regex = re.compile("^(Male)")
    index = np.where([regex.match(x) is not None for x in df.columns])[0][0]
    df.rename(columns={df.columns[index]: "M"}, inplace=True)

    # Parse the age groups to get lower and upper bounds
    df["age_upper"] = df["age_group"].apply(lambda x: parse_age_group(x, max_age=max_age)[1])
    df["age_lower"] = df["age_group"].apply(lambda x: parse_age_group(x, max_age=max_age)[0])

    # Add "sex" column and move the "F" and "M" columns to a new column "eq5d"
    df = df.melt(
        id_vars=["age_group", "age_lower", "age_upper"], value_vars=["F", "M"],
        var_name="sex", value_name="eq5d"
    )

    # Split values in "eq5d" column, formatted as "eq5d (sd)", into separate columns for
    # EQ-5D value and standard deviation
    df["sd"] = df["eq5d"].apply(lambda x: float(x.split(" (")[1].replace(")", "")))
    df["eq5d"] = df["eq5d"].apply(lambda x: float(x.split(" (")[0]))

    df.sort_values(by=["age_lower"], inplace=True)

    # Create a DataFrame with all ages from 18 to 111
    df_utility = pd.DataFrame(
        data=list(itertools.product(list(range(18, max_age + 1)), ["F", "M"], [0], [0])),
        columns=["age", "sex", "eq5d", "sd"]
    )

    # Set the EQ-5D values and standard deviations for each age to the values of the age group
    df_utility["eq5d"] = df_utility.apply(
        lambda x: df.loc[
            (df["age_lower"] <= x["age"]) & (df["age_upper"] >= x["age"]) & (df["sex"] == x["sex"])
        ]["eq5d"].values[0], axis=1
    )
    df_utility["sd"] = df_utility.apply(
        lambda x: df.loc[
            (df["age_lower"] <= x["age"]) & (df["age_upper"] >= x["age"]) & (df["sex"] == x["sex"])
        ]["sd"].values[0], axis=1
    )

    return df_utility


def interpolate_eq5d(
    age: int,
    eq5d_upper: float,
    age_upper: int = 18
) -> float:
    """Interpolate EQ-5D value for ages below 18 based on the EQ-5D value at 18.
    
    Args:
        age: Age for which to interpolate the EQ-5D value. Must be in range ``[0, 18]``.
        eq5d_upper: EQ-5D value at the upper age limit (default is ``18``).
        age_upper: Upper age limit for interpolation (default is ``18``).
        
    Returns:
        Interpolated EQ-5D value for the given age.
    """

    if age > age_upper:
        raise ValueError(f"Age {age} must be less than or equal to {age_upper}.")

    eq5d = 1 - (1 - eq5d_upper) / age_upper * age
    return eq5d
    

def interpolate_eq5d_data(df_utility: pd.DataFrame) -> pd.DataFrame:
    """Interpolate EQ-5D data to fill in missing ages below 18.

    The EQ-5D data was only available for ages 18 and above, so this function
    generates EQ-5D values for ages 0 to 17 by interpolating from the values at age 18.
    
    Args:
        df_utility: DataFrame containing EQ-5D data for ages 18 and above:

            * ``age (int)``: Age of the individual.
            * ``sex (str)``: One of ``"F"`` = female or ``"M"`` = male.
            * ``eq5d (float)``: EQ-5D utility value.
            * ``sd (float)``: Standard deviation of the EQ-5D utility value.

    Returns:
        A dataframe containing EQ-5D data for ages 0 to 17, interpolated from the values at age 18.
        Columns:

        * ``age (int)``: Age of the individual, now including ages 0 to 17.
        * ``sex (str)``: One of ``"F"`` = female, ``"M"`` = male.
        * ``eq5d (float)``: Interpolated EQ-5D utility value for ages 0 to 17.
        * ``sd (float)``: Since the data was interpolated, the standard deviation is set to 0
          for these ages.
    """
    df = pd.DataFrame(
        data=list(itertools.product(list(range(0, 18)), ["F", "M"], [0], [0])),
        columns=["age", "sex", "eq5d", "sd"]
    )
    df["eq5d"] = df.apply(
        lambda x: interpolate_eq5d(
            age=x["age"],
            eq5d_upper=df_utility.loc[
                (df_utility["sex"] == x["sex"]) & (df_utility["age"] == 18)
            ]["eq5d"].values[0],
            age_upper=18
        ),
        axis=1
    )
    return df


def generate_eq5d_data():
    """Generate EQ-5D data for ages 0 to 111."""

    df_utility_adult = load_eq5d_data()
    df_utility_child = interpolate_eq5d_data(df_utility_adult)
    df_utility = pd.concat([df_utility_child, df_utility_adult], ignore_index=True)
    df_utility.to_csv(get_data_path("processed_data/eq5d_canada.csv"), index=False)
    logger.info("EQ-5D data generated and saved to 'processed_data/eq5d_canada.csv'.")


if __name__ == "__main__":
    generate_eq5d_data()

