import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from leap.utils import get_data_path
from leap.logger import get_logger
logger = get_logger(__name__, 20)

PROVINCE_MAP = {
    "Canada": "CA",
    "British Columbia": "BC",
    "Alberta": "AB",
    "Saskatchewan": "SK",
    "Manitoba": "MB",
    "Ontario": "ON",
    "Quebec": "QC",
    "Newfoundland and Labrador": "NL",
    "Nova Scotia": "NS",
    "New Brunswick": "NB",
    "Prince Edward Island": "PE",
    "Yukon": "YT",
    "Northwest Territories": "NT",
    "Nunavut": "NU",
}


def get_province_id(province: str) -> str:
    """Convert full length province name to abbreviation.

    Args:
        province: The full length province name, e.g. ``British Columbia``.

    Returns:
        The abbreviation for the province, e.g. ``BC``.
    """
    return PROVINCE_MAP[province]


def get_sex_id(sex: str) -> str:
    """Convert full length sex to single character.

    Args:
        sex: The full length string, either ``Female`` or ``Male``

    Returns:
        The single character string, either ``F`` or ``M``.
    """
    return sex[0:1]


def format_age_group(age_group: str, upper_age_group: str = "100 years and over") -> int:
    """Convert age group to integer.

    Args:
        age_group: The age group string, e.g. ``5 to 9 years``.
        upper_age_group: The upper age group string, e.g. ``100 years and over``.

    Returns:
        The integer age.

    Examples:

    >>> format_age_group("110 years and over", "110 years and over")
    110
    >>> format_age_group("Under 1 year", "100 years and over")
    0
    >>> format_age_group("9 years")
    9
    """
    if age_group == upper_age_group:
        age = age_group.replace(" years and over", "")
        age = int(age)
    elif age_group == "Under 1 year":
        age = 0
    else:
        age = age_group.replace(" years", "")
        age = age.replace(" year", "")
        age = int(age)
    return age


def interpolate_years_to_months(
    dataset: str, group_cols: list[str], interp_cols: list[str], method: str = "linear"
):
    """
    Interpolates values of specified csv between years as months.

    The resulting file will be saved in the same directory as the source and
    will have the name ``<dataset>_<method>_interpolated.csv``

    Args:
        dataset: Path to csv that will be interpolated. The csv must have a year column.
        group_cols: A list of column names to group the rows by.
                    (i.e. for a given year, group by age and sex)
        interp_cols: A list of column names to interpolate.
                    (e.x. total population ``n``)
        method: Which interpolation method to use, ``linear`` or ``loess``
        
    Examples:
        ``input.csv``:
        
        +------+-----+-------+
        | year | sex | value |
        +======+=====+=======+
        | 2005 | F   | 10    |
        +------+-----+-------+
        | 2005 | M   | 0     |
        +------+-----+-------+
        | 2006 | F   | 120   |
        +------+-----+-------+
        | 2006 | M   | 22    |
        +------+-----+-------+
        
        >>> interpolate_years_to_months(
        ...     dataset='input.csv',
        ...     group_cols=['sex'],
        ...     interp_cols=['value'],
        ...     method='linear'
        ... )
        
        ``input_linear_interpolated.csv``:
        
        +------------+-----+-------+
        | year_month | sex | value |
        +============+=====+=======+
        | 2005-01    | F   | 10    |
        +------------+-----+-------+
        | 2005-01    | M   | 0     |
        +------------+-----+-------+
        | 2005-02    | F   | 20    |
        +------------+-----+-------+
        | 2005-02    | M   | 2     |
        +------------+-----+-------+
        | 2005-03    | F   | 30    |
        +------------+-----+-------+
        | 2005-03    | M   | 4     |
        +------------+-----+-------+
        | 2005-04    | F   | 40    |
        +------------+-----+-------+
        | 2005-04    | M   | 6     |
        +------------+-----+-------+
        | 2005-05    | F   | 50    |
        +------------+-----+-------+
        | 2005-05    | M   | 8     |
        +------------+-----+-------+
        | 2005-06    | F   | 60    |
        +------------+-----+-------+
        | 2005-06    | M   | 10    |
        +------------+-----+-------+
        | 2005-07    | F   | 70    |
        +------------+-----+-------+
        | 2005-07    | M   | 12    |
        +------------+-----+-------+
        | 2005-08    | F   | 80    |
        +------------+-----+-------+
        | 2005-08    | M   | 14    |
        +------------+-----+-------+
        | 2005-09    | F   | 90    |
        +------------+-----+-------+
        | 2005-09    | M   | 16    |
        +------------+-----+-------+
        | 2005-10    | F   | 100   |
        +------------+-----+-------+
        | 2005-10    | M   | 18    |
        +------------+-----+-------+
        | 2005-11    | F   | 110   |
        +------------+-----+-------+
        | 2005-11    | M   | 20    |
        +------------+-----+-------+
        | 2005-12    | F   | 120   |
        +------------+-----+-------+
        | 2005-12    | M   | 22    |
        +------------+-----+-------+
        | 2006-01    | F   | 120   |
        +------------+-----+-------+
        | 2006-01    | M   | 22    |
        +------------+-----+-------+
        | 2006-02    | F   | 120   |
        +------------+-----+-------+
        | 2006-02    | M   | 22    |
        +------------+-----+-------+
        | 2006-03    | F   | 120   |
        +------------+-----+-------+
        | 2006-03    | M   | 22    |
        +------------+-----+-------+
        | 2006-04    | F   | 120   |
        +------------+-----+-------+
        | 2006-04    | M   | 22    |
        +------------+-----+-------+
        | 2006-05    | F   | 120   |
        +------------+-----+-------+
        | 2006-05    | M   | 22    |
        +------------+-----+-------+
        | 2006-06    | F   | 120   |
        +------------+-----+-------+
        | 2006-06    | M   | 22    |
        +------------+-----+-------+
        | 2006-07    | F   | 120   |
        +------------+-----+-------+
        | 2006-07    | M   | 22    |
        +------------+-----+-------+
        | 2006-08    | F   | 120   |
        +------------+-----+-------+
        | 2006-08    | M   | 22    |
        +------------+-----+-------+
        | 2006-09    | F   | 120   |
        +------------+-----+-------+
        | 2006-09    | M   | 22    |
        +------------+-----+-------+
        | 2006-10    | F   | 120   |
        +------------+-----+-------+
        | 2006-10    | M   | 22    |
        +------------+-----+-------+
        | 2006-11    | F   | 120   |
        +------------+-----+-------+
        | 2006-11    | M   | 22    |
        +------------+-----+-------+
        | 2006-12    | F   | 120   |
        +------------+-----+-------+
        | 2006-12    | M   | 22    |
        +------------+-----+-------+

    """
    if method not in ["linear", "loess"]:
        raise ValueError(f"method was {method}. Must be one of ['linear', 'loess']")
    
    logger.info(f"Loading dataset from {dataset}")
    df = pd.read_csv(get_data_path(data_path=dataset))

    # Storage for interpolated output dictionaries
    all_rows = []
    
    for group_key, group_df in df.groupby(group_cols):
        logger.info(f"Interpolating data with {method} for group {group_key}.")
        group_df = group_df.sort_values("year")
        
        # Create monthly x-values for interpolation
        year_min = group_df["year"].min()
        year_max = group_df["year"].max()
        monthly_x = np.linspace(year_min, year_max, int((year_max - year_min) * 12) + 1)
        
        if method == "linear":
            for i in range(len(group_df) - 1):
                row_start = group_df.iloc[i]
                row_end = group_df.iloc[i + 1]
                
                # Cache constant values within row
                row_groups = {col : row_start[col] for col in group_cols}
                
                for m in range(12):
                    fraction = m / 11
                    year_interp = row_start["year"] + fraction
                    year_month = f"{row_start['year']}-{m+1:02d}"

                    # Use interpolated years, and other grouped columns
                    interpolated_row = {"year_month": year_month} | row_groups
                    
                    for icol in interp_cols:
                        interpolated_row[icol] = (
                            row_start[icol] + fraction * (row_end[icol] - row_start[icol])
                        )
                    all_rows.append(interpolated_row)

            # Repeat the final year (all 12 months, constant values)
            final_row = group_df.loc[group_df["year"].idxmax()]
            for m in range(12):
                year_interp = final_row["year"] + m / 11
                year_month = f"{final_row['year']}-{m+1:02d}"
                all_rows.append({
                    "year_month": year_month,
                    **{col: final_row[col] for col in group_cols},
                    **{col: final_row[col] for col in interp_cols}
                })

        elif method == "loess":
            for icol in interp_cols:
                smoothed = lowess(
                    endog=group_df[icol],   # y value
                    exog=group_df["year"],  # x value
                    frac=0.5,               # controls smoothing
                    return_sorted=True
                )
                interp_values = np.interp(monthly_x, smoothed[:, 0], smoothed[:, 1])
                
                # Store into temp dict by column
                if icol == interp_cols[0]: # built row once for first column
                    rows = [
                        {
                            "year_float": x,
                            **{gcol: group_key[i] for i, gcol in enumerate(interp_cols)},
                            icol: val
                        } for x, val in zip(monthly_x, interp_values)
                    ]
                else: # reuse row for next columns
                    for r, val in zip(rows, interp_values):
                        r[icol] = val
            all_rows.extend(rows)

    # Convert to DataFrame and sort
    monthly_df = pd.DataFrame(all_rows)
    SCENARIO_ORDERING = ["past", "M1", "M2", "M3", "M4", "M5", "M6", "LG", "HG", "FA", "SA"]
    # If 'scenario' is present, cast it to categorical with the specified order
    if "projection_scenario" in monthly_df.columns:
        monthly_df["projection_scenario"] = pd.Categorical(
            monthly_df["projection_scenario"], categories=SCENARIO_ORDERING, ordered=True
        )
    
    # Final dataframe cleanup 
    COLUMN_ORDERING = ["province", "projection_scenario", "year_month", "age", "sex"]
    monthly_df = monthly_df.sort_values(
        [col for col in COLUMN_ORDERING if col in monthly_df.columns]
    )
    # monthly_df["year_month"] = monthly_df["year_float"].apply(
    #     lambda y: f"{int(y):04d}-{min(12, round((y - int(y)) * 12) + 1):02d}"
    # )
    # monthly_df = monthly_df.drop(columns="year_float")
    monthly_df = monthly_df[
        ["year_month"] + [col for col in monthly_df.columns if col != "year_month"]
    ]

    # Save to CSV
    dataset_name = dataset.split(".")[0]
    interp_file_name = f"{dataset_name}_{method}_interpolated.csv"
    file_path = get_data_path(interp_file_name, create=True)
    logger.info(f"Saving data to {file_path}")
    monthly_df.to_csv(file_path, float_format="%.8g", index=False)
