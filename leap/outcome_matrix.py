import pandas as pd
import numpy as np
import itertools
import datetime as dt
from dateutil.relativedelta import relativedelta
import pathlib
from leap.utils import date_range, TimeDelta
from leap.logger import get_logger

logger = get_logger(__name__)


class OutcomeTable:
    def __init__(self, data: pd.DataFrame, group_by: list[str] | None = None):
        """Create an outcome table.

        Args:
            data: The data for the table.
            group_by: The columns to group the data by.
        """
        self.group_by = group_by
        self.data = data  # the setter (re)sets the lazy ``grouped_data`` cache

    def __repr__(self):
        return self.data.__repr__()

    @property
    def data(self) -> pd.DataFrame:
        """The underlying dataframe for the table."""
        return self._data

    @data.setter
    def data(self, data: pd.DataFrame):
        self._data = data
        # Invalidate the cached groupby; it is rebuilt lazily on next access. Rebuilding it
        # eagerly on every construction/copy is pure overhead on the simulation's hot path,
        # since ``grouped_data`` is not read during a simulation run.
        self._grouped_data = None

    @property
    def grouped_data(self):
        """The dataframe grouped by ``group_by``, built lazily and cached on first access."""
        if self._grouped_data is None and self.group_by is not None:
            self._grouped_data = self._data.groupby(self.group_by)
        return self._grouped_data

    @grouped_data.setter
    def grouped_data(self, grouped_data):
        self._grouped_data = grouped_data

    def copy(self) -> "OutcomeTable":
        """Return a deep copy of this table.

        Copies the underlying typed arrays directly (via ``DataFrame.copy``) rather than
        re-parsing Python lists and datetimes the way ``OutcomeMatrix.create_table`` does, which
        makes it far cheaper than reconstructing the table from scratch.
        """
        return OutcomeTable(self._data.copy(deep=True), self.group_by)

    def increment(self, column: str, filter_columns: dict | None = None, amount: float | int = 1):
        """Increment the value of a column in the table.

        Args:
            column: The column to increment.
            filter_columns: A dictionary of columns to filter by.
            amount: The amount to increment the column by.

        .. note::

            When ``group_by`` is set, ``grouped_data`` is *not* rebuilt here. Because
            ``increment`` mutates ``self.data`` in place and only touches value columns (never the
            grouping keys or the row set), the existing ``grouped_data`` object still references the
            same underlying frame and reflects the updated values via ``get_group``. Rebuilding the
            groupby on every call was a large, unnecessary cost on the simulation's hottest path.
        """

        if filter_columns is not None:
            # Build a boolean mask directly instead of parsing a ``DataFrame.query`` string
            # (numexpr) on every call. ``filter_columns`` uniquely identifies the row(s) to update.
            mask = None
            for key, value in filter_columns.items():
                column_mask = self.data[key] == value
                mask = column_mask if mask is None else (mask & column_mask)
            self.data.loc[mask, column] += amount
        else:
            self.data[column] += amount

    def combine(self, outcome_table: "OutcomeTable", columns: list[str]):
        """Combine two outcome tables via summation.
        
        This method combines the current outcome table with another ``OutcomeTable`` instance
        by summing the values in the specified columns. The tables must have the same structure
        and groupings.
        
        Args:
            outcome_table: The outcome table to combine with this one.
            columns: The columns to combine. These should be the columns that contain the values
                to be summed. If the columns are not present in both tables, a ``ValueError`` will
                be raised.
        """
        if self.group_by != outcome_table.group_by:
            raise ValueError(
                "Cannot combine outcome tables with different group_by columns."
            )
        if not isinstance(outcome_table, OutcomeTable):
            raise TypeError("outcome_table must be an instance of OutcomeTable.")
        if outcome_table.data.shape[0] != self.data.shape[0]:
            raise ValueError(
                "Cannot combine outcome tables with different number of rows."
            )
        if (
            any(col not in self.data.columns for col in outcome_table.data.columns) or
            any(col not in outcome_table.data.columns for col in self.data.columns)
        ):
            raise ValueError(
                "Cannot combine outcome tables with different columns."
            )
        
        df = self.data.copy(deep=True)
        df = pd.merge(
            df,
            outcome_table.data,
            on=[col for col in df.columns if col not in columns],
            suffixes=("_x", "_y")
        )
        for column in columns:
            df[column] = df.apply(
                lambda x: x[f"{column}_x"] + x[f"{column}_y"],
                axis=1
            )
            df.drop(columns=[f"{column}_x", f"{column}_y"], inplace=True)
        self.data = df  # invalidates the cached ``grouped_data`` (rebuilt lazily on access)

    def get(self, columns: str | list[str], **kwargs) -> float | int | pd.Series | pd.DataFrame:
        """Get the value of a column in the table.

        Args:
            columns: The column(s) to get the value from.
            kwargs: A dictionary of columns to filter by.

        Returns:
            The value of the column.
        """
        df = self.data.copy(deep=True)
        for key, value in kwargs.items():
            df = df.loc[(df[key] == value)]
        df = df[columns]
        if df.shape == (1,):
            return df.iloc[0]
        else:
            return df


class OutcomeMatrix:
    """A class containing information about the outcomes of the model."""
    def __init__(
        self,
        until_all_die: bool,
        min_timepoint: dt.datetime,
        max_timepoint: dt.datetime,
        max_age: int,
        time_delta: dt.timedelta | relativedelta | TimeDelta
    ):
        """Initialize the ``OutcomeMatrix`` class.
        
        Args:
            until_all_die: A boolean indicating whether the simulation should run until all
                people have died.
            min_timepoint: The minimum timepoint of the simulation.
            max_timepoint: The maximum timepoint of the simulation.
            max_age: The maximum age of the people in the simulation.
            time_delta: The time interval between each timepoint in the simulation.
        """

        self.until_all_die = until_all_die
        self.min_timepoint = min_timepoint
        self.max_timepoint = max_timepoint
        self.max_age = max_age
        self.time_delta = time_delta
        self.value_columns = {
            "alive": ["n_alive"],
            "antibiotic_exposure": ["n_antibiotic_exposure"],
            "asthma_incidence": ["n_new_diagnoses"],
            "asthma_prevalence": ["n_asthma"],
            "asthma_incidence_contingency_table": ["n_asthma", "n_no_asthma"],
            "asthma_prevalence_contingency_table": ["n_asthma", "n_no_asthma"],
            "asthma_status": ["status"],
            "control": ["prob"],
            "cost": ["cost"],
            "death": ["n_deaths"],
            "emigration": ["n_emigrants"],
            "exacerbation": ["n_exacerbations"],
            "exacerbation_by_severity": ["p_exacerbations"],
            "exacerbation_hospital": ["n_hospitalizations"],
            "family_history": ["has_family_history"],
            "immigration": ["n_immigrants"],
            "utility": ["utility"]
        }
        time_delta_all_die = TimeDelta(years=max_age) if until_all_die else TimeDelta(days=0)

        self.alive = self.create_table(
            ["timepoint", "age", "sex", "n_alive"],
            None,
            date_range(
                start=min_timepoint,
                stop=max_timepoint + time_delta + time_delta_all_die,
                step=time_delta
            ),
            range(max_age + 1),
            ["F", "M"],
            [0]
        )
        self.antibiotic_exposure = self.create_table(
            ["timepoint", "age", "sex", "n_antibiotic_exposure"],
            None,
            date_range(
                start=min_timepoint,
                stop=max_timepoint + time_delta + time_delta_all_die,
                step=time_delta
            ),
            range(max_age + 1),
            ["F", "M"],
            [0]
        )
        self.asthma_incidence = self.create_table(
            ["timepoint", "age", "sex", "n_new_diagnoses"],
            None,
            date_range(
                start=min_timepoint,
                stop=max_timepoint + time_delta + time_delta_all_die,
                step=time_delta
            ),
            range(max_age + 1),
            ["F", "M"],
            [0]
        )
        self.asthma_prevalence = self.create_table(
            ["timepoint", "age", "sex", "n_asthma"],
            None,
            date_range(
                start=min_timepoint,
                stop=max_timepoint + time_delta + time_delta_all_die,
                step=time_delta
            ),
            range(max_age + 1),
            ["F", "M"],
            [0]
        )
        self.asthma_incidence_contingency_table = self.create_table(
            ["timepoint", "sex", "age", "fam_history", "abx_exposure", "n_asthma", "n_no_asthma"],
            ["timepoint", "sex", "fam_history", "abx_exposure"],
            date_range(min_timepoint, max_timepoint + time_delta, step=time_delta),
            range(0, 2),
            range(0, max_age + 2),
            range(0, 2),
            range(0, 4),
            [0],
            [0]
        )
        self.asthma_prevalence_contingency_table = self.create_table(
            ["timepoint", "sex", "age", "fam_history", "abx_exposure", "n_asthma", "n_no_asthma"],
            ["timepoint", "sex", "fam_history", "abx_exposure"],
            date_range(min_timepoint, max_timepoint + time_delta, step=time_delta),
            range(0, 2),
            range(0, max_age + 2),
            range(0, 2),
            range(0, 4),
            [0],
            [0]
        )
        self.asthma_status = self.create_table(
            ["timepoint", "age", "sex", "status"],
            ["timepoint"],
            date_range(
                start=min_timepoint,
                stop=max_timepoint + time_delta + time_delta_all_die,
                step=time_delta
            ),
            range(max_age + 1),
            ["F", "M"],
            [0]
        )
        self.control = self.create_table(
            ["timepoint", "level", "age", "sex", "prob"],
            ["timepoint", "level"],
            date_range(
                start=min_timepoint,
                stop=max_timepoint + time_delta + time_delta_all_die,
                step=time_delta
            ),
            range(3),
            range(max_age + 1),
            ["F", "M"],
            [0.0]
        )
        self.cost = self.create_table(
            ["timepoint", "age", "sex", "cost"],
            None,
            date_range(
                start=min_timepoint,
                stop=max_timepoint + time_delta + time_delta_all_die,
                step=time_delta
            ),
            range(max_age + 1),
            ["F", "M"],
            [0.0]
        )
        self.death = self.create_table(
            ["timepoint", "age", "sex", "n_deaths"],
            None,
            date_range(
                start=min_timepoint,
                stop=max_timepoint + time_delta + time_delta_all_die,
                step=time_delta
            ),            range(max_age + 1),
            ["F", "M"],
            [0]
        )
        self.emigration = self.create_table(
            ["timepoint", "age", "sex", "n_emigrants"],
            None,
            date_range(
                start=min_timepoint,
                stop=max_timepoint + time_delta + time_delta_all_die,
                step=time_delta
            ),
            range(max_age + 1),
            ["F", "M"],
            [0]
        )
        self.exacerbation = self.create_table(
            ["timepoint", "age", "sex", "n_exacerbations"],
            None,
            date_range(
                start=min_timepoint,
                stop=max_timepoint + time_delta + time_delta_all_die,
                step=time_delta
            ),
            range(max_age + 1),
            ["F", "M"],
            [0]
        )
        self.exacerbation_by_severity = self.create_table(
            ["timepoint", "severity", "age", "sex", "p_exacerbations"],
            ["timepoint", "severity"],
            date_range(
                start=min_timepoint,
                stop=max_timepoint + time_delta + time_delta_all_die,
                step=time_delta
            ),
            range(4),
            range(max_age + 1),
            ["F", "M"],
            [0.0]
        )
        self.exacerbation_hospital = self.create_table(
            ["timepoint", "age", "sex", "n_hospitalizations"],
            None,
            date_range(
                start=min_timepoint,
                stop=max_timepoint + time_delta + time_delta_all_die,
                step=time_delta
            ),
            range(max_age + 1),
            ["F", "M"],
            [0]
        )
        self.family_history = self.create_table(
            ["timepoint", "age", "sex", "has_family_history"],
            None,
            date_range(
                start=min_timepoint,
                stop=max_timepoint + time_delta + time_delta_all_die,
                step=time_delta
            ),
            range(max_age + 1),
            ["F", "M"],
            [0]
        )
        self.immigration = self.create_table(
            ["timepoint", "age", "sex", "n_immigrants"],
            None,
            date_range(
                start=min_timepoint,
                stop=max_timepoint + time_delta + time_delta_all_die,
                step=time_delta
            ),
            range(max_age + 1),
            ["F", "M"],
            [0]
        )
        self.utility = self.create_table(
            ["timepoint", "age", "sex", "utility"],
            None,
            date_range(
                start=min_timepoint,
                stop=max_timepoint + time_delta + time_delta_all_die,
                step=time_delta
            ),
            range(max_age + 1),
            ["F", "M"],
            [0.0]
        )

    def __repr__(self):
        attributes = self.__dict__.keys()
        string_repr = "OutcomeMatrix"
        for attribute in attributes:
            string_repr += f"\n\n{attribute}:\n{self.__getattribute__(attribute)}"

        return string_repr

    @property
    def alive(self) -> OutcomeTable:
        """A table containing the number of people alive in each timepoint, age, and sex."""
        return self._alive

    @alive.setter
    def alive(self, alive: OutcomeTable):
        self._alive = alive

    @property
    def antibiotic_exposure(self) -> OutcomeTable:
        """A table containing the number of rounds of antibiotics for each timepoint, age, and sex."""
        return self._antibiotic_exposure
    
    @antibiotic_exposure.setter
    def antibiotic_exposure(self, antibiotic_exposure: OutcomeTable):
        self._antibiotic_exposure = antibiotic_exposure

    @property
    def asthma_incidence(self) -> OutcomeTable:
        """A table containing the number of new asthma diagnoses for each timepoint, age, and sex."""
        return self._asthma_incidence
    
    @asthma_incidence.setter
    def asthma_incidence(self, asthma_incidence: OutcomeTable):
        self._asthma_incidence = asthma_incidence

    @property
    def asthma_prevalence(self) -> OutcomeTable:
        """A table containing the number of people with asthma for each timepoint, age, and sex."""
        return self._asthma_prevalence
    
    @asthma_prevalence.setter
    def asthma_prevalence(self, asthma_prevalence: OutcomeTable):
        self._asthma_prevalence = asthma_prevalence

    @property
    def asthma_incidence_contingency_table(self) -> OutcomeTable:
        """A contingency table with the following columns:
        
        .. list-table::
           :widths: 12 12 12 12 12 15 15
           :header-rows: 1

           * - timepoint
             - age
             - sex
             - has_family_history
             - abx_exposure
             - n_asthma
             - n_no_asthma
           * - timepoint in datetime format
             - age in years
             - one of ``F`` or ``M``
             - whether the person has a family history of asthma
             - number of courses of antibiotics in infancy
             - number of people diagnosed with asthma in the given timepoint, for the given
               age, sex, family history, and antibiotic exposure
             - number of people not diagnosed with asthma in the given timepoint, for the given
               age, sex, family history, and antibiotic exposure
           * - 2024-01-01 00:00:00
             - 15
             - ``F``
             - 0
             - 2
             - 10
             - 5
           * - ...
             - ...
             - ...
             - ...
             - ...
             - ...
             - ...
        """
        return self._asthma_incidence_contingency_table
    
    @asthma_incidence_contingency_table.setter
    def asthma_incidence_contingency_table(
        self, asthma_incidence_contingency_table: OutcomeTable
    ):
        self._asthma_incidence_contingency_table = asthma_incidence_contingency_table

    @property
    def asthma_prevalence_contingency_table(self) -> OutcomeTable:
        """A contingency table with the following columns:
        
        .. list-table::
           :widths: 12 12 12 12 12 15 15
           :header-rows: 1

           * - timepoint
             - age
             - sex
             - has_family_history
             - abx_exposure
             - n_asthma
             - n_no_asthma
           * - timepoint in datetime format
             - age in years
             - one of ``F`` or ``M``
             - whether the person has a family history of asthma
             - number of courses of antibiotics in infancy
             - number of people with asthma for the given timepoint, age, sex, family history,
               and antibiotic exposure
             - number of people without asthma for the given timepoint, age, sex, family history,
               and antibiotic exposure
           * - 2024-01-01 00:00:00
             - 15
             - ``F``
             - 0
             - 2
             - 100
             - 46
           * - ...
             - ...
             - ...
             - ...
             - ...
             - ...
             - ...

        """
        return self._asthma_prevalence_contingency_table
    
    @asthma_prevalence_contingency_table.setter
    def asthma_prevalence_contingency_table(
        self, asthma_prevalence_contingency_table: OutcomeTable
    ):
        self._asthma_prevalence_contingency_table = asthma_prevalence_contingency_table

    @property
    def asthma_status(self) -> OutcomeTable:
        """A table containing the status of asthma."""
        return self._asthma_status

    @asthma_status.setter
    def asthma_status(self, asthma_status: OutcomeTable):
        self._asthma_status = asthma_status

    @property
    def control(self) -> OutcomeTable:
        """A table containing the level of asthma control for each timepoint, age, and sex."""
        return self._control
    
    @control.setter
    def control(self, control: OutcomeTable):
        self._control = control

    @property
    def cost(self) -> OutcomeTable:
        """A table containing the cost of asthma for each timepoint, age, and sex."""
        return self._cost
    
    @cost.setter
    def cost(self, cost: OutcomeTable):
        self._cost = cost

    @property
    def death(self) -> OutcomeTable:
        """A table containing the number of people who died in a given timepoint, age, and sex."""
        return self._death
    
    @death.setter
    def death(self, death: OutcomeTable):
        self._death = death

    @property
    def emigration(self) -> OutcomeTable:
        """A table containing the number of people who emigrated to Canada for each timepoint, age,
        and sex."""
        return self._emigration

    @emigration.setter
    def emigration(self, emigration: OutcomeTable):
        self._emigration = emigration

    @property
    def exacerbation(self) -> OutcomeTable:
        """A table containing the number of asthma exacerbations for each timepoint, age, and sex."""
        return self._exacerbation

    @exacerbation.setter
    def exacerbation(self, exacerbation: OutcomeTable):
        self._exacerbation = exacerbation

    @property
    def exacerbation_by_severity(self) -> OutcomeTable:
        """A table containing the number of asthma exacerbations by severity."""
        return self._exacerbation_by_severity

    @exacerbation_by_severity.setter
    def exacerbation_by_severity(self, exacerbation_by_severity: OutcomeTable):
        self._exacerbation_by_severity = exacerbation_by_severity

    @property
    def exacerbation_hospital(self) -> OutcomeTable:
        """A table containing the number of asthma exacerbations leading to hospitalization."""
        return self._exacerbation_hospital
    
    @exacerbation_hospital.setter
    def exacerbation_hospital(self, exacerbation_hospital: OutcomeTable):
        self._exacerbation_hospital = exacerbation_hospital

    @property
    def family_history(self) -> OutcomeTable:
        """A table containing the number of people with a family history of asthma."""
        return self._family_history
    
    @family_history.setter
    def family_history(self, family_history: OutcomeTable):
        self._family_history = family_history

    @property
    def immigration(self) -> OutcomeTable:
        """A table containing the number of people who immigrated to Canada for each timepoint, age,
        and sex."""
        return self._immigration

    @immigration.setter
    def immigration(self, immigration: OutcomeTable):
        self._immigration = immigration

    @property
    def utility(self) -> OutcomeTable:
        """A table containing the utility due to asthma for each timepoint, age, and sex."""
        return self._utility
    
    @utility.setter
    def utility(self, utility: OutcomeTable):
        self._utility = utility

    def create_table(
        self,
        columns: list[str],
        group_by: list[str] | None,
        *args
    ) -> OutcomeTable:
        """Create an outcome table.

        Args:
            columns: The list of column names for the table.
            group_by: The list of column names to group by. Set to ``None`` if no grouping
                is needed.
            args: A list of lists, where each list corresponds to a column in the table.
                Each list is used to create a Cartesian product of the columns,
                so it should contain the set of possible values for that column.

        Returns:
            An ``OutcomeTable`` object containing the data for the table.

        Examples:

            >>> from leap.outcome_matrix import OutcomeMatrix
            >>> outcome_matrix = OutcomeMatrix(
            ...     until_all_die=False,
            ...     min_timepoint=dt.datetime(2024, 1, 1),
            ...     max_timepoint=dt.datetime(2030, 1, 1),
            ...     max_age=100,
            ...     time_delta=dt.timedelta(days=366)
            ... )
            >>> table = outcome_matrix.create_table(
            ...     ["timepoint", "age", "sex", "n_alive"],
            ...     None,
            ...     date_range(dt.datetime(2024, 1, 1), dt.datetime(2026, 1, 1), dt.timedelta(days=366)),
            ...     range(1, 3),
            ...     ["F", "M"],
            ...     [0]
            ... )
            >>> print(table) # doctest: +NORMALIZE_WHITESPACE
            timepoint  age sex  n_alive
            0  2024-01-01    1   F        0
            1  2024-01-01    1   M        0
            2  2024-01-01    2   F        0
            3  2024-01-01    2   M        0
            4  2025-01-01    1   F        0
            5  2025-01-01    1   M        0
            6  2025-01-01    2   F        0
            7  2025-01-01    2   M        0

        """
        product = itertools.product(
            *args
        )
        data = []
        for row in product:
            data.append(list(row))
        df = pd.DataFrame(
            data,
            columns=columns
        )
        table = OutcomeTable(df, group_by)
        return table

    def copy(self) -> "OutcomeMatrix":
        """Return a copy of this outcome matrix with independent, freshly-copied tables.

        This lets each simulated agent get its own zero-initialized outcome matrix by copying a
        shared template, instead of rebuilding all ~17 tables from scratch via ``create_table``
        (Cartesian product + DataFrame parsing) for every agent. The scalar attributes
        (timepoints, ``max_age``, ``value_columns`` etc.) are shared by reference since they are
        never mutated; only the ``OutcomeTable`` s are deep-copied.
        """
        new = OutcomeMatrix.__new__(OutcomeMatrix)
        for attribute, value in self.__dict__.items():
            if isinstance(value, OutcomeTable):
                setattr(new, attribute, value.copy())
            else:
                setattr(new, attribute, value)
        return new

    def combine(self, outcome_matrix: "OutcomeMatrix"):
        """Combine two outcome matrices via summation.

        This method combines the outcome tables of the current instance with those of
        another ``OutcomeMatrix`` instance by summing the values in the ``value_columns``
        for each corresponding table. The tables must have the same structure and groupings.

        Args:
            outcome_matrix: The outcome matrix to combine with this one.
        """
        if not isinstance(outcome_matrix, OutcomeMatrix):
            raise TypeError("outcome_matrix must be an instance of OutcomeMatrix.")
        
        attributes = [
            key for key, value in self.__dict__.items() if isinstance(value, OutcomeTable)
        ]
        for attribute in attributes:
            self.__getattribute__(attribute).combine(
                outcome_table=outcome_matrix.__getattribute__(attribute),
                columns=self.value_columns[attribute.removeprefix("_")]
            )

    def save(self, path: pathlib.Path):
        """Save the outcome matrix to ``*.csv`` files.

        Args:
            path: The full path to the folder where the data will be saved.
        """
        if not path.exists():
            raise Exception(f"Path {path} does not exist.")
        
        attributes = [
            key for key, value in self.__dict__.items() if isinstance(value, OutcomeTable)
        ]
        for attribute in attributes:
            file_path = pathlib.Path(path.resolve(), f"outcome_matrix_{attribute.removeprefix('_')}.csv")
            self.__getattribute__(attribute).data.to_csv(file_path, index=False)
            logger.message(f"Saved {attribute.removeprefix('_')} to {file_path}.")


def combine_outcome_tables(outcome_tables: list[OutcomeTable], column: str) -> OutcomeTable:
    """Combine a list of outcome tables into a single outcome table.

    Args:
        outcome_tables: A list of ``OutcomeTable`` instances to combine.
        column: The column to sum across the outcome tables.

    Returns:
        An ``OutcomeTable`` instance containing the combined data.
    """
    df_combined = outcome_tables[0].data.copy()
    df_combined[column] = np.array([df.data[column] for df in outcome_tables]).sum(axis=0)
    return OutcomeTable(df_combined, group_by=outcome_tables[0].group_by)


def combine_outcome_matrices(outcome_matrices: list[OutcomeMatrix]) -> OutcomeMatrix:
    """Combine a list of outcome matrices into a single outcome matrix.

    Args:
        outcome_matrices: A list of ``OutcomeMatrix`` instances to combine.

    Returns:
        An ``OutcomeMatrix`` instance containing the combined data.
    """

    combined_matrix = OutcomeMatrix(
        until_all_die=outcome_matrices[0].until_all_die,
        min_timepoint=outcome_matrices[0].min_timepoint,
        max_timepoint=outcome_matrices[0].max_timepoint,
        max_age=outcome_matrices[0].max_age,
        time_delta=outcome_matrices[0].time_delta
    )
    
    for attribute in combined_matrix.__dict__.keys():
        if isinstance(getattr(combined_matrix, attribute), OutcomeTable):
            outcome_tables = [getattr(matrix, attribute) for matrix in outcome_matrices]
            combined_table = combine_outcome_tables(
                outcome_tables=outcome_tables,
                column=combined_matrix.value_columns[attribute.removeprefix("_")][0]
            )
            setattr(combined_matrix, attribute, combined_table)
    
    return combined_matrix