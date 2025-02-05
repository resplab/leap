import pandas as pd
import itertools
import pathlib
from leap.utils import check_file
from leap.logger import get_logger

logger = get_logger(__name__)


class OutcomeTable:
    def __init__(self, data: pd.DataFrame, group_by: list[str] | None = None):
        """Create an outcome table.

        Args:
            data: The data for the table.
            group_by: The columns to group the data by.
        """
        self.data = data
        self.group_by = group_by
        if group_by is not None:
            self.grouped_data = data.groupby(group_by)
        else:
            self.grouped_data = None

    def __repr__(self):
        return self.data.__repr__()

    def increment(self, column: str, filter_columns: dict | None = None, amount: float | int = 1):
        """Increment the value of a column in the table.

        Args:
            column: The column to increment.
            amount: The amount to increment the column by.
            filter_columns: A dictionary of columns to filter by.
        """
        df = self.data.copy(deep=True)
        if filter_columns is not None:
            df_filtered = df.copy(deep=True)
            for key, value in filter_columns.items():
                df_filtered = df_filtered.loc[(df_filtered[key] == value)]
            df_filtered[column] += amount
            df.update(df_filtered)
        else:
            df[column] += amount
        self.data = df
        if self.group_by is not None:
            self.grouped_data = self.data.groupby(self.group_by)

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
        self, until_all_die: bool, min_year: int, max_year: int, max_age: int
    ):
        """Initialize the ``OutcomeMatrix`` class.

        Args:
            until_all_die: A boolean indicating whether the simulation should run until all
                people have died.
            min_year: The minimum year of the simulation.
            max_year: The maximum year of the simulation.
            max_age: The maximum age of the people in the simulation.
        """

        self.until_all_die = until_all_die
        self.min_year = min_year
        self.max_year = max_year
        self.max_age = max_age

        self.alive = self.create_table(
            ["year", "age", "sex", "n_alive"],
            ["year"],
            range(min_year, max_year + 1 + (max_age if until_all_die else 0)),
            range(max_age + 1),
            ["F", "M"],
            [0]
        )
        self.antibiotic_exposure = self.create_table(
            ["year", "age", "sex", "n_antibiotic_exposure"],
            ["year"],
            range(min_year, max_year + 1 + (max_age if until_all_die else 0)),
            range(max_age + 1),
            ["F", "M"],
            [0]
        )
        self.asthma_incidence = self.create_table(
            ["year", "age", "sex", "n_new_diagnoses"],
            ["year"],
            range(min_year, max_year + 1 + (max_age if until_all_die else 0)),
            range(max_age + 1),
            ["F", "M"],
            [0]
        )
        self.asthma_prevalence = self.create_table(
            ["year", "age", "sex", "n_asthma"],
            ["year"],
            range(min_year, max_year + 1 + (max_age if until_all_die else 0)),
            range(max_age + 1),
            ["F", "M"],
            [0]
        )
        self.asthma_incidence_contingency_table = self.create_table(
            ["year", "sex", "age", "fam_history", "abx_exposure", "n_asthma", "n_no_asthma"],
            ["year", "sex", "fam_history", "abx_exposure"],
            range(min_year, max_year + 1),
            range(0, 2),
            range(0, max_age + 2),
            range(0, 2),
            range(0, 4),
            [0],
            [0]
        )
        self.asthma_prevalence_contingency_table = self.create_table(
            ["year", "sex", "age", "fam_history", "abx_exposure", "n_asthma", "n_no_asthma"],
            ["year", "sex", "fam_history", "abx_exposure"],
            range(min_year, max_year + 1),
            range(0, 2),
            range(0, max_age + 2),
            range(0, 2),
            range(0, 4),
            [0],
            [0]
        )
        self.asthma_status = self.create_table(
            ["year", "age", "sex", "status"],
            ["year"],
            range(min_year, max_year + 1 + (max_age if until_all_die else 0)),
            range(max_age + 1),
            ["F", "M"],
            [0]
        )
        self.control = self.create_table(
            ["year", "level", "age", "sex", "prob"],
            ["year", "level"],
            range(min_year, max_year + 1 + (max_age if until_all_die else 0)),
            range(3),
            range(max_age + 1),
            ["F", "M"],
            [0.0]
        )
        self.cost = self.create_table(
            ["year", "age", "sex", "cost"],
            ["year"],
            range(min_year, max_year + 1 + (max_age if until_all_die else 0)),
            range(max_age + 1),
            ["F", "M"],
            [0.0]
        )
        self.death = self.create_table(
            ["year", "age", "sex", "n_deaths"],
            ["year"],
            range(min_year, max_year + 1 + (max_age if until_all_die else 0)),
            range(max_age + 1),
            ["F", "M"],
            [0]
        )
        self.emigration = self.create_table(
            ["year", "age", "sex", "n_emigrants"],
            ["year"],
            range(min_year, max_year + 1 + (max_age if until_all_die else 0)),
            range(max_age + 1),
            ["F", "M"],
            [0]
        )
        self.exacerbation = self.create_table(
            ["year", "age", "sex", "n_exacerbations"],
            ["year"],
            range(min_year, max_year + 1 + (max_age if until_all_die else 0)),
            range(max_age + 1),
            ["F", "M"],
            [0]
        )
        self.exacerbation_by_severity = self.create_table(
            ["year", "severity", "age", "sex", "p_exacerbations"],
            ["year", "severity"],
            range(min_year, max_year + 1 + (max_age if until_all_die else 0)),
            range(4),
            range(max_age + 1),
            ["F", "M"],
            [0.0]
        )
        self.exacerbation_hospital = self.create_table(
            ["year", "age", "sex", "n_hospitalizations"],
            ["year"],
            range(min_year, max_year + 1 + (max_age if until_all_die else 0)),
            range(max_age + 1),
            ["F", "M"],
            [0]
        )
        self.family_history = self.create_table(
            ["year", "age", "sex", "has_family_history"],
            ["year"],
            range(min_year, max_year + 1 + (max_age if until_all_die else 0)),
            range(max_age + 1),
            ["F", "M"],
            [0]
        )
        self.immigration = self.create_table(
            ["year", "age", "sex", "n_immigrants"],
            ["year"],
            range(min_year, max_year + 1 + (max_age if until_all_die else 0)),
            range(max_age + 1),
            ["F", "M"],
            [0]
        )
        self.utility = self.create_table(
            ["year", "age", "sex", "utility"],
            ["year"],
            range(min_year, max_year + 1 + (max_age if until_all_die else 0)),
            range(max_age + 1),
            ["F", "M"],
            [0]
        )

    def __repr__(self):
        attributes = self.__dict__.keys()
        string_repr = "OutcomeMatrix"
        for attribute in attributes:
            string_repr += f"\n\n{attribute}:\n{self.__getattribute__(attribute)}"

        return string_repr

    @property
    def alive(self) -> OutcomeTable:
        """A table containing the number of people alive in each year, age, and sex."""
        return self._alive

    @alive.setter
    def alive(self, alive: OutcomeTable):
        self._alive = alive

    @property
    def antibiotic_exposure(self) -> OutcomeTable:
        """A table containing the number of rounds of antibiotics for each year, age, and sex."""
        return self._antibiotic_exposure

    @antibiotic_exposure.setter
    def antibiotic_exposure(self, antibiotic_exposure: OutcomeTable):
        self._antibiotic_exposure = antibiotic_exposure

    @property
    def asthma_incidence(self) -> OutcomeTable:
        """A table containing the number of new asthma diagnoses for each year, age, and sex."""
        return self._asthma_incidence

    @asthma_incidence.setter
    def asthma_incidence(self, asthma_incidence: OutcomeTable):
        self._asthma_incidence = asthma_incidence

    @property
    def asthma_prevalence(self) -> OutcomeTable:
        """A table containing the number of people with asthma for each year, age, and sex."""
        return self._asthma_prevalence

    @asthma_prevalence.setter
    def asthma_prevalence(self, asthma_prevalence: OutcomeTable):
        self._asthma_prevalence = asthma_prevalence

    @property
    def asthma_incidence_contingency_table(self) -> OutcomeTable:
        """TODO."""
        return self._asthma_incidence_contingency_table

    @asthma_incidence_contingency_table.setter
    def asthma_incidence_contingency_table(
        self, asthma_incidence_contingency_table: OutcomeTable
    ):
        self._asthma_incidence_contingency_table = asthma_incidence_contingency_table

    @property
    def asthma_prevalence_contingency_table(self) -> OutcomeTable:
        """TODO."""
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
        """A table containing the level of asthma control for each year, age, and sex."""
        return self._control

    @control.setter
    def control(self, control: OutcomeTable):
        self._control = control

    @property
    def cost(self) -> OutcomeTable:
        """A table containing the cost of asthma for each year, age, and sex."""
        return self._cost

    @cost.setter
    def cost(self, cost: OutcomeTable):
        self._cost = cost

    @property
    def death(self) -> OutcomeTable:
        """A table containing the number of people who died in a given year, age, and sex."""
        return self._death

    @death.setter
    def death(self, death: OutcomeTable):
        self._death = death

    @property
    def emigration(self) -> OutcomeTable:
        """A table containing the number of people who emigrated to Canada for each year, age,
        and sex."""
        return self._emigration

    @emigration.setter
    def emigration(self, emigration: OutcomeTable):
        self._emigration = emigration

    @property
    def exacerbation(self) -> OutcomeTable:
        """A table containing the number of asthma exacerbations for each year, age, and sex."""
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
        """A table containing the number of people who immigrated to Canada for each year, age,
        and sex."""
        return self._immigration

    @immigration.setter
    def immigration(self, immigration: OutcomeTable):
        self._immigration = immigration

    @property
    def utility(self) -> OutcomeTable:
        """A table containing the utility due to asthma for each year, age, and sex."""
        return self._utility

    @utility.setter
    def utility(self, utility: OutcomeTable):
        self._utility = utility

    def create_table(self, columns: list[str], group_by: list[str], *args) -> OutcomeTable:
        """Create an outcome table.

        Args:
            columns: The list of column names for the table.
            group_by: The list of column names to group by.
            args: TODO.
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

    def save(self, path: pathlib.Path):
        """Save the outcome matrix to ``*.csv`` files.

        Args:
            path: The path to the folder where the data will be saved.
        """
        if not path.exists():
            raise Exception(f"Path <{path}> does not exist.")

        attributes = [
            key for key, value in self.__dict__.items() if isinstance(value, OutcomeTable)
        ]
        for attribute in attributes:
            file_path = pathlib.Path(
                path.resolve(), f"outcome_matrix_{attribute.removeprefix('_')}.csv")
            self.__getattribute__(attribute).data.to_csv(file_path, index=False)
            logger.message(f"Saved {attribute.removeprefix('_')} to {file_path}.")
