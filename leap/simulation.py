import pathlib
import json
import math
import numpy as np
import pandas as pd
from leap.agent import Agent
from leap.antibiotic_exposure import AntibioticExposure
from leap.birth import Birth
from leap.census_division import CensusTable
from leap.control import Control
from leap.cost import AsthmaCost
from leap.death import Death
from leap.emigration import Emigration
from leap.exacerbation import Exacerbation
from leap.family_history import FamilyHistory
from leap.immigration import Immigration
from leap.occurrence import Incidence, Prevalence, agent_has_asthma
from leap.outcome_matrix import OutcomeMatrix
from leap.pollution import PollutionTable
from leap.reassessment import Reassessment
from leap.severity import ExacerbationSeverity
from leap.utility import Utility
from leap.utils import PROCESSED_DATA_PATH
from leap.logger import get_logger

logger = get_logger(__name__)


class Simulation:
    """A class containing information about the simulation.
    """
    def __init__(self, config: dict | None = None):
        if config is None:
            with open(pathlib.Path(PROCESSED_DATA_PATH, "config.json")) as file:
                config = json.load(file)

        self.max_age = config["simulation"]["max_age"]
        self.province = config["simulation"]["province"]
        self.min_year = config["simulation"]["min_year"]
        self.max_year = self.min_year + config["simulation"]["time_horizon"] - 1
        self.time_horizon = config["simulation"]["time_horizon"]
        self.num_births_initial = config["simulation"]["num_births_initial"]
        self.population_growth_type = config["simulation"]["population_growth_type"]
        self.agent = None
        self.birth = Birth(self.min_year, self.province, self.population_growth_type, self.max_age)
        self.emigration = Emigration(self.min_year, self.province, self.population_growth_type)
        self.immigration = Immigration(
            self.min_year, self.province, self.population_growth_type, self.max_age
        )
        self.death = Death(config["death"], self.province, self.min_year)
        self.incidence = Incidence(config["incidence"])
        self.prevalence = Prevalence(config["prevalence"])
        self.reassessment = Reassessment(self.min_year, self.province)
        self.control = Control(config["control"])
        self.exacerbation = Exacerbation(config["exacerbation"], self.province)
        self.exacerbation_severity = ExacerbationSeverity(config["exacerbation_severity"])
        self.antibiotic_exposure = AntibioticExposure(config["antibiotic_exposure"])
        self.family_history = FamilyHistory(config["family_history"])
        self.utility = Utility(config["utility"])
        self.cost = AsthmaCost(config["cost"])
        self.census_table = CensusTable(config["census_table"])
        self.pollution_table = PollutionTable()
        self.SSP = config["pollution"]["SSP"]
        self.outcome_matrix = None

    def get_num_new_agents(
        self, year: int, min_year: int, num_new_born: int, num_immigrants: int
    ) -> int:
        """Get the number of new agents born/immigrated in a given year.

        Args:
            year (int): The calendar year of the current iteration, e.g. 2027.
            min_year (int): The calendar year of the initial iteration, e.g. 2010.
            num_new_born (int): The number of babies born in the specified year.
            num_immigrants (int): The number of immigrants who moved to Canada in the
                specified year.
        """
        # for the first/initial year, we generate the initial population
        # otherwise we generate num_new_born + num_immigrants

        df = self.birth.initial_population

        if year == min_year:
            num_new_agents = int(math.ceil(
                num_new_born / np.sum(
                    df[df["age"] == 0]["prop"]
                )
            ))
            initial_pop_indices = self.birth.get_initial_population_indices(self.num_births_initial)
            num_new_agents = len(initial_pop_indices)
        else:
            num_new_agents = num_new_born + num_immigrants
        
        return num_new_agents

    def get_new_agents(self, year: int, year_index: int) -> pd.DataFrame:
        """Get the new agents born/immigrated in a given year.

        Args:
            year (int): The year.
            year_index (int): The index of the year.

        Returns:
            pd.DataFrame: The new agents.
        """
        # number of newborns and immigrants in a year
        num_new_born = self.birth.get_num_newborn(self.num_births_initial, year_index)
        num_immigrants = self.immigration.get_num_new_immigrants(num_new_born, year_index)
        num_new_agents = self.get_num_new_agents(
            year, self.min_year, num_new_born, num_immigrants
        )

        if year == self.min_year:
            initial_pop_indices = self.birth.get_initial_population_indices(self.num_births_initial)
            initial_population_df = self.birth.initial_population.iloc[initial_pop_indices]
            new_agents_df = pd.DataFrame({
                "age": list(initial_population_df["age"]),
                "sex": list(np.random.binomial(1, np.array(initial_population_df["prop_male"]))),
                "immigrant": [False] * num_new_agents
            })
        else:
            immigrant_indices = list(np.random.choice(
                a=range(self.immigration.table.get_group((year)).shape[0]),
                size=num_immigrants,
                p=list(self.immigration.table.get_group((year))["prop_immigrants_year"])
            ))
            new_born_indices = list(range(num_immigrants + 1, num_new_agents))
            immigrant_df = self.immigration.table.get_group((year)).iloc[immigrant_indices]
            new_born_df = self.birth.estimate.get_group((year)).iloc[new_born_indices]
            sexes_immigrant = immigrant_df["sex"].astype(bool).tolist()
            ages_immigrant = immigrant_df["age"].tolist()
            sexes_birth = list(np.random.binomial(1, np.array(new_born_df["prop_male"])))
            ages_birth = list(new_born_df["age"])
            sexes = sexes_immigrant + sexes_birth
            ages = ages_immigrant + ages_birth
            new_agents_df = pd.DataFrame({
                "age": ages,
                "sex": sexes,
                "immigrant": [True] * num_immigrants + [False] * num_new_born
            })
        return new_agents_df

    def generate_initial_asthma(self, agent: Agent):
        """Generate the initial asthma status for an agent.

        Args:
            agent (Agent): The agent.
        """
        agent.has_asthma = self.prevalence.compute_has_asthma(agent)

        if agent.has_asthma:
            agent.asthma_status = True
            agent.asthma_age = self.incidence.compute_asthma_age(
                agent, self.prevalence, agent.age
            )
            agent.total_hosp = self.exacerbation.compute_hospitalization_prob(
                agent, self.exacerbation_severity, self.control, self.exacerbation
            )
            agent.control_levels = self.control.compute_control_levels(
                self.control, agent.sex, agent.age, True
            )
            agent.exacerbation_history.num_current_year = self.exacerbation.compute_num_exacerbations(
                agent=agent, initial=True
            )
            # the number of exacerbations by severity
            agent.exacerbation_severity_history.current_year = self.exacerbation.compute_distribution_exac_severity(
                self.exacerbation_severity, agent.exacerbation_history.num_current_year,
                (agent.total_hosp > 0), agent.age
            )
            # update total hospitalizations
            agent.total_hosp += agent.exacerbation_severity_history.current_year[3]

    def reassess_asthma_diagnosis(self, agent: Agent, outcome_matrix: OutcomeMatrix):
        """Reassess if the agent has asthma.

        Args:
            agent (Agent): The agent.
            outcome_matrix (OutcomeMatrix): The outcome matrix.
        """
        agent.has_asthma = self.reassessment.agent_has_asthma(agent)

        if agent.has_asthma:
            agent.exacerbation_history.num_prev_year = agent.exacerbation_history.num_current_year
            agent.exacerbation_severity_history.prev_year = agent.exacerbation_severity_history.current_year
            self.update_asthma_effects(agent, outcome_matrix)

    def update_asthma_effects(self, agent: Agent, outcome_matrix: OutcomeMatrix):
        """Update the asthma effects for an agent.

        Args:
            agent (Agent): The agent.
            outcome_matrix (OutcomeMatrix): The outcome matrix.
        """

        agent.control_levels = self.control.compute_control_levels(
            agent.sex, agent.age
        )
        for level in range(3):
            outcome_matrix.control.increment(
                column="prob",
                filter_columns={"year": agent.year, "level": level, "age": agent.sex},
                amount=agent.control_levels[level]
            )

        agent.exacerbation_history.num_current_year = self.exacerbation.compute_num_exacerbations(
            agent=agent
        )

        if agent.exacerbation_history.num_current_year != 0:
            agent.exacerbation_severity_history.current_year = self.exacerbation_severity.compute_distribution(
                agent.exacerbation_history.num_current_year,
                agent.total_hosp > 0,
                agent.age
            )
            agent.total_hosp += agent.exacerbation_severity_history.current_year[3]
            outcome_matrix.exacerbation.increment(
                column="n_exacerbations",
                filter_columns={"year": agent.year, "age": agent.age, "sex": agent.sex},
                amount=agent.exacerbation_history.num_current_year
            )
            outcome_matrix.exacerbation_hospital.increment(
                column="n_hospitalizations",
                filter_columns={"year": agent.year, "age": agent.age, "sex": agent.sex},
                amount=agent.exacerbation_severity_history.current_year[3]
            )
            for level in range(4):
                outcome_matrix.exacerbation_by_severity.increment(
                    column="p_exacerbations",
                    filter_columns={
                        "year": agent.year, "age": agent.age, "sex": agent.sex, "severity": level
                    },
                    amount=agent.exacerbation_severity_history.current_year[level]
                )

    def check_if_agent_gets_new_asthma_diagnosis(self, agent: Agent, outcome_matrix: OutcomeMatrix):
        """Check if the agent gets a new asthma diagnosis.

        If the agent does not have asthma, check to see if they get a new diagnosis this year.
        Mutates both the ``agent`` and ``outcome_matrix`` arguments.

        Args:
            agent (Agent): The agent.
            outcome_matrix (OutcomeMatrix): The outcome matrix.
        """
        agent.has_asthma = agent_has_asthma(
            agent, occurrence_type="inc", incidence=self.incidence, prevalence=self.prevalence
        )

        if agent.has_asthma:
            # if they did not have asthma dx in the past, then record it
            agent.asthma_age = agent.age
            outcome_matrix.asthma_incidence.increment(
                column="asthma_incidence",
                filter_columns={"year": agent.year, "age": agent.age, "sex": agent.sex}
            )
            self.update_asthma_effects(agent, outcome_matrix)

            # keep track of patients who got asthma for the first time
            if not agent.asthma_status:
                agent.asthma_status = True
                outcome_matrix.asthma_status.increment(
                    column="asthma_status",
                    filter_columns={"year": agent.year, "age": agent.age, "sex": agent.sex}
                )

        outcome_matrix.asthma_incidence_contingency_table.increment(
            column="n_asthma" if agent.has_asthma else "n_no_asthma",
            filter_columns={
                "year": agent.year, "age": agent.age, "sex": agent.sex,
                "fam_history": agent.has_family_hist,
                "abx_exposure": agent.num_antibiotic_use,
            }
        )

    def run(self, seed=None, until_all_die: bool = False, verbose: bool = True):
        if seed is not None:
            np.random.seed(seed)

        month = 1
        max_age = self.max_age
        min_year = self.min_year
        max_year = self.max_year

        max_time_horizon = np.iinfo(np.int32).max if until_all_die else self.time_horizon
        years = np.arange(min_year, max_year + 1)
        total_years = max_year - min_year + 1

        outcome_matrix = OutcomeMatrix(until_all_die, min_year, max_year, max_age)

        # loop by year
        for year in years:
            year_index = year - min_year

            new_agents_df = self.get_new_agents(
                year=year,
                year_index=year_index
            )

            logger.info(f"Year {year}, year {year_index} of {total_years} years, "
                        f"{new_agents_df.shape[0]} new agents born/immigrated.")
            logger.info(new_agents_df)

            # for each agent i born/immigrated in year
            for i in range(new_agents_df.shape[0]):
                self.control.assign_random_β0()
                self.exacerbation.assign_random_β0()
                self.exacerbation_severity.assign_random_p()

                agent = Agent(
                    sex=new_agents_df["sex"].iloc[i],
                    age=new_agents_df["age"].iloc[i],
                    year=year,
                    year_index=year_index,
                    family_hist=self.family_history,
                    antibiotic_exposure=self.antibiotic_exposure,
                    province=self.province,
                    month=month,
                    SSP=self.SSP,
                    census_table=self.census_table,
                    pollution_table=self.pollution_table
                )

                logger.info(
                    f"Agent {agent.uuid.short} born/immigrated in year {year}, "
                    f"age {agent.age}, sex {int(agent.sex)}, "
                    f"immigrant: {new_agents_df['immigrant'].iloc[i]}, "
                    f"newborn: {not new_agents_df['immigrant'].iloc[i]}"
                )
                logger.info(
                    f"| -- Year: {agent.year_index + min_year - 1}, "
                    f"age: {agent.age}"
                )

                if new_agents_df["immigrant"].iloc[i]:
                    outcome_matrix.immigration.increment(
                        "n_immigrants", {"year": agent.year, "age": agent.age, "sex": agent.sex}
                    )

                outcome_matrix.antibiotic_exposure.increment(
                    column="antibiotic_exposure",
                    filter_columns={"year": agent.year, "age": agent.age, "sex": agent.sex},
                    amount=agent.num_antibiotic_use
                )

                outcome_matrix.family_history.increment(
                    column="has_family_history",
                    filter_columns={"year": agent.year, "age": agent.age, "sex": agent.sex},
                    amount=agent.has_family_hist
                )

                # if age > 4, we need to generate the initial distribution of asthma related events

                if agent.age > 3:
                    self.generate_initial_asthma(agent)

                    logger.info(
                        f"Agent age > 3, agent has asthma (prevalence)? {agent.has_asthma}"
                    )

                # go through event processes for each agent
                while agent.alive and agent.age <= max_age and agent.year_index <= max_time_horizon:
                    if not agent.has_asthma:
                        self.check_if_agent_gets_new_asthma_diagnosis(agent, outcome_matrix)
                        logger.info(f"Agent has asthma (incidence)? {agent.has_asthma}")
                    else:
                        self.reassess_asthma_diagnosis(agent, outcome_matrix)
                        logger.info(
                            "Agent was diagnosed with asthma, is this diagnosis correct? "
                            f"{agent.has_asthma}"
                        )

                    # if no asthma, record it
                    if agent.has_asthma:
                        outcome_matrix.asthma_prevalence.increment(
                            column="n_asthma",
                            filter_columns={"year": agent.year, "age": agent.age, "sex": agent.sex},
                        )

                    outcome_matrix.asthma_prevalence_contingency_table.increment(
                        column="n_asthma" if agent.has_asthma else "n_no_asthma",
                        filter_columns={
                            "year": agent.year, "age": agent.age, "sex": agent.sex,
                            "fam_history": agent.has_family_hist,
                            "abx_exposure": agent.num_antibiotic_use
                        }
                    )

                    # compute utility
                    utility = self.utility.compute_utility(agent)
                    logger.info(f"Utility of asthma: {utility}")
                    outcome_matrix.utility.increment(
                        column="utility",
                        filter_columns={"year": agent.year, "age": agent.age, "sex": agent.sex},
                        amount=utility
                    )

                    # compute cost
                    cost = self.cost.compute_cost(agent)
                    logger.info(f"Cost of asthma: {cost} CAD")

                    outcome_matrix.cost.increment(
                        column="cost",
                        filter_columns={"year": agent.year, "age": agent.age, "sex": agent.sex},
                        amount=cost
                    )

                    # death or emigration, assume death occurs first
                    if self.death.agent_dies(agent):
                        agent.alive = False
                        outcome_matrix.death.increment(
                            column="n_deaths",
                            filter_columns={"year": agent.year, "age": agent.age, "sex": agent.sex}
                        )
                        logger.info(f"Agent has died at age {agent.age}")
                    # emigration
                    elif self.emigration.compute_prob_emigration(
                        agent.year_index, agent.age, agent.sex
                    ):
                        agent.alive = False
                        outcome_matrix.emigration.increment(
                            column="n_emigrants",
                            filter_columns={"year": agent.year, "age": agent.age, "sex": agent.sex}
                        )
                        logger.info(f"Agent has emigrated at age {agent.age}")
                    else:
                        # record alive
                        outcome_matrix.alive.increment(
                            column="n_alive",
                            filter_columns={"year": agent.year, "age": agent.age, "sex": agent.sex}
                        )

                        # update the patient stats
                        agent.age += 1
                        agent.year += 1
                        agent.year_index += 1

                        if agent.age <= max_age and agent.year_index <= max_time_horizon:
                            logger.info(
                                f"| -- Year: {agent.year_index + min_year - 1}, age: {agent.age}"
                            )

        self.outcome_matrix = outcome_matrix
        logger.info("\n Simulation finished. Check your simulation object for results.")

        return outcome_matrix
