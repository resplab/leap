import json
import argparse
import pathlib
from leap.simulation import Simulation
from leap.utils import check_file, get_data_path
from leap.logger import get_logger

logger = get_logger(__name__)


def get_parser() -> argparse.ArgumentParser:
    """Get the command line interface parser."""

    parser = argparse.ArgumentParser(add_help=False)
    command_group = parser.add_mutually_exclusive_group(required=False)
    command_group.add_argument(
        "-r", "--run", dest="run", action="store_true",
        help="Run the simulation."
    )

    args = parser.add_argument_group("ARGUMENTS")
    args.add_argument(
        "-c", "--config", dest="config", required=False, type=str,
        default=get_data_path("processed_data/config.json"),
        help="Path to configuration file."
    )
    args.add_argument(
        "-o", "--path-output", dest="path_output", required=True, type=str,
        help="Path to output file."
    )
    args.add_argument(
        "-ma", "--max-age", dest="max_age", required=False, type=int,
        help="Maximum age for agents in the model."
    )
    args.add_argument(
        "-my", "--min-year", dest="min_year", required=False, type=int,
        help="Starting year for the simulation."
    )
    args.add_argument(
        "-p", "--province", dest="province", required=False, type=str,
        help="Province for the simulation."
    )
    args.add_argument(
        "-th", "--time-horizon", dest="time_horizon", required=False, type=int,
        help="Time horizon for the simulation."
    )
    args.add_argument(
        "-nb", "--num-births-initial", dest="num_births_initial", required=False, type=int,
        help="Number of initial births for the simulation."
    )
    args.add_argument(
        "-gt", "--population-growth-type", dest="population_growth_type", required=False, type=str,
        help="Population growth type for the simulation."
    )
    args.add_argument(
        "-h", "--help", action="help", default=argparse.SUPPRESS,
        help="Shows function documentation."
    )

    return parser


def get_config(path_config: str) -> dict:
    """Get the configuration settings from a ``json`` file.

    Args:
        path_config: The path to the configuration file.

    Returns:
        A dictionary containing the configuration settings.
    """

    check_file(path_config, ext=".json")
    with open(path_config) as file:
        config = json.load(file)
    return config


def run_main():
    """The entry point for the command line interface."""

    parser = get_parser()
    args = parser.parse_args()
    config = get_config(args.config)

    logger.message(f"Config:\n{config}")

    simulation = Simulation(
        config=config,
        max_age=args.max_age,
        min_year=args.min_year,
        province=args.province,
        time_horizon=args.time_horizon,
        num_births_initial=args.num_births_initial,
        population_growth_type=args.population_growth_type
    )

    if args.run:
        logger.message("Running simulation...")
        outcome_matrix = simulation.run()
        logger.message(outcome_matrix)
        outcome_matrix.save(path=pathlib.Path(args.path_output))


if __name__ == "__main__":
    run_main()