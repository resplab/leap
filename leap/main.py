import json
import argparse
import pathlib
import pprint
import sys
from leap.simulation import Simulation
from leap.utils import check_file, get_data_path
from leap.logger import get_logger, set_logging_level
import warnings
warnings.filterwarnings("ignore")

logger = get_logger(__name__)
pretty_printer = pprint.PrettyPrinter(indent=2, sort_dicts=False)


def get_parser() -> argparse.ArgumentParser:
    """Get the command line interface parser."""

    parser = argparse.ArgumentParser(add_help=False, formatter_class=argparse.RawTextHelpFormatter)
    command_group = parser.add_mutually_exclusive_group(required=False)
    command_group.add_argument(
        "-r", "--run-simulation", dest="run", action="store_true",
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
        help="Starting year for the simulation. Must be >= 2024."
    )
    args.add_argument(
        "-p", "--province", dest="province", required=False, type=str,
        help="""Province for the simulation. Must be one of:
        * "AB": Alberta
        * "BC": British Columbia
        * "MB": Manitoba
        * "NB": New Brunswick
        * "NL": Newfoundland and Labrador
        * "NS": Nova Scotia
        * "NT": Northwest Territories
        * "NU": Nunavut
        * "ON": Ontario
        * "PE": Prince Edward Island
        * "QC": Quebec
        * "SK": Saskatchewan
        * "YT": Yukon
        * "CA": Canada
        """
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
        help="""Population growth type for the simulation. Must be one of:
        * "past": past projection
        * "LG": low-growth projection
        * "HG": high-growth projection
        * "M1": medium-growth 1 projection
        * "M2": medium-growth 2 projection
        * "M3": medium-growth 3 projection
        * "M4": medium-growth 4 projection
        * "M5": medium-growth 5 projection
        * "M6": medium-growth 6 projection
        * "FA": fast-aging projection
        * "SA": slow-aging projection
        """
    )
    args.add_argument(
        "-v", "--verbose", dest="verbose", action="store_true",
        help="Print all the output."
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

    # Ensure user is running in virtural environment
    in_venv = sys.prefix != sys.base_prefix
    if not in_venv:
        raise Exception("Please run command while in the virtual environment.")

    parser = get_parser()
    args = parser.parse_args()
    config = get_config(args.config)
    if args.verbose:
        set_logging_level(20)

    # Check if path exists before running
    output_path = pathlib.Path(args.path_output)
    extended_output_path = pathlib.Path(*output_path.parts[:-1],
                                            "output",
                                            output_path.parts[-1])
    if extended_output_path.exists():
        logger.message(f"Path <{extended_output_path.absolute()}> already exists.")
        logger.message(f"Are you sure you would like to continue (WARNING THIS WILL OVERWRITE EXISTING RESULT CSV FILES)?")
        path_msg = f"""
          - type y for to overwrite files located at <{extended_output_path.absolute()}> 
          - type n to stop
        """
        response = input(path_msg).strip().lower()
        if response == 'n':
            quit()
    else:
        logger.message(f"Path <{output_path}> does not exist.")
        logger.message(f"Would you like to create a directory at:")
        path_msg = f"""
          - type 1 for <{output_path.absolute()}> 
          - type 2 for <{extended_output_path.absolute()}>
          - type 3 to quit
        """
        response = input(path_msg).strip().lower()
        if response == '1':
            output_path.mkdir(parents=True, exist_ok=True)
            logger.message(f"Directory created at <{output_path.absolute()}>")
        elif response == '2':
            extended_output_path.mkdir(parents=True, exist_ok=True)
            logger.message(f"Directory created at <{extended_output_path.absolute()}>")
        elif response == '3':
            quit()
        else:
            logger.error("Aborting\n")
            quit()

    # logger.message(f"Config:\n{pretty_printer.pformat(config)}")

    simulation = Simulation(
        config=config,
        max_age=args.max_age,
        min_year=args.min_year,
        province=args.province,
        time_horizon=args.time_horizon,
        num_births_initial=args.num_births_initial,
        population_growth_type=args.population_growth_type
    )

    logger.message(f"Results will be saved to <{extended_output_path}>")
    if args.run:
        logger.message("Running simulation...")
        outcome_matrix = simulation.run()
        logger.message(outcome_matrix)
        outcome_matrix.save(path=extended_output_path)


if __name__ == "__main__":
    run_main()
