import json
import argparse
import pathlib
import pprint
import socket
import numpy as np
from datetime import datetime
from leap.simulation import Simulation
from leap.utils import check_file, get_data_path
from leap.logger import get_logger, set_logging_level
from leap.utils import convert_non_serializable
import warnings

warnings.filterwarnings("ignore")

logger = get_logger(__name__)
pretty_printer = pprint.PrettyPrinter(indent=2, sort_dicts=False)


def get_parser() -> argparse.ArgumentParser:
    """Get the command line interface parser."""

    parser = argparse.ArgumentParser(add_help=False, formatter_class=argparse.RawTextHelpFormatter)
    command_group = parser.add_mutually_exclusive_group(required=False)
    command_group.add_argument(
        "-r",
        "--run-simulation",
        dest="run",
        action="store_true",
        help="Run the simulation.",
    )

    args = parser.add_argument_group("ARGUMENTS")
    args.add_argument(
        "-c",
        "--config",
        dest="config",
        required=False,
        type=str,
        default=get_data_path("processed_data/config.json"),
        help="Path to configuration file.",
    )
    args.add_argument(
        "-p",
        "--province",
        dest="province",
        required=False,
        type=str,
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
        """,
    )
    args.add_argument(
        "-ma",
        "--max-age",
        dest="max_age",
        required=False,
        type=int,
        help="Maximum age for agents in the model.",
    )
    args.add_argument(
        "-my",
        "--min-year",
        dest="min_year",
        required=False,
        type=int,
        help="Starting year for the simulation. Must be >= 2024.",
    )
    args.add_argument(
        "-th",
        "--time-horizon",
        dest="time_horizon",
        required=False,
        type=int,
        help="Time horizon for the simulation.",
    )
    args.add_argument(
        "-gt",
        "--population-growth-type",
        dest="population_growth_type",
        required=False,
        type=str,
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
        """,
    )
    args.add_argument(
        "-nb",
        "--num-births-initial",
        dest="num_births_initial",
        required=False,
        type=int,
        help="Number of initial births for the simulation.",
    )
    args.add_argument(
        "-ip",
        "--ignore-pollution",
        dest="ignore_pollution",
        action="store_true",
        help="Do not include pollution as an element affecting the simulation.",
    )
    args.add_argument(
        "-o",
        "--path-output",
        dest="path_output",
        required=False,
        type=str,
        help="Path to output file.",
    )
    args.add_argument(
        "-f",
        "--force",
        dest="force",
        action="store_true",
        help="Stores outputs at provided or default destination, without prompting for confirmation.",
    )
    args.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Print all the output.",
    )
    args.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Shows function documentation.",
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


def handle_output_path(dir_name: str) -> pathlib.Path | None:
    """Provides path for output data by handling user input through CLI prompts.
    
    - Assuming ``WORKSPACE`` is directory where ``leap`` was called,
      then ``WORKSPACE/leap_output/dir_name`` is checked
    - If that path exists then user is prompted to continue (overwriting the current outputs)
    - If that path doesn't exist then user is prompted whether to create it

    Args:
        dir_name: The name of the directory to store the outputs in.

    Returns:
        Either the path to the output folder, or ``None``, signifying to abort

    Examples:

        (assuming ``/home/user/WORKSPACE`` is the currect working directory)

        .. code-block:: python

            handle_output_path('mydir1')
            # assuming the user confirmed to create mydir1
            > '/home/user/WORKSPACE/leap_output/mydir1'

            handle_output_path('mydir1')
            # assuming the user confirmed to continue with existing mydir1
            > '/home/user/WORKSPACE/leap_output/mydir1'

            handle_output_path('mydir1')
            # assuming the user did not confirm to continue with existing mydir1
            > None

            handle_output_path('mydir2')
            # assuming the user did not confirm to create mydir2
            > None
    """

    # pathlib automatically prefixes the path with the current working directory
    output_path = pathlib.Path("leap_output", dir_name)

    # Prompt user to continue with existing path or quit
    if output_path.exists():
        logger.message(f"Path <{output_path.absolute()}> already exists.")
        logger.message(
            f"Are you sure you would like to continue (WARNING THIS WILL OVERWRITE EXISTING RESULT CSV FILES)?"
        )
        path_msg = f"""
          - type y for to overwrite files located at <{output_path.absolute()}>
          - type n to stop
        """
        response = input(path_msg).strip().lower()
        # Only need to check if response is 'y' since any other response will quit
        if not response == "y":
            return None
    # Prompt user to create directory or quit
    else:
        logger.message(f"Path <{output_path.absolute()}> does not exist.")
        logger.message(f"Would you like to create a directory?")
        path_msg = f"""
          - type y for to create directory <{output_path.absolute()}>
          - type n to quit
        """
        response = input(path_msg).strip().lower()
        if response == "y":
            # Create directory and continue
            output_path.mkdir(parents=True, exist_ok=True)
            logger.message(f"Directory created at <{output_path.absolute()}>")
        else:
            return None

    # Return output_path if successful and continuing with program
    return output_path


def force_output_path(dir_name: str) -> pathlib.Path:
    """Provides path for output data without user input.
    
    - Assuming ``WORKSPACE`` is the directory where ``leap`` was called,
      then ``WORKSPACE/leap_output/dir_name`` is checked
    - If that path exists then that dir is used (overwriting any existing data)
    - If that path doesn't exist then is created and used

    Args:
        dir_name: The name of the directory to store the outputs in

    Returns:
        Either the path to the output folder or ``None``, signifying to abort

    Examples:

        (assuming ``/home/user/WORKSPACE`` is the currect working directory)

        .. code-block:: python

            force_output_path('mydir')
            > '/home/user/WORKSPACE/leap_output/mydir'
    """

    # Use resolve() to normalize the path and make it absolute
    output_path = pathlib.Path("leap_output", dir_name).resolve()

    # Prompt user to continue with existing path or quit
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    # Return output_path if successful and continuing with program
    return output_path


def run_main():
    """The entry point for the command line interface."""

    parser = get_parser()
    args = parser.parse_args()
    config = get_config(args.config)
    if args.verbose:
        set_logging_level(20)

    logger.message(f"Initializing simulation object.")
    # Create simulation object using arguments
    simulation = Simulation(
        config=config,
        province=args.province,
        max_age=args.max_age,
        min_year=args.min_year,
        time_horizon=args.time_horizon,
        population_growth_type=args.population_growth_type,
        num_births_initial=args.num_births_initial,
        ignore_pollution_flag=args.ignore_pollution,
    )
    logger.message(f"Simulation object initialized.")

    # Check if path_output argument is supplied or not
    if args.path_output is None or args.path_output == "":
        # Default dir name based on simulation arguments
        dir_name = f"{simulation.province}-{simulation.max_age}-{simulation.min_year}-{simulation.time_horizon}-{simulation.population_growth_type}-{simulation.num_births_initial}"
    else:
        # Get the name of the dir to store outputs in
        dir_name = args.path_output

    # If --force flag is given, then use dir name regardless of user input
    if args.force:
        output_path = force_output_path(dir_name)
        logger.message(f"--force flag included, so output directory not checked.")
    else:
        # Prompt user with CLI instructions to handle output path
        output_path = handle_output_path(dir_name)

    # If output_path is None and user decides to quit, then exit the program
    if output_path is None:
        logger.error("Aborting\n")
        quit()

    # Get start time of simulation
    simulation_start_time = datetime.now()

    if args.run:
        logger.message(f"Results will be saved to <{output_path.absolute()}>")
        logger.message("Running simulation...")

        # This is the main function that runs the entire simulation
        outcome_matrix = simulation.run()

        logger.message(outcome_matrix)
        outcome_matrix.save(path=output_path)

    # Get end time of simulation
    simulation_end_time = datetime.now()

    # Include log file containing additional information
    # Get the current timestamp
    current_date = datetime.now().strftime("%Y-%m-%d")
    # Define the file name
    log_file_path = output_path.joinpath("log.json")

    # Get text to include in the logfile
    # Create dict to store metadata info
    metadata = {
        "hostname": socket.gethostname(),
        "simulation_bundle_name": str(dir_name),
        "simulation_run_date": current_date,
        "simulation_start_time": str(simulation_start_time),
        "simulation_end_time": str(simulation_end_time),
        "simulation_runtime": str(simulation_end_time - simulation_start_time),
    }
    # Create dict to store parameter info
    parameters = {
        "province": simulation.province,
        "max_age": simulation.max_age,
        "min_year": simulation.min_year,
        "time_horizon": simulation.time_horizon,
        "population_growth_type": simulation.population_growth_type,
        "num_births_initial": simulation.num_births_initial,
        "pollution ignored": args.ignore_pollution,
        "max_year": simulation.max_year,
    }

    # Combine metadata and parameters into the log message
    log_data = {"metadata": metadata, "parameters": parameters, "config": config}
    # json.dumps lays out the data in a nice indented format for the log file
    log_msg = json.dumps(log_data, indent=4, default=convert_non_serializable, ensure_ascii=False)

    with open(log_file_path, "w") as file:
        # Write message to logfile
        file.write(log_msg)


if __name__ == "__main__":
    run_main()
