# Lifetime Exposures and Asthma outcomes Projection (LEAP)

## Installation

1. To install this package on your computer locally, first download it from `GitHub`:

```sh
git clone https://github.com/resplab/leap
```

2. Next, create a virtual environment:

```sh
cd leap
python -m venv env
source env/bin/activate
```

3. Finally, install the package using `pip` and the `-e` flag, which makes the package editable:

```sh
pip install -e .
```

### Testing

To run all the tests:

```sh
cd leap
pytest tests/
```

To run a single test file:

```sh
cd leap
pytest tests/test_name.py
```

To run doctests:

```sh
cd leap
pytest leap/ --doctest-modules
```

## Running the Simulation

The default simulation configuration options are found in the `LEAP/processed_data/config.json`
file.

To run the `LEAP` model from the command line, using the default settings:

```sh
leap --run-simulation
```

### Examples

To run the simulation for 1 year, starting in `2024`, with the maximum age of `4`,
and 10 new borns in the first year:

```sh
leap --run-simulation --time-horizon 1 --num-births-initial 10 --max-age 4 --min-year 2024 \
--path-output PATH/TO/OUTPUT
```

To specify the province and population growth scenario:

```sh
leap --run-simulation --time-horizon 1 --num-births-initial 10 --max-age 4 --province "CA" \
--min-year 2024 --population-growth-type "M3" --path-output PATH/TO/OUTPUT
```

If you would like to use your own `config.json` file instead of the default one:

```sh
leap --run-simulation --config PATH/TO/YOUR/CONFIG.json
```



