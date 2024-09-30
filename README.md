# Lifetime Exposures and Asthma outcomes Projection (LEAP)

## Developers

### Installation

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


