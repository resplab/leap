Testing
=======

To run the test suite, from the ``leap`` root directory, type in a shell terminal:

.. code:: bash

  cd leap
  pytest tests/

To run a specific test file:

.. code:: bash

  pytest tests/test_simulation.py


To run a specific test in a file, use the ``-k`` flag:

.. code:: bash

  pytest tests/test_simulation.py -k generate_initial_asthma


Doctests
********

Doctests are examples within the function and class docstrings. To run these examples:

.. code:: bash

  cd leap
  pytest leap/ --doctest-modules


Test Data
*********

The test data is located in the ``tests/data`` directory.


GitHub Actions
***************

The test suite is run automatically on GitHub Actions every time a commit is pushed.
The configuration file is located at:
`.github/workflows/tests_workflow.yml
<https://github.com/resplab/leap/.github/workflows/test_workflow.yml>`_.


.. code:: yaml

  name: tests_workflow
  
  # execute this workflow automatically when a we push to any branch
  on: [push]
  
  jobs:
    tests:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
          with:
            lfs: true
        - uses: actions/setup-python@v5
          with:
            python-version: '3.10'
        - name: Install dependencies for tests
          run: |
            pip install pytest
        - name: Install LEAP
          run: |
            pip install .
        - name: Run Python tests agent
          run: |
            pytest tests/test_agent.py
          continue-on-error: false
        - name: Run Python tests antibiotic_exposure
          run: |
            pytest tests/test_antibiotic_exposure.py
          continue-on-error: false
        - name: Run Python tests birth
          run: |
            pytest tests/test_birth.py
          continue-on-error: false
        - name: Run Python tests census_division
          run: |
            pytest tests/test_census_division.py
          continue-on-error: false
        - name: Run Python tests control
          run: |
            pytest tests/test_control.py
        - name: Run Python tests cost
          run: |
            pytest tests/test_cost.py
          continue-on-error: false
        - name: Run Python tests death
          run: |
            pytest tests/test_death.py
          continue-on-error: false
        - name: Run Python tests emigration
          run: |
            pytest tests/test_emigration.py
          continue-on-error: false
        - name: Run Python tests exacerbation
          run: |
            pytest tests/test_exacerbation.py
          continue-on-error: false
        - name: Run Python tests family_history
          run: |
            pytest tests/test_family_history.py
          continue-on-error: false
        - name: Run Python tests immigration
          run: |
            pytest tests/test_immigration.py
          continue-on-error: false
        - name: Run Python tests occurrence
          run: |
            pytest tests/test_occurrence.py
          continue-on-error: false
        - name: Run Python tests outcome_matrix
          run: |
            pytest tests/test_outcome_matrix.py
          continue-on-error: false
        - name: Run Python tests pollution
          run: |
            pytest tests/test_pollution.py
          continue-on-error: false
        - name: Run Python tests reassessment
          run: |
            pytest tests/test_reassessment.py
          continue-on-error: false
        - name: Run Python tests severity
          run: |
            pytest tests/test_severity.py
          continue-on-error: false
        - name: Run Python tests simulation
          run: |
            pytest tests/test_simulation.py
          continue-on-error: false
        - name: Run Python tests utility
          run: |
            pytest tests/test_utility.py
          continue-on-error: false
        - name: Run doctests
          run: |
            pytest leap/ --doctest-modules
          continue-on-error: false
        - name: Test Sphinx build
          run: |
            sphinx-build docs _build -E -a

  concurrency:
    group: ci-${{ github.ref }}
    cancel-in-progress: true


If you add a new test file, make sure to add it to the test workflow file. For example, if you add
the file ``tests/test_new_file.py``, you will need to add the following lines to the workflow file:

.. code:: yaml

  - name: Run Python tests new_file
    run: |
      pytest tests/test_new_file.py
    continue-on-error: false

The tests are run in alphabetical order, so place the new lines accordingly.

The reason the tests are run in separate steps instead of running them all via ``pytests tests/*``
is to allow the workflow to exit as soon as one of the test files fails. This uses less
computing resources, and allows for easier debugging.