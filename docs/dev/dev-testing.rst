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

The test suite is run automatically on GitHub Actions every time a commit is pushed. There are two
different test files:

1. `.github/workflows/tests_workflow.yml
<https://github.com/resplab/leap/.github/workflows/test_workflow.yml>`_

2. `.github/workflows/tests_workflow_lfs.yml
<https://github.com/resplab/leap/.github/workflows/test_workflow_lfs.yml>`_

The first file is run on every push, and the second file is run only if you tag the commit with
``test-lfs``. The reason for this is that the second file contains tests that rely on large files
that are stored using `Git LFS <https://git-lfs.github.com/>`_. Currently, the only file that is
used for testing is:

``leap/processed_data/census_divisions/census_division_boundaries/lcd_000b21a_e.shp``

However, this is only used for a few tests, which are quite separate from the rest of the simulation.
In general, you won't need to worry about this, and can just push your commits and the tests will
run automatically. 


Entire Test Suite with LFS (``test_workflow_lfs.yml``)
------------------------------------------------------
If you do need to run the tests with the LFS files, you can follow the instructions
below. This is useful if you are making changes to the tests that rely on the LFS files.

1. Push the commit that you want to tag:

.. code:: bash

  git add .
  git commit -m "my commit message"
  git push origin $MY_BRANCH

.. note::

  This will trigger the ``tests_workflow.yml`` to run.

2. Make sure the ``test-lfs`` tag has been deleted from any previous commits:

.. code:: bash

  git push --delete origin test-lfs
  git tag -d test-lfs

3. Add the ``test-lfs`` tag to the most recent commit and push the tag:

.. code:: bash

  git tag test-lfs
  git push origin test-lfs

.. note::
  
  This will trigger the ``tests_workflow_lfs.yml`` to run.


Adding New Tests
----------------

If you add a new test file, make sure to add it to the test workflow file. For example, if you add
the file ``tests/test_new_file.py``, you will need to add the following lines to the workflow files:

.. code:: yaml

  - name: Run Python tests new_file
    run: |
      pytest tests/test_new_file.py
    continue-on-error: false

The tests are run in alphabetical order, so place the new lines accordingly.

The reason the tests are run in separate steps instead of running them all via ``pytests tests/*``
is to allow the workflow to exit as soon as one of the test files fails. This uses less
computing resources, and allows for easier debugging.


Test Workflow Files
-------------------

.. code:: yaml

  # Tests Workflow: Main Test Suite (no LFS)
  # This workflow is triggered by a push to any branch with any tag except 'test-lfs' or 'v*'.
  # It runs the entire test suite, excluding tests that require large files stored in Git LFS.

  name: tests_workflow
  
  # execute this workflow automatically when a we push to any branch
  on:
    push:
      branches:
        - '**'
      tags-ignore:
        - v*
        - test-lfs
  
  jobs:
    tests:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
          with:
            python-version: '3.10'
        - name: Install dependencies for tests
          run: |
            pip install pytest
        - name: Install dependencies for docs
          run: |
            pip install -r requirements-docs.txt
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
            pytest tests/test_census_division.py -k "not test_census_boundaries"
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
        - name: Clean previous builds
          run: |
            rm -rf _build .tox
        - name: Test Sphinx build
          run: |
            sphinx-build docs _build -E -a

  concurrency:
    group: ci-${{ github.ref }}
    cancel-in-progress: true


.. code:: yaml

  # Tests Workflow: Entire Test Suite with LFS
  # This workflow is triggered by a push to any branch with the tag 'test-lfs'.
  # It runs the entire test suite, including tests that require large files stored in Git LFS.
  # To conserve LFS bandwidth, this test suite should only be run when necessary.

  name: tests_workflow_lfs
  
  # execute this workflow automatically on a push with tag 'test-lfs'
  on:
    push:
      tags:
        - 'test-lfs'
  
  jobs:
    tests:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
          with:
            lfs: true # enable Git LFS
        - uses: actions/setup-python@v5
          with:
            python-version: '3.10'
        - name: Install dependencies for tests
          run: |
            pip install pytest
        - name: Install dependencies for docs
          run: |
            pip install -r requirements-docs.txt
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
        - name: Clean previous builds
          run: |
            rm -rf _build .tox
        - name: Test Sphinx build
          run: |
            sphinx-build docs _build -E -a

  concurrency:
    group: ci-${{ github.ref }}
    cancel-in-progress: true


