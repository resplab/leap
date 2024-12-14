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


