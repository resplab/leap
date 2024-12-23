Sphinx Documentation
====================

The site you are currently viewing is generated using ``Sphinx`` documentation.
To modify the documentation, see the ``docs`` folder. To build the documentation:

.. code:: bash

  cd LEAP/docs
  make clean
  make html

This will create a ``build`` folder containing the ``HTML`` files. Open up the ``index.html`` file
in your browser to view the documentation.

autodoc
********

The ``autodoc`` extension automatically generates documentation from docstrings in the source code.
As you update the package, each time you build the documentation, the ``autodoc`` extension will
update the docstrings for existing modules/functions/classes accordingly. However, 
if you add a new module/function/class, you will need to rerun the initial ``autodoc`` command:

.. code:: bash

  cd LEAP/docs
  sphinx-apidoc -o ./dev/api/ ../leap/ --separate


Sphinx-Immaterial Theme
***********************

The documentation is styled using the `Sphinx-Immaterial`_ theme.

.. _Sphinx-Immaterial: https://sphinx-immaterial.readthedocs.io/en/stable/index.html
