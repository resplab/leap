Installation
============

If you plan on developing the ``LEAP`` package, you will need to install via the following steps:


1. Install ``R``
*******************

In order to work on the ``data_generation`` submodule, you will need to install ``R`` and the
``rpy2`` package. The ``rpy2`` package is a Python interface to ``R``, which allows you to run ``R``
code from Python.

MacOS
-----

Go to the `R for MacOS <https://cran.r-project.org/bin/macosx/>`_ page and download the latest
version of R.

Linux (Ubuntu/Debian)
---------------------

Follow the instructions on the
`R for Ubuntu <https://cran.r-project.org/bin/linux/ubuntu/fullREADME.html/>`_ page
to install the latest version of R.

.. code:: bash

  sudo apt-get update
  sudo apt-get install r-base
  sudo apt-get install r-base-dev


Windows
-------

Go to the `R for Windows <https://cran.r-project.org/bin/windows/base/>`_ page and download the
latest version of R.


1. Install ``pandoc``
*********************

If you plan on working on the ``docs``, you will also need to install ``pandoc``, as this is used
to convert ``Jupyter`` notebooks to ``rst`` files.

MacOS
-----

.. code:: bash

  brew install pandoc
  pandoc --version


Linux (Ubuntu/Debian)
---------------------

.. code:: bash

  wget https://github.com/jgm/pandoc/releases/download/3.6.4/pandoc-3.6.4-1-amd64.deb
  sudo dpkg -i pandoc-3.6.4-1-amd64.deb
  sudo apt-get install -f
  pandoc --version


Windows
-------

See the `pandoc installation instructions for Windows <https://pandoc.org/installing.html#windows>`_.

3. Install ``git-lfs``
***********************

MacOS
-----

See the 
`git-lfs installation instructions for MacOS 
<https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage?platform=mac>`_.

.. code:: bash

  brew install git-lfs
  git lfs install

Linux (Ubuntu/Debian)
---------------------

See the `git-lfs installation instructions for Linux
<https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage?platform=linux>`_.


Windows
-------

See the `git-lfs installation instructions for Windows 
<https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage?platform=windows>`_.


4. Install ``LEAP``
***********************

To install a development version of the ``LEAP`` package:

.. code:: bash

  git clone https://github.com/resplab/leap.git
  cd leap
  pip install -e ".[dev]"
  pip install -r requirements-docs.txt
  pip install -r requirements-data-generation.txt