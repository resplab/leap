Installation
============

To install a development version of the ``LEAP`` package:

.. code:: bash

  git clone https://github.com/resplab/leap.git
  cd leap
  pip install -e ".[dev]"


Installing ``pandoc``
*********************

If you plan on working on the ``docs``, you will also need to install ``pandoc``, as this is used
to convert ``Jupyter`` notebooks to ``rst`` files.

MacOS
-----

.. code:: bash

  brew install pandoc


Linux (Ubuntu/Debian)
---------------------

.. code:: bash

  wget https://github.com/jgm/pandoc/releases/download/3.6.4/pandoc-3.6.4-1-amd64.deb
  sudo dpkg -i pandoc-3.6.4-1-amd64.deb


Windows
-------

See the `pandoc installation instructions for Windows <https://pandoc.org/installing.html#windows>`_.