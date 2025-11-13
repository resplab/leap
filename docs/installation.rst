Installation
==============

Prerequisites
***************

- Administrator or ``sudo`` privileges (Linux/macOS).
- ``GitHub`` account created.
- ``git`` installed on your system.
- Ensure ``pip`` is installed.
- Internet access to download dependencies.


To get started, choose your operating system below:

1. :ref:`MacOS Installation <macos-installation>`
2. :ref:`Linux Installation <linux-installation>`
3. :ref:`Windows Installation <windows-installation>`


.. _macos-installation:
MacOS
******

Step 1: Install Python
----------------------------

If you already have ``Python`` installed, you can skip this step. To check your ``Python`` version,
open a terminal and type:


.. code:: bash

    python3 --version


If the output is ``command not found: python3``, or the version is lower than ``Python 3.9``,
then you will need to install Python.

To install Python, download the installer from:
`Mac Installer Python 3.10 <https://www.python.org/ftp/python/3.10.0/python-3.10.0post2-macos11.pkg>`_.

Once the install has finished, check to make sure that it's using the correct version:

.. code:: bash

  python3 --version
    # Expected output:
    # Python 3.10.*

Step 2: Choose Install Folder
-------------------------------

Create a directory to install ``LEAP``. In this example, we'll use ``my_folder``.
To create this directory, open a terminal and type:

.. code:: bash

    mkdir my_folder
    cd my_folder

This directory will serve as the workspace where you install and manage the LEAP package.

Step 3: Create a Virtual Environment
--------------------------------------

When working with Python, it's best to create a virtual environment to keep your packages
separate from the system Python. You will want to use your same Python version so if you
installed ``python3.10`` with the installer above, the command would be:

.. code:: bash

  python3.x -m venv env
  source env/bin/activate

where ``python3.x`` is your Python version. For example, if you installed Python with the installer
above, you installed ``python3.10``, so the command would be:

.. code:: bash

    python3.10 -m venv env
    source env/bin/activate

Step 4: (Optional) Setup Git Token
-----------------------------------

If you haven't used ``git`` on the command line before, you will need to set up a personal access
token. This is so you can install the ``LEAP`` package from ``GitHub``. To do this:

1. Go to `GitHub Settings: Tokens <https://github.com/settings/tokens>`_ to create a personal
   access token.
2. Click on ``Tokens: classic`` and then ``Generate new token (classic)``.
3. Set the expiration to ``No expiration`` so you don't have to do this again. If you want more
   security, set an earlier date.
4. Tick all the boxes available, and click ``Generate token``.
5. Copy this token to a secure file on your system. **You won't be shown it again.**
6. In your terminal, type:

.. code:: bash

    git config --global credential.helper osxkeychain


Step 5: Install ``LEAP``
-------------------------------

To install the ``LEAP`` package, type:

.. code-block:: bash

    pip3 install git+https://github.com/resplab/leap.git


To install a specific release, append ``@vx.y.z`` at the end:

.. code-block:: bash

    pip3 install git+https://github.com/resplab/leap.git@vx.y.z

where ``vx.y.z`` is the release version you would like to install. To see all the releases, go to:
`LEAP Releases <https://github.com/resplab/leap/releases>`_.


*(Optional)* If you set up your ``git`` token in the last step, you will now be asked for you
username and password. Paste the token in as your password:

.. code:: bash

  Username for 'https://github.com': YOUR_GIT_USERNAME # not your email address
  Password: YOUR_TOKEN # not your GitHub password
  


Step 6: Install Success!
-------------------------

To verify the installation was a success, type:

.. code-block:: bash

    leap --help

After running ``leap --help``, you should see:

.. code-block:: bash

    usage: leap [-r] [-c CONFIG] [-p PROVINCE] [-ma MAX_AGE] [-my MIN_YEAR] [-th TIME_HORIZON]
    [-gt POPULATION_GROWTH_TYPE] [-nb NUM_BIRTHS_INITIAL] [-ip] [-o PATH_OUTPUT] [-f] [-v] [-h]

    options:
      -r, --run-simulation  Run the simulation.

    ARGUMENTS:
      ...

Now that you've successfully installed the ``LEAP`` package, you can start using it. To get
started, see the :doc:`Command-Line Interface <cli/index>` documentation.

To get out of the virtual environment:

.. code:: bash

  deactivate


... _linux-installation:
Linux
******

Step 1: Install Python
----------------------------

If you already have ``Python`` installed, you can skip this step. To check your ``Python`` version,
open a terminal and type:


.. code:: bash

  python3 --version


If the output is ``command not found: python3``, or the version is lower than ``Python 3.9``,
then you will need to install Python.

To install Python, use your package manager. For example, on Ubuntu, type:

.. code-block:: bash

    sudo apt update
    sudo apt install python3.10

Once the install has finished, check to make sure that it's using the correct version:

.. code:: bash

  python3 --version
  # Expected output:
  # Python 3.10.*

Step 2: Choose Install Folder
-------------------------------

You can install ``LEAP`` in any folder. For this example, we'll use ``my_folder``. To
create this folder, open a terminal and type:

.. code:: bash

  mkdir my_folder
  cd my_folder

Step 3: Create a Virtual Environment
--------------------------------------

When working with Python, it's best to create a virtual environment to keep your packages separate
from the system Python. To create a virtual environment, type:

.. code:: bash

  python3.x -m venv env
  source env/bin/activate

where ``python3.x`` is your Python version. For example, if you installed Python with the installer
above, you installed ``python3.10``, so the command would be:

.. code:: bash

  python3.10 -m venv env
  source env/bin/activate


Step 4: (Optional) Setup Git Token
-----------------------------------

If you haven't used ``git`` on the command line before, you will need to set up a personal access
token. This is so you can install the ``LEAP`` package from ``GitHub``. To do this:

1. Go to `GitHub Settings: Tokens <https://github.com/settings/tokens>`_ to create a personal
   access token.
2. Click on ``Tokens: classic`` and then ``Generate new token (classic)``.
3. Set the expiration to ``No expiration`` so you don't have to do this again. If you want more
   security, set an earlier date.
4. Tick all the boxes available, and click ``Generate token``.
5. Copy this token to a secure file on your system. **You won't be shown it again.**
6. In your terminal, either type:

.. code-block:: bash

    git config --global credential.helper cache

Use this for temporary storage of credentials in memory (default 15 minutes). Add a note about
setting a custom timeout with ``cache --timeout=3600`` for longer sessions.

.. code-block:: bash

    git config --global credential.helper store

Use this for persistent storage of credentials in plain text.

.. warning::

    This command should only be used on trusted machines.
    Add a security warning that credentials are stored in ``~/.git-credentials`` 


Step 5: Install ``LEAP``
-------------------------------

To install the ``LEAP`` package, type:

.. code:: bash

  pip3 install git+https://github.com/resplab/leap.git


To install a specific release:

.. code:: bash

  pip3 install git+https://github.com/resplab/leap.git@vx.y.z

where ``vx.y.z`` is the release version you would like to install. To see all the releases, go to:
`LEAP Releases <https://github.com/resplab/leap/releases>`_.


*(Optional)* If you set up your ``git`` token in the last step, you will now be asked for you
username and password. Paste the token in as your password:

.. code:: bash

  Username for 'https://github.com': YOUR_GIT_USERNAME # not your email address
  Password: YOUR_TOKEN # not your GitHub password

Step 6: Install Success!
-------------------------

To verify the installation was a success, type:

.. code-block:: bash

  leap --help

After running ``leap --help``, you should see:

.. code-block:: bash

  usage: leap [-r] [-c CONFIG] [-p PROVINCE] [-ma MAX_AGE] [-my MIN_YEAR] [-th TIME_HORIZON]
  [-gt POPULATION_GROWTH_TYPE] [-nb NUM_BIRTHS_INITIAL] [-ip] [-o PATH_OUTPUT] [-f] [-v] [-h]

  options:
    -r, --run-simulation  Run the simulation.

  ARGUMENTS:
    ...

Now that you've successfully installed the ``LEAP`` package, you can start using it. To get
started, see the :doc:`Command-Line Interface <cli/index>` documentation.

To get out of the virtual environment:

.. code:: bash

  deactivate


.. _windows-installation:
Windows
********

Step 1: Install Python
----------------------------

If you already have ``Python`` installed, you can skip this step. To check your ``Python`` version,
open a terminal and type:


.. code:: bash

  python3 --version


If the output is ``command not found: python3``, or the version is lower than ``Python 3.9``,
then you will need to install Python.

To install Python, download the installer from:
`Windows Installer Python 3.10
<https://www.python.org/downloads/release/python-31016/>`_.

.. note::

  For other distributions, refer to your distribution's documentation.

Once the installation has finished, check to make sure that it's using the correct version:

.. code-block:: bash

   python3 --version
   # Expected output:
   # Python 3.10.*

Step 2: Choose Install Folder
--------------------------------------

Create a directory to install ``LEAP``. In this example, we'll use ``my_folder``.
To create this directory, open a terminal and type:

.. code-block:: bash

    mkdir my_folder
    cd my_folder

This directory will serve as the workspace where you install and manage the LEAP package.

Step 3: Create a Virtual Environment
****************************************

When working with Python, it's best to create a virtual environment to keep your packages
separate from the system Python. 

Windows Command Prompt
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  python -m venv env
  env\Scripts\activate

Windows PowerShell
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python -m venv env
    .\env\Scripts\Activate.ps1

.. warning::

    For PowerShell, you may need to allow scripts temporarily:

    If you get an error, run ``Set-ExecutionPolicy Unrestricted -Scope Process``
    before running ``.\env\Scripts\Activate.ps1``


Step 4: (Optional) Setup Git Token
****************************************

If you haven't used ``git`` on the command line before, you will need to set up a personal access
token. This is so you can install the ``LEAP`` package from ``GitHub``. To do this:

1. Go to `GitHub Settings: Tokens <https://github.com/settings/tokens>`_ to create a personal
   access token.
2. Click on ``Tokens: classic`` and then ``Generate new token (classic)``.
3. Set the expiration to ``No expiration`` so you don't have to do this again. If you want more
   security, set an earlier date.
4. Tick all the boxes available, and click ``Generate token``.
5. Copy this token to a secure file on your system. **You won't be shown it again.**
6. In your terminal, type:

.. code-block:: bash

    git config --global credential.helper manager

7. Test your setup by cloning a private repository or running a Git command like:

.. code-block:: bash

    git ls-remote https://github.com/YOUR_USERNAME/YOUR_PRIVATE_REPO.git

    # Replace YOUR_USERNAME and YOUR_PRIVATE_REPO with your GitHub username and a private repository name.

*(Optional)* If you encounter issues, ensure that Git is installed on your system. You can download
it from: `Git Downloads <https://git-scm.com/downloads>`_.

Step 5: Install ``LEAP``
-------------------------------

To install the ``LEAP`` package, type:

Windows Command Prompt
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  python -m pip install git+https://github.com/resplab/leap.git

Windows PowerShell
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

  python -m pip install git+https://github.com/resplab/leap.git


To install a specific release:

.. code-block:: bash

  python -m pip install git+https://github.com/resplab/leap.git@vx.y.z

where ``vx.y.z`` is the release version you would like to install. To see all the releases, go to:
`LEAP Releases <https://github.com/resplab/leap/releases>`_.


*(Optional)* If you set up your ``git`` token in the last step, you will now be asked for you
username and password. Paste the token in as your password:

.. code:: bash

  Username for 'https://github.com': YOUR_GIT_USERNAME # not your email address
  Password: YOUR_TOKEN # not your GitHub password


Step 6: Install Success!
-------------------------

To verify the installation was a success, type:

.. code-block:: bash

  leap --help

After running ``leap --help``, you should see:

.. code-block:: bash

  usage: leap [-r] [-c CONFIG] [-p PROVINCE] [-ma MAX_AGE] [-my MIN_YEAR] [-th TIME_HORIZON]
  [-gt POPULATION_GROWTH_TYPE] [-nb NUM_BIRTHS_INITIAL] [-ip] [-o PATH_OUTPUT] [-f] [-v] [-h]

  options:
    -r, --run-simulation  Run the simulation.

  ARGUMENTS:
    ...

Now that you've successfully installed the ``LEAP`` package, you can start using it.
To get started running a simulation, see the :doc:`Command-Line Interface <cli/index>` documentation.

To get out of the virtual environment:

.. code:: bash

    deactivate


Developers
***********


If you want to develop this package, please see the installation instructions for
developers: :doc:`Developer Installation <../dev/dev-installation>`.



Troubleshooting
****************

- If ``python3 --version`` does not return Python 3.10 or higher, ensure Python 3.10 is installed
  and that the ``python3`` command points to the correct version.
- If ``pip`` is not installed, you can install it manually:

  .. code-block:: bash
    
    python3 -m ensurepip --upgrade
