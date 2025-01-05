Installation
==============

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
`Mac Installer Python 3.10
<https://www.python.org/ftp/python/3.10.0/python-3.10.0post2-macos11.pkg>`_.

Once the install has finished, check to make sure that it's using the correct version:

.. code:: bash

  python3 --version

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
5. Copy this token. You won't be shown it again.
6. In your terminal, type:

.. code:: bash

  git config --global credential.helper osxkeychain

Step 5: Install ``LEAP``
-------------------------------

To install the ``LEAP`` package, type:

.. code:: bash

  pip3 install git+https://github.com/resplab/leap.git


*(Optional)* If you set up your ``git`` token in the last step, you will now be asked for you
username and password. Paste the token in as your password:

.. code:: bash

  Username for 'https://github.com': YOUR_GIT_USERNAME # not your email address
  Password: YOUR_TOKEN # not your GitHub password


Step 6: Install Success!
--------------------------

Now that you've successfully installed the ``LEAP`` package, you can start using it. To get
started, see the :doc:`Command-Line Interface <cli/index>` documentation.

To get out of the virtual environment:

.. code:: bash

  deactivate



Developers
***********


If you want to develop this package, please see the installation instructions for
developers: :doc:`Developer Installation <../dev/dev-installation>`.
