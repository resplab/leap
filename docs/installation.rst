Installation
==============

Prerequisites
----------------------------
- Administrator or ``sudo`` privileges (Linux/macOS).
- ``GitHub`` account created.
- ``git`` installed on your system.
- Internet access to download dependencies.

Step 1: Install Python
----------------------------

If you already have ``Python`` installed, you can skip this step. To check your ``Python`` version,
open a terminal and type:


.. code-block:: bash

  python3 --version


If the output is ``command not found: python3``, then you will need to install Python.
LEAP requires Python 3.10 or higher.

macOS
******
To install Python, download the installer from:
`Mac Installer Python 3.10 <https://www.python.org/ftp/python/3.10.0/python-3.10.0post2-macos11.pkg>`_.

Windows
******
To install Python, download the installer from:
`Windows Installer Python 3.10
<https://www.python.org/downloads/release/python-31016/>`_.

Linux
******

To install Python, use your package manager. For example, on Ubuntu, type:

.. code-block:: bash

  sudo apt update
  sudo apt install python3.10

For other distributions, refer to your distribution's documentation.

Confirming Installation
******

Once the installation has finished, check to make sure that it's using the correct version:

.. code-block:: bash

   python3 --version
   # Expected output:
   # Python 3.10.*

Troubleshooting
***************
- If ``python3 --version`` does not return Python 3.10 or higher, ensure Python 3.10 is installed and that the ``python3`` command points to the correct version.
- If ``pip`` is not installed, you can install it manually:

  .. code-block:: bash

    python3 -m ensurepip --upgrade

Step 2: Choose Folder to Install LEAP
-------------------------------

Create a directory to install ``LEAP``. In this example, we'll use ``my_folder``. To
create this folder, open a terminal and type:

.. code-block:: bash

  mkdir my_folder
  cd my_folder

This folder will serve as the workspace where you install and manage the LEAP package.

Step 3: Create a Virtual Environment
--------------------------------------

When working with Python, it's best to create a virtual environment to keep your packages
separate from the system Python. You will want to use your same Python version so if you
installed ``python3.10`` with the installer above, the command would be:

Linux/macOS
******

.. code-block:: bash

  python3.10 -m venv env
  source env/bin/activate

Windows Command Prompt
******

.. code-block:: cmd

  python -m venv env
  env\Scripts\activate

Windows PowerShell
******

For PowerShell, you may need to allow scripts temporarily:

.. code-block:: powershell

  python -m venv env
  Set-ExecutionPolicy Unrestricted -Scope Process # You may need to include this command
  .\env\Scripts\Activate.ps1

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

macOS
******

6. In your terminal, type:

.. code-block:: bash

  git config --global credential.helper osxkeychain

Linux
******

6. In your terminal, either type:

.. code-block:: bash

  git config --global credential.helper cache

Use this for temporary storage of credentials in memory (default 15 minutes). Add a note about setting a custom timeout with ``cache --timeout=3600`` for longer sessions.

.. code-block:: bash

  git config --global credential.helper store

Use this for persistent storage of credentials in plain text. Add a security warning that credentials are stored in ``~/.git-credentials`` and should only be used on trusted machines.

Windows
******

6. In your terminal, type:

.. code-block:: cmd

  git config --global credential.helper manager

7. Test your setup by cloning a private repository or running a Git command like:

.. code-block:: bash

  git ls-remote https://github.com/YOUR_USERNAME/YOUR_PRIVATE_REPO.git

  # Replace YOUR_USERNAME and YOUR_PRIVATE_REPO with your GitHub username and a private repository name.

*(Optional)* If you encounter issues, ensure that Git is installed on your system. You can download it from:
`Git Downloads <https://git-scm.com/downloads>`_.

Step 5: Install ``LEAP``
-------------------------------

To install the ``LEAP`` package, type:


.. code-block:: bash

  pip3 install git+https://github.com/resplab/leap.git

Windows users may need to run the following command if the one above does not work:

.. code-block:: cmd

  python -m pip install git+https://github.com/resplab/leap.git

To install a specific release, append ``@vx.y.z`` at the end like:

.. code-block:: bash

  pip3 install git+https://github.com/resplab/leap.git@vx.y.z

Here, ``vx.y.z`` is the release version you would like to install. To see all the releases, go to:
`LEAP Releases <https://github.com/resplab/leap/releases>`_.

*(Optional)* If you set up your ``git`` token in the last step, you will now be asked for you
username and password. Paste the token in as your password:

.. code-block:: bash

  Username for 'https://github.com': YOUR_GIT_USERNAME # not your email address
  Password: YOUR_TOKEN # not your GitHub password


Step 6: Install Success!
--------------------------

Now that you've successfully installed the ``LEAP`` package, you can start using it.
To verify the installation was a success, type:

.. code-block:: bash

  leap --help

You should see various options appear of how to use the LEAP command.
To get started running a simulation, see the :doc:`Command-Line Interface <cli/index>` documentation.

To get out of the virtual environment:

.. code-block:: bash

  deactivate



Developers
***********


If you want to develop this package, please see the installation instructions for
developers: :doc:`Developer Installation <../dev/dev-installation>`.
