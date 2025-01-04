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

1. Run ``sphinx-apidoc``:

.. code:: bash

  cd LEAP/docs
  sphinx-apidoc -o ./dev/api/ ../leap/ --separate


2. Delete the automatially generated ``docs/dev/api/modules.rst`` file.
3. Add the new module to the ``docs/dev/api/leap.rst`` file.


Sphinx-Immaterial Theme
***********************

The documentation is styled using the `Sphinx-Immaterial`_ theme.

GitHub Pages
************

The documentation is currently hosted by ``GitHub Pages``. This is done using ``GitHub Actions``,
which automatically builds the documentation and pushes it to the ``gh-pages`` branch. The
action can be found under `.github/workflows/docs_pages_workflow.yml`_:

.. code:: yaml

  name: docs_pages_workflow
  
  # execute this workflow automatically when a we push to main branch
  on:
    push:
      branches: [ docs ]
  
  jobs:
    docs:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v4
        - uses: actions/setup-python@v5
          with:
            python-version: '3.10'
        - name: Install dependencies for docs
          run: |
            pip install -r requirements-docs.txt
        - name: Install LEAP requirements
          run: |
            pip install -r requirements.txt
        - name: Sphinx build
          run: |
            sphinx-build docs _build -E -a
        - name: Deploy to GitHub Pages
          uses: peaceiris/actions-gh-pages@v3
          if: ${{ github.event_name == 'push' && github.ref == 'refs/heads/docs' }}
          with:
            publish_branch: gh-pages
            github_token: ${{ secrets.GITHUB_TOKEN }}
            publish_dir: _build/
            force_orphan: true



.. _Sphinx-Immaterial: https://sphinx-immaterial.readthedocs.io/en/stable/index.html
.. _.github/workflows/docs_pages_workflow.yml: https://github.com/resplab/leap/blob/main/.github/workflows/docs_pages_workflow.yml