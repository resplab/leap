Releases
====================

To see all the releases, go to:
`LEAP Releases <https://github.com/resplab/leap/releases>`_.

Creating a Release
*******************

To create a release, you will need to add a version tag to your commit, in the format ``vx.y.z``.
First, commit your changes:

.. code:: bash

  git add .
  git commit -m "Your commit message"

Then, tag your commit:

.. code:: bash

  git tag -a vx.y.z -m "Your tag message"
  git push origin vx.y.z


``GitHub Actions`` will automatically release any commits that are tagged starting with ``v``. 
This workflow can be found here: `.github/workflows/release_workflow.yml
<https://github.com/resplab/leap/.github/workflows/release_workflow.yml>`_.


.. code:: yaml

    name: release_workflow

    # execute this workflow automatically when a we create a release
    on:
    push:
        tags:
        - 'v*'

    jobs:
    publish_release:
        name: Build and publish Python package
        runs-on: ubuntu-latest
        steps:
        - uses: actions/checkout@v4
            with:
            lfs: true
        - name: Create Release
            id: create_release
            uses: actions/create-release@v1
            env:
            GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
            with:
            tag_name: ${{ github.ref }}
            release_name: Release ${{ github.ref }}
            draft: false
            prerelease: false
        - uses: actions/setup-python@v5
            with:
            python-version: '3.10'
        - name: Install LEAP requirements
            run: |
            pip install -r requirements.txt
        - name: Install wheel
            run: pip install wheel

        - name: Build a binary wheel
            run: python setup.py bdist_wheel

        - name: Archive package
            if: ${{ github.event_name == 'push' }}
            uses: actions/upload-artifact@v4
            with:
            name: leap
            path: dist/

        - name: upload linux artifact
            uses: actions/upload-release-asset@v1
            env:
            GITHUB_TOKEN: ${{ github.token }}
            with:
            upload_url: ${{ steps.create_release.outputs.upload_url }}
            asset_path: ./dist/leap-1.0-py3-none-any.whl
            asset_name: leap-1.0-py3-none-any.whl
            asset_content_type: application/zip