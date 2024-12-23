import shutil
from pathlib import Path

PATH_LEAP = Path(__file__).parents[1].absolute()
__test_dir__ = Path(PATH_LEAP, "tests")
__tmp_dir__ = Path(PATH_LEAP, "tmp")


def create_tmp_dir():
    """Create a temporary directory for testing.

    1. Remove the ``tmp`` directory if it exists.
    2. Create a new ``tmp`` directory.

    Any data files created during testing will go into ``tmp`` directory.
    This is created/removed for each test.

    """
    remove_tmp_dir()
    Path(__tmp_dir__).mkdir()


def remove_tmp_dir():
    """Recursively remove the ``tmp`` directory if it exists."""
    shutil.rmtree(__tmp_dir__, ignore_errors=True)
