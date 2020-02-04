"""Function to check the installation

"""

import logging

from rascil.data_models import rascil_path

log = logging.getLogger(__file__)

__all__ = ['check_data_directory']

def check_data_directory(verbose=False):
    """ Check the data directory to see if it has been downloaded correctly
    """
    canary = rascil_path("data/configurations/LOWBD2.csv")
    try:
        with open(canary, "r") as f:
            first = f.read(1)
            if first == "version https://git-lfs.github.com/spec/v1":
                log.warning(
                    "The data directory is not available - if required then git-lfs is needed")
            else:
                if verbose: print("The data directory appears to have been installed correctly")
    except FileNotFoundError:
        log.warning("The data directory is not available - if required then git-lfs is needed")


if __name__ == "__main__":
    check_data_directory()