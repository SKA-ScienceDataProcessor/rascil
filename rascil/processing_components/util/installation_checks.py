"""Function to check the installation

"""

from rascil.data_models import rascil_path

__all__ = ['check_data_directory']

def check_data_directory(verbose=False):
    """ Check the data directory to see if it has been downloaded correctly
    """
    canary = rascil_path("data/configurations/LOWBD2.csv")
    with open(canary, "r") as f:
        first = f.read(1)
        if first == "version https://git-lfs.github.com/spec/v1":
            raise ValueError("The data directory has not been installed correctly - perhaps git-lfs is needed")
        else:
            if verbose: print("The data directory appears to have been installed correctly")

if __name__ == "__main__":
    check_data_directory()