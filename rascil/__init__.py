
from . import data_models
from . import processing_components
from . import workflows

from .processing_components.util.installation_checks import check_data_directory

from astroplan import download_IERS_A

check_data_directory()

download_IERS_A()
