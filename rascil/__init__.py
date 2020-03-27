
from . import data_models
from . import processing_components
from . import workflows

from .processing_components.util.installation_checks import check_data_directory

from astroplan import get_IERS_A_or_workaround
get_IERS_A_or_workaround()

check_data_directory()

