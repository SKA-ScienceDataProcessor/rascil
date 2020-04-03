
from . import data_models
from . import processing_components
from . import workflows

from .processing_components.util.installation_checks import check_data_directory
from astropy.utils import iers

check_data_directory()

# This turns off all downloads of the IERS tables. This is a hack until astropy
# addresses the multiple reader behaviour of the cache reader/updater.
# TODO: Fix IERS table updates when astropy changes cache access
iers.conf.auto_max_age = None
iers.conf.auto_download = False
