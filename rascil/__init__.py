
from . import data_models
from . import processing_components
from . import workflows

from .processing_components.util.installation_checks import check_data_directory
from astropy.utils import iers, data

check_data_directory()

# iers.conf.auto_max_age = None
iers.conf.remote_timeout = 100.0
data.conf.download_cache_lock_attempts = 10
