""" Image operations visible to the Execution Framework as Components

"""

import logging
import warnings

from astropy.io import fits
from astropy.wcs import FITSFixedWarning
from astropy.wcs import WCS

from rascil.data_models.memory_data_models import Image
from rascil.processing_components.image import polarisation_frame_from_wcs

warnings.simplefilter('ignore', FITSFixedWarning)
log = logging.getLogger('logger')

if __name__ == "__main__":
    fitsfile = "/Users/timcornwell/Code/rascil/data/models/MID_FEKO_VP_B1_45_0765_real_original.fits"
    fim = Image()
    hdulist = fits.open(fitsfile)
    fim.data = hdulist[0].data
    fim.wcs = WCS(fitsfile)
    hdulist.close()
    fim.polarisation_frame = polarisation_frame_from_wcs(fim.wcs, fim.data.shape)
    fim.data[:,1,...] *= -1.0
    fitsfile = "/Users/timcornwell/Code/rascil/data/models/MID_FEKO_VP_B1_45_0765_real.fits"
    fits.writeto(filename=fitsfile, data=fim.data, header=fim.wcs.to_header(), overwrite=True)
