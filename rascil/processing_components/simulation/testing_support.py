"""
Functions that aid testing in various ways. A typical use would be::

        lowcore = create_named_configuration('LOWBD2-CORE')
        times = numpy.linspace(-3, +3, 13) * (numpy.pi / 12.0)
        
        frequency = numpy.array([1e8])
        channel_bandwidth = numpy.array([1e7])
        
        # Define the component and give it some polarisation and spectral behaviour
        f = numpy.array([100.0])
        flux = numpy.array([f])
        
        phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')
        compabsdirection = SkyCoord(ra=17.0 * u.deg, dec=-36.5 * u.deg, frame='icrs', equinox='J2000')
        
        comp = create_skycomponent(flux=flux, frequency=frequency, direction=compabsdirection,
                                        polarisation_frame=PolarisationFrame('stokesI'))
        image = create_test_image(frequency=frequency, phasecentre=phasecentre,
                                                      cellsize=0.001,
                                                      polarisation_frame=PolarisationFrame('stokesI')
        
        vis = create_visibility(lowcore, times=times, frequency=frequency,
                                     channel_bandwidth=channel_bandwidth,
                                     phasecentre=phasecentre, weight=1,
                                     polarisation_frame=PolarisationFrame('stokesI'),
                                     integration_time=1.0)

"""

__all__ = ['create_blockvisibility_iterator',
           'create_low_test_image_from_gleam',
           'create_low_test_skycomponents_from_gleam',
           'create_low_test_skymodel_from_gleam',
           'create_test_image',
           'create_test_image_from_s3',
           'create_test_skycomponents_from_s3',
           'create_unittest_components',
           'create_unittest_model',
           'ingest_unittest_visibility',
           'insert_unittest_errors',
           'replicate_image',
           'simulate_gaintable']

import csv
import logging
from typing import List

import astropy.units as u
import numpy
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
from scipy import interpolate

from rascil.data_models.memory_data_models import Configuration, Image, GainTable, Skycomponent, SkyModel
from rascil.data_models.parameters import rascil_path, rascil_data_path
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.calibration.chain_calibration import create_calibration_controls
from rascil.processing_components.calibration.operations import create_gaintable_from_blockvisibility, apply_gaintable
from rascil.processing_components.image.operations import create_image_from_array
from rascil.processing_components.image.operations import import_image_from_fits
from rascil.processing_components.imaging import predict_2d, dft_skycomponent_visibility, \
    create_image_from_visibility, advise_wide_field
from rascil.processing_components.imaging.primary_beams import create_pb
from rascil.processing_components.skycomponent.operations import create_skycomponent, insert_skycomponent, \
    apply_beam_to_skycomponent, filter_skycomponents_by_flux
from rascil.processing_components.visibility.base import create_blockvisibility, create_visibility
from rascil.processing_components.visibility.coalesce import convert_blockvisibility_to_visibility, \
    convert_visibility_to_blockvisibility
from rascil.processing_components.util.installation_checks import check_data_directory

check_data_directory()
log = logging.getLogger('logger')


def create_test_image(canonical=True, cellsize=None, frequency=None, channel_bandwidth=None,
                      phasecentre=None, polarisation_frame=PolarisationFrame("stokesI")) -> Image:
    """Create a useful test image

    This is the test image M31 widely used in ALMA and other simulations. It is actually part of an Halpha region in
    M31.

    :param canonical: Make the image into a 4 dimensional image
    :param cellsize:
    :param frequency: Frequency (array) in Hz
    :param channel_bandwidth: Channel bandwidth (array) in Hz
    :param phasecentre: Phase centre of image (SkyCoord)
    :param polarisation_frame: Polarisation frame
    :return: Image
    """
    check_data_directory()

    if frequency is None:
        frequency = [1e8]
    im = import_image_from_fits(rascil_path("data/models/M31.MOD"))
    if canonical:

        if polarisation_frame is None:
            im.polarisation_frame = PolarisationFrame("stokesI")
        elif isinstance(polarisation_frame, PolarisationFrame):
            im.polarisation_frame = polarisation_frame
        else:
            raise ValueError("polarisation_frame is not valid")

        im = replicate_image(im, frequency=frequency, polarisation_frame=im.polarisation_frame)
        if cellsize is not None:
            im.wcs.wcs.cdelt[0] = -180.0 * cellsize / numpy.pi
            im.wcs.wcs.cdelt[1] = +180.0 * cellsize / numpy.pi
        if frequency is not None:
            im.wcs.wcs.crval[3] = frequency[0]
        if channel_bandwidth is not None:
            im.wcs.wcs.cdelt[3] = channel_bandwidth[0]
        else:
            if len(frequency) > 1:
                im.wcs.wcs.cdelt[3] = frequency[1] - frequency[0]
            else:
                im.wcs.wcs.cdelt[3] = 0.001 * frequency[0]
        im.wcs.wcs.radesys = 'ICRS'
        im.wcs.wcs.equinox = 2000.00

    if phasecentre is not None:
        im.wcs.wcs.crval[0] = phasecentre.ra.deg
        im.wcs.wcs.crval[1] = phasecentre.dec.deg
        # WCS is 1 relative
        im.wcs.wcs.crpix[0] = im.data.shape[3] // 2 + 1
        im.wcs.wcs.crpix[1] = im.data.shape[2] // 2 + 1

    return im


def create_test_image_from_s3(npixel=16384, polarisation_frame=PolarisationFrame("stokesI"), cellsize=0.000015,
                              frequency=numpy.array([1e8]), channel_bandwidth=numpy.array([1e6]),
                              phasecentre=None, fov=20, flux_limit=1e-3) -> Image:
    """Create MID test image from S3

    The input catalog was generated at http://s-cubed.physics.ox.ac.uk/s3_sex using the following query::
        Database: s3_sex
        SQL: select * from Galaxies where (pow(10,itot_151)*1000 > 1.0) and (right_ascension between -5 and 5) and (declination between -5 and 5);;

    Number of rows returned: 29966

    For frequencies < 610MHz, there are three tables to use::

        data/models/S3_151MHz_10deg.csv, use fov=10
        data/models/S3_151MHz_20deg.csv, use fov=20
        data/models/S3_151MHz_40deg.csv, use fov=40

    For frequencies > 610MHz, there are three tables:

        data/models/S3_1400MHz_1mJy_10deg.csv, use flux_limit>= 1e-3
        data/models/S3_1400MHz_100uJy_10deg.csv, use flux_limit < 1e-3
        data/models/S3_1400MHz_1mJy_18deg.csv, use flux_limit>= 1e-3
        data/models/S3_1400MHz_100uJy_18deg.csv, use flux_limit < 1e-3

    The component spectral index is calculated from the 610MHz and 151MHz or 1400MHz and 610MHz, and then calculated
    for the specified frequencies.

    If polarisation_frame is not stokesI then the image will a polarised axis but the values will be zero.

    :param npixel: Number of pixels
    :param polarisation_frame: Polarisation frame (default PolarisationFrame("stokesI"))
    :param cellsize: cellsize in radians
    :param frequency:
    :param channel_bandwidth: Channel width (Hz)
    :param phasecentre: phasecentre (SkyCoord)
    :param fov: fov 10 | 20 | 40
    :param flux_limit: Minimum flux (Jy)
    :return: Image
    """
    check_data_directory()

    ras = []
    decs = []
    fluxes = []

    if phasecentre is None:
        phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')

    if polarisation_frame is None:
        polarisation_frame = PolarisationFrame("stokesI")

    npol = polarisation_frame.npol

    nchan = len(frequency)

    shape = [nchan, npol, npixel, npixel]
    w = WCS(naxis=4)
    # The negation in the longitude is needed by definition of RA, DEC
    w.wcs.cdelt = [-cellsize * 180.0 / numpy.pi, cellsize * 180.0 / numpy.pi, 1.0, channel_bandwidth[0]]
    w.wcs.crpix = [npixel // 2 + 1, npixel // 2 + 1, 1.0, 1.0]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", 'STOKES', 'FREQ']
    w.wcs.crval = [phasecentre.ra.deg, phasecentre.dec.deg, 1.0, frequency[0]]
    w.naxis = 4

    w.wcs.radesys = 'ICRS'
    w.wcs.equinox = 2000.0

    model = create_image_from_array(numpy.zeros(shape), w, polarisation_frame=polarisation_frame)

    if numpy.max(frequency) > 6.1E8:
        if fov > 10:
            fovstr = '18'
        else:
            fovstr = '10'
        if flux_limit >= 1e-3:
            csvfilename = rascil_data_path('models/S3_1400MHz_1mJy_%sdeg.csv' % fovstr)
        else:
            csvfilename = rascil_data_path('models/S3_1400MHz_100uJy_%sdeg.csv' % fovstr)
        log.info('create_test_image_from_s3: Reading S3 sources from %s ' % csvfilename)
    else:
        assert fov in [10, 20, 40], "Field of view invalid: use one of %s" % ([10, 20, 40])
        csvfilename = rascil_data_path('models/S3_151MHz_%ddeg.csv' % (fov))
        log.info('create_test_image_from_s3: Reading S3 sources from %s ' % csvfilename)

    with open(csvfilename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        r = 0
        for row in readCSV:
            # Skip first row
            if r > 0:
                ra = float(row[4]) + phasecentre.ra.deg
                dec = float(row[5]) + phasecentre.dec.deg
                if numpy.max(frequency) > 6.1E8:
                    alpha = (float(row[11]) - float(row[10])) / numpy.log10(1400.0 / 610.0)
                    flux = numpy.power(10, float(row[10])) * numpy.power(frequency / 1.4e9, alpha)
                else:
                    alpha = (float(row[10]) - float(row[9])) / numpy.log10(610.0 / 151.0)
                    flux = numpy.power(10, float(row[9])) * numpy.power(frequency / 1.51e8, alpha)
                if numpy.max(flux) > flux_limit:
                    ras.append(ra)
                    decs.append(dec)
                    fluxes.append(flux)
            r += 1

    csvfile.close()

    assert len(fluxes) > 0, "No sources found above flux limit %s" % flux_limit

    log.info('create_test_image_from_s3: %d sources read' % (len(fluxes)))

    p = w.sub(2).wcs_world2pix(numpy.array(ras), numpy.array(decs), 1)
    fluxes = numpy.array(fluxes)
    total_flux = numpy.sum(fluxes)
    ip = numpy.round(p).astype('int')
    ok = numpy.where((0 <= ip[0, :]) & (npixel > ip[0, :]) & (0 <= ip[1, :]) & (npixel > ip[1, :]))[0]
    ps = ip[:, ok]
    fluxes = fluxes[ok]
    actual_flux = numpy.sum(fluxes)

    log.info('create_test_image_from_s3: %d sources inside the image' % (ps.shape[1]))

    log.info('create_test_image_from_s3: average channel flux in S3 model = %.3f, actual average channel flux in '
             'image = %.3f' % (total_flux / float(nchan), actual_flux / float(nchan)))
    for chan in range(nchan):
        for iflux, flux in enumerate(fluxes):
            model.data[chan, 0, ps[1, iflux], ps[0, iflux]] = flux[chan]

    return model


def create_test_skycomponents_from_s3(polarisation_frame=PolarisationFrame("stokesI"),
                                      frequency=numpy.array([1e8]), channel_bandwidth=numpy.array([1e6]),
                                      phasecentre=None, fov=20, flux_limit=1e-3,
                                      radius=None):
    """Create test image from S3

    The input catalog was generated at http://s-cubed.physics.ox.ac.uk/s3_sex using the following query::
        Database: s3_sex
        SQL: select * from Galaxies where (pow(10,itot_151)*1000 > 1.0) and (right_ascension between -5 and 5) and (declination between -5 and 5);;

    Number of rows returned: 29966

    For frequencies < 610MHz, there are three tables to use::

        data/models/S3_151MHz_10deg.csv, use fov=10
        data/models/S3_151MHz_20deg.csv, use fov=20
        data/models/S3_151MHz_40deg.csv, use fov=40

    For frequencies > 610MHz, there are three tables:

        data/models/S3_1400MHz_1mJy_10deg.csv, use flux_limit>= 1e-3
        data/models/S3_1400MHz_100uJy_10deg.csv, use flux_limit < 1e-3
        data/models/S3_1400MHz_1mJy_18deg.csv, use flux_limit>= 1e-3
        data/models/S3_1400MHz_100uJy_18deg.csv, use flux_limit < 1e-3

    The component spectral index is calculated from the 610MHz and 151MHz or 1400MHz and 610MHz, and then calculated
    for the specified frequencies.

    If polarisation_frame is not stokesI then the image will a polarised axis but the values will be zero.

    :param polarisation_frame: Polarisation frame (default PolarisationFrame("stokesI"))
    :param frequency:
    :param channel_bandwidth: Channel width (Hz)
    :param phasecentre: phasecentre (SkyCoord)
    :param fov: fov 10 | 20 | 40
    :param flux_limit: Minimum flux (Jy)
    :return: Image
    """
    check_data_directory()

    ras = []
    decs = []
    fluxes = []
    names = []

    if phasecentre is None:
        phasecentre = SkyCoord(ra=+180.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')

    if polarisation_frame is None:
        polarisation_frame = PolarisationFrame("stokesI")

    if numpy.max(frequency) > 6.1E8:
        if fov > 10:
            fovstr = '18'
        else:
            fovstr = '10'
        if flux_limit >= 1e-3:
            csvfilename = rascil_data_path('models/S3_1400MHz_1mJy_%sdeg.csv' % fovstr)
        else:
            csvfilename = rascil_data_path('models/S3_1400MHz_100uJy_%sdeg.csv' % fovstr)
        log.info('create_test_skycomponents_from_s3: Reading S3-SEX sources from %s ' % csvfilename)
    else:
        assert fov in [10, 20, 40], "Field of view invalid: use one of %s" % ([10, 20, 40])
        csvfilename = rascil_data_path('models/S3_151MHz_%ddeg.csv' % (fov))
        log.info('create_test_skycomponents_from_s3: Reading S3-SEX sources from %s ' % csvfilename)

    skycomps = list()

    with open(csvfilename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        r = 0
        for row in readCSV:
            # Skip first row
            if r > 0:
                ra = float(row[4]) / numpy.cos(phasecentre.dec.rad) + phasecentre.ra.deg
                dec = float(row[5]) + phasecentre.dec.deg
                if numpy.max(frequency) > 6.1E8:
                    alpha = (float(row[11]) - float(row[10])) / numpy.log10(1400.0 / 610.0)
                    flux = numpy.power(10, float(row[10])) * numpy.power(frequency / 1.4e9, alpha)
                else:
                    alpha = (float(row[10]) - float(row[9])) / numpy.log10(610.0 / 151.0)
                    flux = numpy.power(10, float(row[9])) * numpy.power(frequency / 1.51e8, alpha)
                if numpy.max(flux) > flux_limit:
                    ras.append(ra)
                    decs.append(dec)
                    if polarisation_frame == PolarisationFrame("stokesIQUV"):
                        polscale = numpy.array([1.0, 0.0, 0.0, 0.0])
                        fluxes.append(numpy.outer(flux, polscale))
                    else:
                        fluxes.append([[f] for f in flux])
                    names.append("S3_%s" % row[0])
            r += 1

    csvfile.close()

    assert len(fluxes) > 0, "No sources found above flux limit %s" % flux_limit

    directions = SkyCoord(ra=ras * u.deg, dec=decs * u.deg)
    if phasecentre is not None:
        separations = directions.separation(phasecentre).to('rad').value
    else:
        separations = numpy.zeros(len(names))

    for isource, name in enumerate(names):
        direction = directions[isource]
        if separations[isource] < radius:
            if not numpy.isnan(flux).any():
                skycomps.append(Skycomponent(direction=direction, flux=fluxes[isource], frequency=frequency,
                                             name=names[isource], shape='Point',
                                             polarisation_frame=polarisation_frame))

    log.info('create_test_skycomponents_from_s3: %d sources found above fluxlimit inside search radius' %
             len(skycomps))

    return skycomps


def create_low_test_image_from_gleam(npixel=512, polarisation_frame=PolarisationFrame("stokesI"), cellsize=0.000015,
                                     frequency=numpy.array([1e8]), channel_bandwidth=numpy.array([1e6]),
                                     phasecentre=None, kind='cubic', applybeam=False, flux_limit=0.1,
                                     flux_max=numpy.inf, flux_min=-numpy.inf,
                                     radius=None, insert_method='Nearest') -> Image:
    """Create LOW test image from the GLEAM survey

    Stokes I is estimated from a cubic spline fit to the measured fluxes. The polarised flux is always zero.

    See http://www.mwatelescope.org/science/gleam-survey The catalog is available from Vizier.

    VIII/100   GaLactic and Extragalactic All-sky MWA survey  (Hurley-Walker+, 2016)

    GaLactic and Extragalactic All-sky Murchison Wide Field Array (GLEAM) survey. I: A low-frequency extragalactic
    catalogue. Hurley-Walker N., et al., Mon. Not. R. Astron. Soc., 464, 1146-1167 (2017), 2017MNRAS.464.1146H

    :param npixel: Number of pixels
    :param polarisation_frame: Polarisation frame (default PolarisationFrame("stokesI"))
    :param cellsize: cellsize in radians
    :param frequency:
    :param channel_bandwidth: Channel width (Hz)
    :param phasecentre: phasecentre (SkyCoord)
    :param kind: Kind of interpolation (see scipy.interpolate.interp1d) Default: linear
    :return: Image

    """
    check_data_directory()

    if phasecentre is None:
        phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')

    if radius is None:
        radius = npixel * cellsize / numpy.sqrt(2.0)

    sc = create_low_test_skycomponents_from_gleam(flux_limit=flux_limit, polarisation_frame=polarisation_frame,
                                                  frequency=frequency, phasecentre=phasecentre,
                                                  kind=kind, radius=radius)

    sc = filter_skycomponents_by_flux(sc, flux_min=flux_min, flux_max=flux_max)

    if polarisation_frame is None:
        polarisation_frame = PolarisationFrame("stokesI")

    npol = polarisation_frame.npol
    nchan = len(frequency)
    shape = [nchan, npol, npixel, npixel]
    w = WCS(naxis=4)
    # The negation in the longitude is needed by definition of RA, DEC
    w.wcs.cdelt = [-cellsize * 180.0 / numpy.pi, cellsize * 180.0 / numpy.pi, 1.0, channel_bandwidth[0]]
    w.wcs.crpix = [npixel // 2 + 1, npixel // 2 + 1, 1.0, 1.0]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", 'STOKES', 'FREQ']
    w.wcs.crval = [phasecentre.ra.deg, phasecentre.dec.deg, 1.0, frequency[0]]
    w.naxis = 4
    w.wcs.radesys = 'ICRS'
    w.wcs.equinox = 2000.0

    model = create_image_from_array(numpy.zeros(shape), w, polarisation_frame=polarisation_frame)

    model = insert_skycomponent(model, sc, insert_method=insert_method)
    if applybeam:
        beam = create_pb(model, telescope='LOW', use_local=False)
        model.data[...] *= beam.data[...]

    return model


def create_low_test_skymodel_from_gleam(npixel=512, polarisation_frame=PolarisationFrame("stokesI"), cellsize=0.000015,
                                        frequency=numpy.array([1e8]), channel_bandwidth=numpy.array([1e6]),
                                        phasecentre=None, kind='cubic', applybeam=True, flux_limit=0.1,
                                        flux_max=numpy.inf, flux_threshold=1.0, insert_method='Nearest',
                                        telescope='LOW') -> SkyModel:
    """Create LOW test skymodel from the GLEAM survey

    Stokes I is estimated from a cubic spline fit to the measured fluxes. The polarised flux is always zero.

    See http://www.mwatelescope.org/science/gleam-survey The catalog is available from Vizier.

    VIII/100   GaLactic and Extragalactic All-sky MWA survey  (Hurley-Walker+, 2016)

    GaLactic and Extragalactic All-sky Murchison Wide Field Array (GLEAM) survey. I: A low-frequency extragalactic
    catalogue. Hurley-Walker N., et al., Mon. Not. R. Astron. Soc., 464, 1146-1167 (2017), 2017MNRAS.464.1146H

    :param telescope:
    :param npixel: Number of pixels
    :param polarisation_frame: Polarisation frame (default PolarisationFrame("stokesI"))
    :param cellsize: cellsize in radians
    :param frequency:
    :param channel_bandwidth: Channel width (Hz)
    :param phasecentre: phasecentre (SkyCoord)
    :param kind: Kind of interpolation (see scipy.interpolate.interp1d) Default: cubic
    :param applybeam: Apply the primary beam?
    :param flux_limit: Weakest component
    :param flux_max: Maximum strength component to be included in components
    :param flux_threshold: Split between components (brighter) and image (weaker)
    :param insert_method: Nearest | PSWF | Lanczos
    :return:
    :return: SkyModel

    """
    check_data_directory()

    if phasecentre is None:
        phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')

    radius = npixel * cellsize

    sc = create_low_test_skycomponents_from_gleam(flux_limit=flux_limit, polarisation_frame=polarisation_frame,
                                                  frequency=frequency, phasecentre=phasecentre,
                                                  kind=kind, radius=radius)

    sc = filter_skycomponents_by_flux(sc, flux_max=flux_max)
    if polarisation_frame is None:
        polarisation_frame = PolarisationFrame("stokesI")

    npol = polarisation_frame.npol
    nchan = len(frequency)
    shape = [nchan, npol, npixel, npixel]
    w = WCS(naxis=4)
    # The negation in the longitude is needed by definition of RA, DEC
    w.wcs.cdelt = [-cellsize * 180.0 / numpy.pi, cellsize * 180.0 / numpy.pi, 1.0, channel_bandwidth[0]]
    w.wcs.crpix = [npixel // 2 + 1, npixel // 2 + 1, 1.0, 1.0]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN", 'STOKES', 'FREQ']
    w.wcs.crval = [phasecentre.ra.deg, phasecentre.dec.deg, 1.0, frequency[0]]
    w.naxis = 4
    w.wcs.radesys = 'ICRS'
    w.wcs.equinox = 2000.0

    model = create_image_from_array(numpy.zeros(shape), w, polarisation_frame=polarisation_frame)

    if applybeam:
        beam = create_pb(model, telescope=telescope, use_local=False)
        sc = apply_beam_to_skycomponent(sc, beam)

    weaksc = filter_skycomponents_by_flux(sc, flux_max=flux_threshold)
    brightsc = filter_skycomponents_by_flux(sc, flux_min=flux_threshold)
    model = insert_skycomponent(model, weaksc, insert_method=insert_method)

    log.info(
        'create_low_test_skymodel_from_gleam: %d bright sources above flux threshold %.3f, %d weak sources below ' %
        (len(brightsc), flux_threshold, len(weaksc)))

    return SkyModel(components=brightsc, image=model, mask=None, gaintable=None)


def create_low_test_skycomponents_from_gleam(flux_limit=0.1, polarisation_frame=PolarisationFrame("stokesI"),
                                             frequency=numpy.array([1e8]), kind='cubic', phasecentre=None,
                                             radius=1.0) \
        -> List[Skycomponent]:
    """Create sky components from the GLEAM survey

    Stokes I is estimated from a cubic spline fit to the measured fluxes. The polarised flux is always zero.
    
    See http://www.mwatelescope.org/science/gleam-survey The catalog is available from Vizier.
    
    VIII/100   GaLactic and Extragalactic All-sky MWA survey  (Hurley-Walker+, 2016)

    GaLactic and Extragalactic All-sky Murchison Wide Field Array (GLEAM) survey. I: A low-frequency extragalactic
    catalogue. Hurley-Walker N., et al., Mon. Not. R. Astron. Soc., 464, 1146-1167 (2017), 2017MNRAS.464.1146H


    :param flux_limit: Only write components brighter than this (Jy)
    :param polarisation_frame: Polarisation frame (default PolarisationFrame("stokesI"))
    :param frequency: Frequencies at which the flux will be estimated
    :param kind: Kind of interpolation (see scipy.interpolate.interp1d) Default: linear
    :param phasecentre: Desired phase centre (SkyCoord) default None implies all sources
    :param radius: Radius of sources selected around phasecentre (default 1.0 rad)
    :return: List of Skycomponents
    """
    check_data_directory()

    fitsfile = rascil_path("data/models/GLEAM_EGC.fits")

    rad2deg = 180.0 / numpy.pi
    decmin = phasecentre.dec.to('deg').value - rad2deg * radius / 2.0
    decmax = phasecentre.dec.to('deg').value + rad2deg * radius / 2.0

    hdulist = fits.open(fitsfile, lazy_load_hdus=False)
    recs = hdulist[1].data[0].array

    fluxes = recs['peak_flux_wide']

    mask = fluxes > flux_limit
    filtered_recs = recs[mask]

    decs = filtered_recs['DEJ2000']
    mask = decs > decmin
    filtered_recs = filtered_recs[mask]

    decs = filtered_recs['DEJ2000']
    mask = decs < decmax
    filtered_recs = filtered_recs[mask]

    ras = filtered_recs['RAJ2000']
    decs = filtered_recs['DEJ2000']
    names = filtered_recs['Name']

    if polarisation_frame is None:
        polarisation_frame = PolarisationFrame("stokesI")

    npol = polarisation_frame.npol

    nchan = len(frequency)

    # For every source, we read all measured fluxes and interpolate to the
    # required frequencies
    gleam_freqs = numpy.array([76, 84, 92, 99, 107, 115, 122, 130, 143, 151, 158, 166, 174, 181, 189, 197, 204,
                               212, 220, 227])
    gleam_flux_freq = numpy.zeros([len(names), len(gleam_freqs)])
    for i, f in enumerate(gleam_freqs):
        gleam_flux_freq[:, i] = filtered_recs['int_flux_%03d' % (f)][:]

    skycomps = []

    directions = SkyCoord(ra=ras * u.deg, dec=decs * u.deg)
    if phasecentre is not None:
        separations = directions.separation(phasecentre).to('rad').value
    else:
        separations = numpy.zeros(len(names))

    for isource, name in enumerate(names):
        direction = directions[isource]
        if separations[isource] < radius:

            fint = interpolate.interp1d(gleam_freqs * 1.0e6, gleam_flux_freq[isource, :], kind=kind)
            flux = numpy.zeros([nchan, npol])
            flux[:, 0] = fint(frequency)
            if not numpy.isnan(flux).any():
                skycomps.append(Skycomponent(direction=direction, flux=flux, frequency=frequency,
                                             name=name, shape='Point',
                                             polarisation_frame=polarisation_frame))

    log.info('create_low_test_skycomponents_from_gleam: %d sources above flux limit %.3f' % (len(skycomps), flux_limit))

    hdulist.close()

    return skycomps


def replicate_image(im: Image, polarisation_frame=PolarisationFrame('stokesI'), frequency=numpy.array([1e8])) \
        -> Image:
    """ Make a new canonical shape Image, extended along third and fourth axes by replication.

    The order of the data is [chan, pol, dec, ra]


    :param frequency:
    :param im:
    :param polarisation_frame: Polarisation_frame
    :return: Image
    """

    if len(im.data.shape) == 2:
        fim = Image()

        newwcs = WCS(naxis=4)

        newwcs.wcs.crpix = [im.wcs.wcs.crpix[0] + 1.0, im.wcs.wcs.crpix[1] + 1.0, 1.0, 1.0]
        newwcs.wcs.cdelt = [im.wcs.wcs.cdelt[0], im.wcs.wcs.cdelt[1], 1.0, 1.0]
        newwcs.wcs.crval = [im.wcs.wcs.crval[0], im.wcs.wcs.crval[1], 1.0, frequency[0]]
        newwcs.wcs.ctype = [im.wcs.wcs.ctype[0], im.wcs.wcs.ctype[1], 'STOKES', 'FREQ']

        nchan = len(frequency)
        npol = polarisation_frame.npol
        fim.polarisation_frame = polarisation_frame

        fim.wcs = newwcs
        fshape = [nchan, npol, im.data.shape[1], im.data.shape[0]]
        fim.data = numpy.zeros(fshape)
        log.info("replicate_image: replicating shape %s to %s" % (im.data.shape, fim.data.shape))
        for i3 in range(nchan):
            fim.data[i3, 0, :, :] = im.data[:, :]
        return fim
    else:
        return im


def create_blockvisibility_iterator(config: Configuration, times: numpy.array, frequency: numpy.array,
                                    channel_bandwidth, phasecentre: SkyCoord, weight: float = 1,
                                    polarisation_frame=PolarisationFrame('stokesI'), integration_time=1.0,
                                    number_integrations=1, predict=predict_2d, model=None, components=None,
                                    phase_error=0.0, amplitude_error=0.0, sleep=0.0, **kwargs):
    """ Create a sequence of Visibilities and optionally predicting and coalescing

    This is useful mainly for performing large simulations. Do something like::
    
        vis_iter = create_blockvisibility_iterator(config, times, frequency, channel_bandwidth, phasecentre=phasecentre,
                                              weight=1.0, integration_time=30.0, number_integrations=3)

        for i, vis in enumerate(vis_iter):
        if i == 0:
            fullvis = vis
        else:
            fullvis = append_visibility(fullvis, vis)


    :param config: Configuration of antennas
    :param times: hour angles in radians
    :param frequency: frequencies (Hz] Shape [nchan]
    :param weight: weight of a single sample
    :param phasecentre: phasecentre of observation
    :param npol: Number of polarizations
    :param integration_time: Integration time ('auto' or value in s)
    :param number_integrations: Number of integrations to be created at each time.
    :param model: Model image to be inserted
    :param components: Components to be inserted
    :param sleep_time: Time to sleep between yields
    :return: Visibility

    """
    for time in times:
        actualtimes = time + numpy.arange(0, number_integrations) * integration_time * numpy.pi / 43200.0
        bvis = create_blockvisibility(config, actualtimes, frequency=frequency, phasecentre=phasecentre, weight=weight,
                                      polarisation_frame=polarisation_frame, integration_time=integration_time,
                                      channel_bandwidth=channel_bandwidth)

        if model is not None:
            vis = convert_blockvisibility_to_visibility(bvis)
            vis = predict(vis, model, **kwargs)
            bvis = convert_visibility_to_blockvisibility(vis)

        if components is not None:
            vis = dft_skycomponent_visibility(bvis, components)

        # Add phase errors
        if phase_error > 0.0 or amplitude_error > 0.0:
            gt = create_gaintable_from_blockvisibility(bvis)
            gt = simulate_gaintable(gt=gt, phase_error=phase_error, amplitude_error=amplitude_error)
            bvis = apply_gaintable(bvis, gt)

        import time
        time.sleep(sleep)

        yield bvis


def simulate_gaintable(gt: GainTable, phase_error=0.1, amplitude_error=0.0, smooth_channels=1, leakage=0.0,
                       **kwargs) -> GainTable:
    """ Simulate a gain table

    :type gt: GainTable
    :param phase_error: std of normal distribution, zero mean
    :param amplitude_error: std of log normal distribution
    :param leakage: std of cross hand leakage
    :param smooth_channels: Use bspline over smooth_channels
    :param kwargs:
    :return: Gaintable

    """

    def moving_average(a, n=3):
        return numpy.convolve(a, numpy.ones((n,)) / n, mode='valid')

    log.debug("simulate_gaintable: Simulating amplitude error = %.4f, phase error = %.4f"
              % (amplitude_error, phase_error))
    amps = 1.0
    phases = 1.0
    ntimes, nant, nchan, nrec, _ = gt.data['gain'].shape
    if phase_error > 0.0:
        phases = numpy.zeros(gt.data['gain'].shape)
        for time in range(ntimes):
            for ant in range(nant):
                phase = numpy.random.normal(0, phase_error, nchan + int(smooth_channels) - 1)
                if smooth_channels > 1:
                    phase = moving_average(phase, smooth_channels)
                phases[time, ant, ...] = phase[..., numpy.newaxis, numpy.newaxis]

    if amplitude_error > 0.0:
        amps = numpy.ones(gt.data['gain'].shape, dtype='complex')
        for time in range(ntimes):
            for ant in range(nant):
                amp = numpy.random.lognormal(mean=0.0, sigma=amplitude_error, size=nchan + int(smooth_channels) - 1)
                if smooth_channels > 1:
                    amp = moving_average(amp, smooth_channels)
                    amp = amp / numpy.average(amp)
                amps[time, ant, ...] = amp[..., numpy.newaxis, numpy.newaxis]

    gt.data['gain'] = amps * numpy.exp(0 + 1j * phases)
    nrec = gt.data['gain'].shape[-1]
    if nrec > 1:
        if leakage > 0.0:
            leak = numpy.random.normal(0, leakage, gt.data['gain'][..., 0, 0].shape) + 1j * \
                   numpy.random.normal(0, leakage, gt.data['gain'][..., 0, 0].shape)
            gt.data['gain'][..., 0, 1] = gt.data['gain'][..., 0, 0] * leak
            leak = numpy.random.normal(0, leakage, gt.data['gain'][..., 1, 1].shape) + 1j * \
                   numpy.random.normal(0, leakage, gt.data['gain'][..., 1, 1].shape)
            gt.data['gain'][..., 1, 0] = gt.data['gain'][..., 1, 1] * leak
        else:
            gt.data['gain'][..., 0, 1] = 0.0
            gt.data['gain'][..., 1, 0] = 0.0

    return gt


def ingest_unittest_visibility(config, frequency, channel_bandwidth, times, vis_pol, phasecentre, block=False,
                               zerow=False):
    if block:
        vt = create_blockvisibility(config, times, frequency, channel_bandwidth=channel_bandwidth,
                                    phasecentre=phasecentre, weight=1.0, polarisation_frame=vis_pol, zerow=zerow)
    else:
        vt = create_visibility(config, times, frequency, channel_bandwidth=channel_bandwidth,
                               phasecentre=phasecentre, weight=1.0, polarisation_frame=vis_pol, zerow=zerow)
    vt.data['vis'][...] = 0.0
    return vt


def create_unittest_components(model, flux, applypb=False, telescope='LOW', npixel=None,
                               scale=1.0, single=False, symmetric=False, angular_scale=1.0):
    # Fill the visibility with exactly computed point sources.

    if npixel is None:
        _, _, _, npixel = model.data.shape
    spacing_pixels = int(scale * npixel) // 4
    log.info('Spacing in pixels = %s' % spacing_pixels)

    if not symmetric:
        centers = [(0.2 * angular_scale, 1.1 * angular_scale)]
    else:
        centers = list()

    if not single:
        centers.append([0.0, 0.0])

        for x in numpy.linspace(-1.2 * angular_scale, 1.2 * angular_scale, 7):
            if abs(x) > 1e-15:
                centers.append([x, x])
                centers.append([x, -x])
    model_pol = model.polarisation_frame
    # Make the list of components
    rpix = model.wcs.wcs.crpix
    components = []
    for center in centers:
        ix, iy = center
        # The phase center in 0-relative coordinates is n // 2 so we centre the grid of
        # components on ny // 2, nx // 2. The wcs must be defined consistently.
        p = int(round(rpix[0] + ix * spacing_pixels * numpy.sign(model.wcs.wcs.cdelt[0]))), \
            int(round(rpix[1] + iy * spacing_pixels * numpy.sign(model.wcs.wcs.cdelt[1])))
        sc = pixel_to_skycoord(p[0], p[1], model.wcs, origin=1)
        log.info("Component at (%f, %f) [0-rel] %s" % (p[0], p[1], str(sc)))

        # Channel images
        comp = create_skycomponent(direction=sc, flux=flux, frequency=model.frequency, polarisation_frame=model_pol)
        components.append(comp)

    if applypb:
        beam = create_pb(model, telescope=telescope, use_local=False)
        components = apply_beam_to_skycomponent(components, beam)

    return components


def create_unittest_model(vis, model_pol, npixel=None, cellsize=None, nchan=1):
    advice = advise_wide_field(vis, guard_band_image=2.0, delA=0.02, facets=1,
                               wprojection_planes=1, oversampling_synthesised_beam=4.0)
    if cellsize is None:
        cellsize = advice['cellsize']
    if npixel is None:
        npixel = advice['npixels2']
    model = create_image_from_visibility(vis, npixel=npixel, cellsize=cellsize, nchan=nchan,
                                         polarisation_frame=model_pol)
    return model


def insert_unittest_errors(vt, seed=180555, calibration_context="TG", amp_errors=None, phase_errors=None):
    """Simulate gain errors and apply
    
    :param vt:
    :param seed: Random number seed, set to big integer repeat values from run to run
    :param phase_errors: e.g. {'T': 1.0, 'G': 0.1, 'B': 0.01}
    :param amp_errors: e.g. {'T': 0.0, 'G': 0.01, 'B': 0.01}
    :return:
    """
    controls = create_calibration_controls()

    if amp_errors is None:
        amp_errors = {'T': 0.0, 'G': 0.01, 'B': 0.01}

    if phase_errors is None:
        phase_errors = {'T': 1.0, 'G': 0.1, 'B': 0.01}

    for c in calibration_context:
        gaintable = create_gaintable_from_blockvisibility(vt, timeslice=controls[c]['timeslice'])
        gaintable = simulate_gaintable(gaintable, phase_error=phase_errors[c], amplitude_error=amp_errors[c],
                                       timeslice=controls[c]['timeslice'], phase_only=controls[c]['phase_only'],
                                       crosspol=controls[c]['shape'] == 'matrix')

        vt = apply_gaintable(vt, gaintable, inverse=True, timeslice=controls[c]['timeslice'])

    return vt
