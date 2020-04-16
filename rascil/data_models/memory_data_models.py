"""The data models used in RASCIL:

"""

__all__ = ['Configuration',
           'GainTable',
           'PointingTable',
           'Image',
           'GridData',
           'ConvolutionFunction',
           'Skycomponent',
           'SkyModel',
           'Visibility',
           'BlockVisibility',
           'FlagTable',
           'QA',
           'ScienceDataModel',
           'assert_same_chan_pol',
           'assert_vis_gt_compatible'
           ]



import logging
import sys
import warnings
from copy import deepcopy
from typing import Union

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.utils.exceptions import AstropyDeprecationWarning
from astropy.wcs import FITSFixedWarning

warnings.simplefilter('ignore', FITSFixedWarning)
warnings.simplefilter('ignore', AstropyDeprecationWarning)

from rascil.data_models.polarisation import PolarisationFrame, ReceptorFrame

log = logging.getLogger('logger')


class Configuration:
    """ Describe a Configuration as locations in x,y,z, mount type, diameter, names, and
        overall location
    """
    
    def __init__(self, name='', data=None, location=None,
                 names="%s", xyz=None, mount="alt-az", frame="",
                 receptor_frame=ReceptorFrame("linear"),
                 diameter=None):
        
        """Configuration object describing data for processing

        :param name: Name of configuration e.g. 'LOWR3'
        :param data: Data array (used in copying)
        :param location: Location of array as an astropy EarthLocation
        :param names: Names of the dishes/stations
        :param xyz: Geocentric coordinates of dishes/stations
        :param mount: Mount types of dishes/stations 'altaz' | 'xy' | 'equatorial'
        :param frame: Reference frame of locations
        :param receptor_frame: Receptor frame
        :param diameter: Diameters of dishes/stations (m)
        """
        if data is None and xyz is not None:
            desc = [('names', 'U12'),
                    ('xyz', 'f8', (3,)),
                    ('diameter', 'f8'),
                    ('mount', 'U5')]
            nants = xyz.shape[0]
            if isinstance(names, str):
                names = [names % ant for ant in range(nants)]
            if isinstance(mount, str):
                mount = numpy.repeat(mount, nants)
            data = numpy.zeros(shape=[nants], dtype=desc)
            data['names'] = names
            data['xyz'] = xyz
            data['mount'] = mount
            data['diameter'] = diameter
        
        self.name = name
        self.data = data
        self.location = location
        self.frame = frame
        self.receptor_frame = receptor_frame
    
    def __str__(self):
        """Default printer for Configuration

        """
        s = "Configuration:\n"
        s += "\nName: %s\n" % self.name
        s += "\tNumber of antennas/stations: %s\n" % len(self.names)
        s += "\tNames: %s\n" % self.names
        s += "\tDiameter: %s\n" % self.diameter
        s += "\tMount: %s\n" % self.mount
        s += "\tXYZ: %s\n" % self.xyz
        
        return s
    
    def size(self):
        """ Return size in GB
        """
        size = 0
        size += self.data.size * sys.getsizeof(self.data)
        return size / 1024.0 / 1024.0 / 1024.0
    
    @property
    def names(self):
        """ Names of the dishes/stations"""
        return self.data['names']
    
    @property
    def diameter(self):
        """ Diameter of dishes/stations (m)
        """
        return self.data['diameter']
    
    @property
    def xyz(self):
        """ XYZ locations of dishes/stations [:,3] (m)
        """
        return self.data['xyz']
    
    @property
    def mount(self):
        """ Mount types of dishes/stations ('azel' | 'equatorial'
        """
        return self.data['mount']


class GainTable:
    """ Gain table with data_models: time, antenna, gain[:, chan, rec, rec], weight columns

    The weight is usually that output from gain solvers.
    """
    
    def __init__(self, data=None, gain: numpy.array = None, time: numpy.array = None, interval=None,
                 weight: numpy.array = None, residual: numpy.array = None, frequency: numpy.array = None,
                 receptor_frame: ReceptorFrame = ReceptorFrame("linear"), phasecentre=None, configuration=None):
        """ Create a gaintable from arrays

        The definition of gain is:

            Vobs = g_i g_j^* Vmodel

        :param data: Structured data (used in copying)
        :param gain: Complex gain [nrows, nants, nchan, nrec, nrec]
        :param time: Centroid of solution [nrows]
        :param interval: Interval of validity
        :param weight: Weight of gain [nrows, nchan, nrec, nrec]
        :param residual: Residual of fit [nchan, nrec, nrec]
        :param frequency: Frequency [nchan]
        :param receptor_frame: Receptor frame
        :param phasecentre: Phasecentre (SkyCoord)
        :param configuration: Configuration
        """
        if data is None and gain is not None:
            nrec = receptor_frame.nrec
            nrows = gain.shape[0]
            nants = gain.shape[1]
            nchan = gain.shape[2]
            assert len(frequency) == nchan, "Discrepancy in frequency channels"
            desc = [('gain', 'c16', (nants, nchan, nrec, nrec)),
                    ('weight', 'f8', (nants, nchan, nrec, nrec)),
                    ('residual', 'f8', (nchan, nrec, nrec)),
                    ('time', 'f8'),
                    ('interval', 'f8')]
            data = numpy.zeros(shape=[nrows], dtype=desc)
            data['gain'] = gain
            data['weight'] = weight
            data['time'] = time
            data['interval'] = interval
            data['residual'] = residual
        
        self.data = data
        self.frequency = frequency
        self.receptor_frame = receptor_frame
        self.phasecentre = phasecentre
        self.configuration = configuration
    
    def size(self):
        """ Return size in GB
        """
        size = 0
        size += self.data.size * sys.getsizeof(self.data)
        return size / 1024.0 / 1024.0 / 1024.0
    
    @property
    def time(self):
        """ Centroid of solution [nrows]
        """
        return self.data['time']
    
    @property
    def interval(self):
        """ Interval of validity [nrows]
        """
        return self.data['interval']
    
    @property
    def gain(self):
        """ Complex gain [nrows, nants, nchan, nrec, nrec]
        """
        return self.data['gain']
    
    @property
    def weight(self):
        """ Weight of gain [nrows, nchan, nrec, nrec]

        """
        return self.data['weight']
    
    @property
    def residual(self):
        """ Residual of fit [nchan, nrec, nrec]
        """
        return self.data['residual']
    
    @property
    def ntimes(self):
        """ Number of times (i.e. rows) in this table
        """
        return self.data['gain'].shape[0]
    
    @property
    def nants(self):
        """ Number of dishes/stations
        """
        return self.data['gain'].shape[1]
    
    @property
    def nchan(self):
        """ Number of channels
        """
        return self.data['gain'].shape[2]
    
    @property
    def nrec(self):
        """ Number of receivers

        """
        return self.receptor_frame.nrec
    
    def __str__(self):
        """Default printer for GainTable

        """
        s = "GainTable:\n"
        s += "\tTimes: %s\n" % str(self.ntimes)
        s += "\tData shape: %s\n" % str(self.data.shape)
        s += "\tReceptor frame: %s\n" % str(self.receptor_frame.type)
        s += "\tPhasecentre: %s\n" % str(self.phasecentre)
        return s


class PointingTable:
    """ Pointing table with data_models: time, antenna, offset[:, chan, rec, 2], weight columns

    The weight is usually that output from gain solvers.
    """
    
    def __init__(self, data=None, pointing: numpy.array = None, nominal: numpy.array = None,
                 time: numpy.array = None, interval=None,
                 weight: numpy.array = None, residual: numpy.array = None, frequency: numpy.array = None,
                 receptor_frame: ReceptorFrame = ReceptorFrame("linear"), pointing_frame: str = "local",
                 pointingcentre=None, configuration=None):
        """ Create a pointing table from arrays

        :param data: Structured data (used in copying)
        :param pointing: Pointing (rad) [:, nants, nchan, nrec, 2]
        :param nominal: Nominal pointing (rad) [:, nants, nchan, nrec, 2]
        :param time: Centroid of solution [:]
        :param interval: Interval of validity
        :param weight: Weight [: nants, nchan, nrec]
        :param residual: Residual [: nants, nchan, nrec, 2]
        :param frequency: [nchan]
        :param receptor_frame: e.g. Receptor_frame("linear")
        :param pointing_frame: Pointing frame
        :param pointingcentre: SkyCoord
        :param configuration: Configuration
        """
        if data is None and pointing is not None:
            nrec = receptor_frame.nrec
            nrows = pointing.shape[0]
            nants = pointing.shape[1]
            nchan = pointing.shape[2]
            assert len(frequency) == nchan, "Discrepancy in frequency channels"
            desc = [('pointing', 'f16', (nants, nchan, nrec, 2)),
                    ('nominal', 'f16', (nants, nchan, nrec, 2)),
                    ('weight', 'f8', (nants, nchan, nrec, 2)),
                    ('residual', 'f8', (nchan, nrec, 2)),
                    ('time', 'f8'),
                    ('interval', 'f8')]
            data = numpy.zeros(shape=[nrows], dtype=desc)
            data['pointing'] = pointing
            data['weight'] = weight
            data['time'] = time
            data['interval'] = interval
            data['residual'] = residual
            data['nominal'] = nominal
        
        self.data = data
        self.frequency = frequency
        self.receptor_frame = receptor_frame
        self.pointing_frame = pointing_frame
        self.pointingcentre = pointingcentre
        self.configuration = configuration
    
    def size(self):
        """ Return size in GB
        """
        size = 0
        size += self.data.size * sys.getsizeof(self.data)
        return size / 1024.0 / 1024.0 / 1024.0
    
    @property
    def time(self):
        """ Time (s UTC) [:]
        """
        return self.data['time']
    
    @property
    def interval(self):
        """ Interval of validity (s) [:]
        """
        return self.data['interval']
    
    @property
    def nominal(self):
        """ Nominal pointing (rad) [:, nants, nchan, nrec, 2]
        """
        return self.data['nominal']
    
    @property
    def pointing(self):
        """ Pointing (rad) [:, nants, nchan, nrec, 2]
        """
        return self.data['pointing']
    
    @property
    def weight(self):
        """ Weight [: nants, nchan, nrec]
        """
        return self.data['weight']
    
    @property
    def residual(self):
        """ Residual [: nants, nchan, nrec, 2]
        """
        return self.data['residual']
    
    @property
    def ntimes(self):
        """ Number of time (i.e. rows in table)"""
        return self.data['pointing'].shape[0]
    
    @property
    def nants(self):
        """ Number of dishes/stations
        """
        return self.data['pointing'].shape[1]
    
    @property
    def nchan(self):
        """ Number of channels
        """
        return self.data['pointing'].shape[2]
    
    @property
    def nrec(self):
        """ Number of receptors
        """
        return self.receptor_frame.nrec
    
    def __str__(self):
        """Default printer for PointingTable

        """
        s = "PointingTable:\n"
        s += "\tTimes: %s\n" % str(self.ntimes)
        s += "\tData shape: %s\n" % str(self.data.shape)
        s += "\tReceptor frame: %s\n" % str(self.receptor_frame.type)
        s += "\tPointing centre: %s\n" % str(self.pointingcentre)
        s += "\tConfiguration: %s\n" % str(self.configuration)
        return s


class Image:
    """Image class with Image data (as a numpy.array) and the AstroPy `implementation of
    a World Coodinate System <http://docs.astropy.org/en/stable/wcs>`_

    Many operations can be done conveniently using numpy processing_components on Image.data_models.

    Most of the imaging processing_components require an image in canonical format:
    - 4 axes: RA, DEC, POL, FREQ

    The conventions for indexing in WCS and numpy are opposite.
    - In astropy.wcs, the order is (longitude, latitude, polarisation, frequency)
    - in numpy, the order is (frequency, polarisation, latitude, longitude)

    .. warning::
        The polarisation_frame is kept in two places, the WCS and the polarisation_frame
        variable. The latter should be considered definitive.

    """
    
    def __init__(self, data=None, wcs=None, polarisation_frame=None):
        """ Create Image

        :param data: Data for image
        :param wcs: Astropy WCS object
        :param polarisation_frame: e.g. PolarisationFrame('stokesIQUV')
        """
        self.data = data
        self.wcs = wcs
        self.polarisation_frame = polarisation_frame
    
    def size(self):
        """ Return size in GB
        """
        size = 0
        size += self.data.nbytes
        return size / 1024.0 / 1024.0 / 1024.0
    
    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result
    
    # noinspection PyArgumentList
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result
    
    @property
    def nchan(self):
        """ Number of channels
        """
        return self.data.shape[0]
    
    @property
    def npol(self):
        """ Number of polarisations
        """
        return self.data.shape[1]
    
    @property
    def nheight(self):
        """ Number of pixels height i.e. y
        """
        return self.data.shape[2]
    
    @property
    def nwidth(self):
        """ Number of pixels width i.e. x
        """
        return self.data.shape[3]
    
    @property
    def frequency(self):
        """ Frequency values
        """
        w = self.wcs.sub(['spectral'])
        return w.wcs_pix2world(range(self.nchan), 0)[0]
    
    @property
    def shape(self):
        """ Shape of data array
        """
        return self.data.shape
    
    @property
    def phasecentre(self):
        """ Phasecentre (from WCS)
        """
        return SkyCoord(self.wcs.wcs.crval[0] * u.deg, self.wcs.wcs.crval[1] * u.deg)
    
    def __str__(self):
        """Default printer for Image

        """
        s = "Image:\n"
        s += "\tShape: %s\n" % str(self.data.shape)
        s += "\tData type: %s\n" % str(self.data.dtype)
        s += "\tWCS: %s\n" % self.wcs.__repr__()
        s += "\tPolarisation frame: %s\n" % str(self.polarisation_frame.type)
        return s


class GridData:
    """Class to hold Gridded data for Fourier processing
    - Has four or more coordinates: [chan, pol, z, y, x] where x can be u, l; y can be v, m; z can be w, n

    The conventions for indexing in WCS and numpy are opposite.
    - In astropy.wcs, the order is (longitude, latitude, polarisation, frequency)
    - in numpy, the order is (frequency, polarisation, depth, latitude, longitude)

    .. warning::
        The polarisation_frame is kept in two places, the WCS and the polarisation_frame
        variable. The latter should be considered definitive.

    """
    
    def __init__(self, data=None, grid_wcs=None, projection_wcs=None, polarisation_frame=None):
        """Create Griddata
        
        :param data: Data for griddata
        :param grid_wcs: Astropy WCS object for the grid
        :param projection_wcs: Astropy WCS object for the projection
        :param polarisation_frame: Polarisation_frame e.g. PolarisationFrame('linear')
        """
        self.data = data
        self.grid_wcs = grid_wcs
        self.projection_wcs = projection_wcs
        self.polarisation_frame = polarisation_frame
    
    def size(self):
        """ Return size in GB
        """
        size = 0
        size += self.data.nbytes
        return size / 1024.0 / 1024.0 / 1024.0
    
    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result
    
    # noinspection PyArgumentList
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result
    
    @property
    def nchan(self):
        """ Number of channels
        """
        return self.data.shape[0]
    
    @property
    def npol(self):
        """ Number of polarisations
        """
        return self.data.shape[1]
    
    @property
    def nheight(self):
        """ Number of pixels height i.e. y
        """
        return self.data.shape[2]
    
    @property
    def nwidth(self):
        """ Number of pixels width i.e. x
        """
        return self.data.shape[3]
    
    @property
    def ndepth(self):
        """ Number of pixels deep i.e. z
        """
        return self.data.shape[4]
    
    @property
    def frequency(self):
        """ Frequency values
        """
        w = self.grid_wcs.sub(['spectral'])
        return w.wcs_pix2world(range(self.nchan), 0)[0]
    
    @property
    def shape(self):
        """ Shape of data array
        """
        assert len(self.data.shape) == 5
        return self.data.shape
    
    @property
    def phasecentre(self):
        """ Phasecentre (from projection WCS)

        """
        return SkyCoord(self.projection_wcs.wcs.crval[0] * u.deg, self.projection_wcs.wcs.crval[1] * u.deg)
    
    def __str__(self):
        """Default printer for GriddedData

        """
        s = "Gridded data:\n"
        s += "\tShape: %s\n" % str(self.data.shape)
        s += "\tGrid WCS: %s\n" % self.grid_wcs
        s += "\tProjection WCS: %s\n" % self.projection_wcs
        s += "\tPolarisation frame: %s\n" % str(self.polarisation_frame.type)
        return s


class ConvolutionFunction:
    """Class to hold Convolution function for Fourier processing
    - Has four or more coordinates: [chan, pol, z, y, x] where x can be u, l; y can be v, m; z can be w, n

    The cf has axes [chan, pol, z, dy, dx, y, x] where z, y, x are spatial axes in either sky or Fourier plane. The
    order in the WCS is reversed so the grid_WCS describes UU, VV, WW, STOKES, FREQ axes

    The axes UU,VV have the same physical stride as the image, The axes DUU, DVV are subsampled.

    Convolution function holds the original sky plane projection in the projection_wcs.

    """
    
    def __init__(self, data=None, grid_wcs=None, projection_wcs=None, polarisation_frame=None):
        """Create ConvolutionFunction

        :param data: Data for cf
        :param grid_wcs: Astropy WCS object for the grid
        :param projection_wcs: Astropy WCS object for the projection
        :param polarisation_frame: Polarisation_frame e.g. PolarisationFrame('linear')
        """
        self.data = data
        self.grid_wcs = grid_wcs
        self.projection_wcs = projection_wcs
        self.polarisation_frame = polarisation_frame
    
    def size(self):
        """ Return size in GB
        """
        size = 0
        size += self.data.nbytes
        return size / 1024.0 / 1024.0 / 1024.0
    
    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result
    
    # noinspection PyArgumentList
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result
    
    @property
    def nchan(self):
        """ Number of channels
        """
        return self.data.shape[0]
    
    @property
    def npol(self):
        """ Number of polarisations
        """
        return self.data.shape[1]
    
    @property
    def nheight(self):
        """ Number of pixels height i.e. y
        """
        return self.data.shape[2]
    
    @property
    def nwidth(self):
        def nwidth(self):
            """ Number of pixels width i.e. x
            """

        return self.data.shape[3]
    
    @property
    def ndepth(self):
        """ Number of pixels deep i.e. z
        """
        return self.data.shape[4]
    
    @property
    def frequency(self):
        """ Frequency values
        """
        w = self.grid_wcs.sub(['spectral'])
        return w.wcs_pix2world(range(self.nchan), 0)[0]
    
    @property
    def shape(self):
        """ Shape of data array
        """
        assert len(self.data.shape) == 7
        return self.data.shape
    
    @property
    def phasecentre(self):
        """ Phasecentre (from projection WCS)
        """
        return SkyCoord(self.projection_wcs.wcs.crval[0] * u.deg, self.projection_wcs.wcs.crval[1] * u.deg)
    
    def __str__(self):
        """Default printer for ConvolutionFunction

        """
        s = "Convolution function:\n"
        s += "\tShape: %s\n" % str(self.data.shape)
        s += "\tGrid WCS: %s\n" % self.grid_wcs
        s += "\tProjection WCS: %s\n" % self.projection_wcs
        s += "\tPolarisation frame: %s\n" % str(self.polarisation_frame.type)
        return s


class Skycomponent:
    """Skycomponents are used to represent compact sources on the sky. They possess direction,
    flux as a function of frequency and polarisation, shape (with params), and polarisation frame.

    For example, the following creates and predicts the visibility from a collection of point sources
    drawn from the GLEAM catalog::

        sc = create_low_test_skycomponents_from_gleam(flux_limit=1.0,
                                                    polarisation_frame=PolarisationFrame("stokesIQUV"),
                                                    frequency=frequency, kind='cubic',
                                                    phasecentre=phasecentre,
                                                    radius=0.1)
        model = create_image_from_visibility(vis, cellsize=0.001, npixel=512, frequency=frequency,
                                            polarisation_frame=PolarisationFrame('stokesIQUV'))

        bm = create_low_test_beam(model=model)
        sc = apply_beam_to_skycomponent(sc, bm)
        vis = dft_skycomponent_visibility(vis, sc)
    """
    
    def __init__(self,
                 direction=None, frequency=None, name=None, flux=None, shape='Point',
                 polarisation_frame=PolarisationFrame('stokesIQUV'), params=None):
        """ Define the required structure

        :param direction: SkyCoord
        :param frequency: numpy.array [nchan]
        :param name: user friendly name
        :param flux: numpy.array [nchan, npol]
        :param shape: str e.g. 'Point' 'Gaussian'
        :param polarisation_frame: Polarisation_frame e.g. PolarisationFrame('stokesIQUV')
        :param params: numpy.array shape dependent parameters
        """
        
        self.direction = direction
        self.frequency = numpy.array(frequency)
        self.name = name
        self.flux = numpy.array(flux)
        self.shape = shape
        if params is None:
            params = {}
        self.params = params
        self.polarisation_frame = polarisation_frame
        
        assert len(self.frequency.shape) == 1, frequency
        assert len(self.flux.shape) == 2, flux
        assert self.frequency.shape[0] == self.flux.shape[0], \
            "Frequency shape %s, flux shape %s" % (self.frequency.shape, self.flux.shape)
        assert polarisation_frame.npol == self.flux.shape[1], \
            "Polarisation is %s, flux shape %s" % (polarisation_frame.type, self.flux.shape)
    
    @property
    def nchan(self):
        """ Number of channels
        """
        return self.flux.shape[0]
    
    @property
    def npol(self):
        """ Number of polarisations
        """
        return self.flux.shape[1]
    
    def __str__(self):
        """Default printer for Skycomponent

        """
        s = "Skycomponent:\n"
        s += "\tName: %s\n" % self.name
        s += "\tFlux: %s\n" % self.flux
        s += "\tFrequency: %s\n" % self.frequency
        s += "\tDirection: %s\n" % self.direction
        s += "\tShape: %s\n" % self.shape
        
        s += "\tParams: %s\n" % self.params
        s += "\tPolarisation frame: %s\n" % str(self.polarisation_frame.type)
        return s


class SkyModel:
    """ A model for the sky, including an image, components, gaintable and a mask
    """
    
    def __init__(self, image=None, components=None, gaintable=None, mask=None, fixed=False):
        """ A model of the sky as an image, components, gaintable and a mask

        Use copy_skymodel to make a proper copy of skymodel
        :param image: Image
        :param components: List of components
        :param gaintable: Gaintable for this skymodel
        :param mask: Mask for the image
        :param fixed: Is this model fixed?
        """
        if components is None:
            components = []
        
        if image is not None:
            assert isinstance(image, Image), image
        self.image = image
        
        if components is not None:
            assert isinstance(components, list)
            for comp in components:
                assert isinstance(comp, Skycomponent), comp
        self.components = [sc for sc in components]
        
        if gaintable is not None:
            assert isinstance(gaintable, GainTable), gaintable
        self.gaintable = gaintable
        
        if mask is not None:
            assert isinstance(mask, Image), mask
        self.mask = mask
        
        self.fixed = fixed
    
    def __str__(self):
        """Default printer for SkyModel

        """
        s = "SkyModel: fixed: %s\n" % self.fixed
        for i, sc in enumerate(self.components):
            s += str(sc)
        s += "\n"
        
        s += str(self.image)
        s += "\n"
        
        s += str(self.mask)
        s += "\n"
        
        s += str(self.gaintable)
        
        return s


class Visibility:
    """ Visibility table class

    Visibility with uvw, time, integration_time, frequency, channel_bandwidth, a1, a2, vis, weight
    as separate columns in a numpy structured array, The fundemental unit is a complex vector of polarisation.

    Visibility is defined to hold an observation with one direction.
    Polarisation frame is the same for the entire data set and can be stokes, circular, linear
    The configuration is also an attribute

    The phasecentre is the direct of delay tracking i.e. n=0. If uvw are rotated then this
    should be updated with the new delay tracking centre. This is important for wstack and wproject
    algorithms.

    If a visibility is created by coalescence then the cindex column is filled with a pointer to the
    row in the original block visibility that this row has a value for. The original blockvisibility
    is also preserves as n attribute so that decoalescence is expedited. If you don't need that then
    the storage can be released by setting self.blockvis to None
    
    There are two visibility formats:

    :class:`BlockVisibility` is conceived as an ingest and calibration format. The visibility
    data are kept in a block of shape (number antennas, number antennas, number channels,
    number polarisation). One block is kept per integration. The other columns are time and uvw.
    The sampling in time is therefore the same for all baselines.

    :class:`Visibility` is designed to hold coalesced data where the integration time and
    channel width can vary with baseline length. The visibility data are kept in a visibility
    vector of length equal to the number of polarisations. Everything else is a separate
    column: time, frequency, uvw, channel_bandwidth, integration time.

    """
    
    def __init__(self,
                 data=None, frequency=None, channel_bandwidth=None,
                 phasecentre=None, configuration=None, uvw=None,
                 time=None, antenna1=None, antenna2=None, vis=None, flags=None,
                 weight=None, imaging_weight=None, integration_time=None,
                 polarisation_frame=PolarisationFrame('stokesI'), cindex=None,
                 blockvis=None, source='anonymous', meta=None):
        """Visibility

        :param data: Structured data (used in copying)
        :param frequency: Frequency, one per row
        :param channel_bandwidth: Channel bamdwidth, one per row
        :param phasecentre: Phasecentre (skycoord)
        :param configuration: Configuration
        :param uvw: uvw coordinates [:,3]
        :param time: Time (UTC) per row
        :param antenna1: dish/station index
        :param antenna2: dish/station index
        :param vis: Complex visibility [:, npol]
        :param flags: Flags [:, npol]
        :param weight: Weight [: npol]
        :param imaging_weight: Imaging weight [:, npol]
        :param integration_time: Integration time, per row
        :param polarisation_frame: Polarisation Frame e.g. Polarisation_frame("linear")
        :param cindex: Index of row into original block visibility
        :param blockvis: original block visibility
        :param source: Source name
        :param meta: Meta info
        """
        if data is None and vis is not None:
            if imaging_weight is None:
                imaging_weight = weight
            nvis = vis.shape[0]
            assert len(time) == nvis
            assert len(frequency) == nvis
            assert len(channel_bandwidth) == nvis
            assert len(antenna1) == nvis
            assert len(antenna2) == nvis
            
            npol = polarisation_frame.npol
            desc = [('index', 'i8'),
                    ('uvw', 'f8', (3,)),
                    ('time', 'f8'),
                    ('frequency', 'f8'),
                    ('channel_bandwidth', 'f8'),
                    ('integration_time', 'f8'),
                    ('antenna1', 'i8'),
                    ('antenna2', 'i8'),
                    ('vis', 'c16', (npol,)),
                    ('flags', 'i8', (npol,)),
                    ('weight', 'f8', (npol,)),
                    ('imaging_weight', 'f8', (npol,))]
            data = numpy.zeros(shape=[nvis], dtype=desc)
            data['index'] = list(range(nvis))
            data['uvw'] = uvw
            data['time'] = time
            data['frequency'] = frequency
            data['channel_bandwidth'] = channel_bandwidth
            data['integration_time'] = integration_time
            data['antenna1'] = antenna1
            data['antenna2'] = antenna2
            data['vis'] = vis
            data['flags'] = flags
            data['weight'] = weight
            data['imaging_weight'] = imaging_weight
        
        self.data = data  # numpy structured array
        self.cindex = cindex
        self.blockvis = blockvis
        self.phasecentre = phasecentre  # Phase centre of observation
        self.configuration = configuration  # Antenna/station configuration
        self.polarisation_frame = polarisation_frame
        self.frequency_map = None
        self.source = source
        self.meta = meta
    
    def __str__(self):
        """Default printer for Visibility

        """
        ufrequency = numpy.unique(self.frequency)
        uchannel_bandwidth = numpy.unique(self.channel_bandwidth)
        s = "Visibility:\n"
        s += "\tSource: %s\n" % self.source
        s += "\tNumber of visibilities: %s\n" % self.nvis
        s += "\tNumber of channels: %d\n" % len(ufrequency)
        s += "\tFrequency: %s\n" % str(ufrequency)
        s += "\tChannel bandwidth: %s\n" % str(uchannel_bandwidth)
        s += "\tNumber of polarisations: %s\n" % self.npol
        s += "\tVisibility shape: %s\n" % str(self.vis.shape)
        s += "\tNumber flags: %s\n" % numpy.sum(self.flags)
        s += "\tPolarisation Frame: %s\n" % self.polarisation_frame.type
        s += "\tPhasecentre: %s\n" % self.phasecentre
        s += "\tConfiguration: %s\n" % self.configuration.name
        s += "\tMetadata: %s\n" % self.meta
        
        return s
    
    def size(self):
        """ Return size in GB
        """
        size = 0
        for col in self.data.dtype.fields.keys():
            size += self.data[col].nbytes
        return size / 1024.0 / 1024.0 / 1024.0
    
    @property
    def index(self):
        """ Index into BlockVisibility"""
        return self.data['index']
    
    @property
    def nchan(self):
        """ Number of channels
        """
        return len(self.frequency)

    @property
    def npol(self):
        """ Number of polarisations
        """
        return self.polarisation_frame.npol
    
    @property
    def nvis(self):
        """ Number of visibilities (i.e. rows)
        """
        return self.data['vis'].shape[0]
    
    @property
    def uvw(self):
        """ UVW coordinates (wavelengths) [nrows, 3]
        """
        return self.data['uvw']
    
    @property
    def u(self):
        """ u coordinate (wavelengths) [nrows]
        """
        return self.data['uvw'][:, 0]
    
    @property
    def v(self):
        """ v coordinate (wavelengths) [nrows]
        """
        return self.data['uvw'][:, 1]
    
    @property
    def w(self):
        """ w coordinate (wavelengths) [nrows]
        """
        return self.data['uvw'][:, 2]
    
    @property
    def uvdist(self):
        """ uv distance (wavelengths) [nrows]
        """
        return numpy.hypot(self.u, self.v)
    
    @property
    def uvwdist(self):
        """ uvw distance (wavelengths) [nrows]
        """
        return numpy.hypot(self.u, self.v, self.w)
    
    @property
    def time(self):
        """ Time (UTC) [nrows]
        """
        return self.data['time']
    
    @property
    def integration_time(self):
        """ Integration time [nrows]
        """
        return self.data['integration_time']
    
    @property
    def frequency(self):
        """ Frequency values
        """
        return self.data['frequency']
    
    @property
    def channel_bandwidth(self):
        """ Channel bandwidth values
        """
        return self.data['channel_bandwidth']
    
    @property
    def antenna1(self):
        """ Antenna index
        """
        return self.data['antenna1']
    
    @property
    def antenna2(self):
        """ Antenna index
        """
        return self.data['antenna2']

    @property
    def vis(self):
        """Complex visibility [:, npol]
        """
        return self.data['vis']

    @property
    def flagged_vis(self):
        """Flagged complex visibility [:, npol]
        """
        return self.data['vis'] * (1 - self.flags)

    @property
    def flags(self):
        """flags [:, npol]
        """
        return self.data['flags']

    @property
    def weight(self):
        """Weight [: npol]
        """
        return self.data['weight']

    @property
    def flagged_weight(self):
        """Weight [: npol]
        """
        return self.data['weight'] * (1 - self.data['flags'])

    @property
    def imaging_weight(self):
        """  Imaging weight [:, npol]
        """
        return self.data['imaging_weight']

    @property
    def flagged_imaging_weight(self):
        """  Flagged Imaging weight [:, npol]
        """
        return self.data['imaging_weight'] * (1 - self.data['flags'])


class BlockVisibility:
    """ Block Visibility table class

    BlockVisibility with uvw, time, integration_time, frequency, channel_bandwidth, pol,
    a1, a2, vis, weight Columns in a numpy structured array.

    BlockVisibility is defined to hold an observation with one direction.

    The phasecentre is the direct of delay tracking i.e. n=0. If uvw are rotated then this
    should be updated with the new delay tracking centre. This is important for wstack and wproject
    algorithms.

    Polarisation frame is the same for the entire data set and can be stokesI, circular, linear

    The configuration is also an attribute
    """
    
    def __init__(self,
                 data=None, frequency=None, channel_bandwidth=None,
                 phasecentre=None, configuration=None, uvw=None,
                 time=None, vis=None, weight=None, integration_time=None,
                 flags=None,
                 polarisation_frame=PolarisationFrame('stokesI'),
                 imaging_weight=None, source='anonymous', meta=None):
        """BlockVisibility

        :param data: Structured data (used in copying)
        :param frequency: Frequency [nchan]
        :param channel_bandwidth: Channel bandwidth [nchan]
        :param phasecentre: Phasecentre (SkyCoord)
        :param configuration: Configuration
        :param uvw: UVW coordinates (m) [:, nant, nant, 3]
        :param time: Time (UTC) [:]
        :param vis: Complex visibility [:, nant, nant, nchan, npol]
        :param flags: Flags [:, nant, nant, nchan]
        :param weight: [:, nant, nant, nchan, npol]
        :param imaging_weight: [:, nant, nant, nchan, npol]
        :param integration_time: Integration time [:]
        :param polarisation_frame: Polarisation_Frame e.g. Polarisation_Frame("linear")
        :param source: Source name
        :param meta: Meta info
        """
        if meta is None:
            meta = dict()
        if data is None and vis is not None:
            ntimes, nants, _, nchan, npol = vis.shape
            assert vis.shape == weight.shape
            assert len(frequency) == nchan
            assert len(channel_bandwidth) == nchan
            desc = [('index', 'i8'),
                    ('uvw', 'f8', (nants, nants, 3)),
                    ('time', 'f8'),
                    ('integration_time', 'f8'),
                    ('vis', 'c16', (nants, nants, nchan, npol)),
                    ('flags', 'i8', (nants, nants, nchan, npol)),
                    ('weight', 'f8', (nants, nants, nchan, npol)),
                    ('imaging_weight', 'f8', (nants, nants, nchan, npol))]
            data = numpy.zeros(shape=[ntimes], dtype=desc)
            data['index'] = list(range(ntimes))
            data['uvw'] = uvw
            data['time'] = time  # MJD in seconds
            data['integration_time'] = integration_time  # seconds
            data['vis'] = vis
            data['flags'] = flags
            data['weight'] = weight
            data['imaging_weight'] = imaging_weight
        
        self.data = data  # numpy structured array
        self.frequency = frequency
        self.channel_bandwidth = channel_bandwidth
        self.phasecentre = phasecentre  # Phase centre of observation
        self.configuration = configuration  # Antenna/station configuration
        self.polarisation_frame = polarisation_frame
        self.source = source
        self.meta = meta
    
    def __str__(self):
        """Default printer for BlockVisibility

        """
        s = "BlockVisibility:\n"
        s += "\tSource %s\n" % self.source
        s += "\tPhasecentre: %s\n" % self.phasecentre
        s += "\tNumber of visibilities: %s\n" % self.nvis
        s += "\tNumber of integrations: %s\n" % len(self.time)
        s += "\tVisibility shape: %s\n" % str(self.vis.shape)
        s += "\tNumber of flags: %s\n" % str(numpy.sum(self.flags))
        s += "\tNumber of channels: %d\n" % len(self.frequency)
        s += "\tFrequency: %s\n" % self.frequency
        s += "\tChannel bandwidth: %s\n" % self.channel_bandwidth
        s += "\tNumber of polarisations: %s\n" % self.npol
        s += "\tPolarisation Frame: %s\n" % self.polarisation_frame.type
        s += "\tConfiguration: %s\n" % self.configuration.name
        s += "\tMetadata: %s\n" % self.meta
        
        return s
    
    def size(self):
        """ Return size in GB
        """
        size = 0
        for col in self.data.dtype.fields.keys():
            size += self.data[col].nbytes
        return size / 1024.0 / 1024.0 / 1024.0
    
    @property
    def nchan(self):
        """ Number of channels
        """
        return self.data['vis'].shape[3]
    
    @property
    def npol(self):
        """ Number of polarisations
        """
        return self.data['vis'].shape[4]
    
    @property
    def nants(self):
        """ Number of antennas
        """
        return self.data['vis'].shape[1]
    
    @property
    def uvw(self):
        """ UVW coordinates (metres) [nrows, nant, nant, 3]
        """
        return self.data['uvw']
    
    @property
    def u(self):
        """ u coordinate (metres) [nrows, nant, nant]
        """
        return self.data['uvw'][..., 0]
    
    @property
    def v(self):
        """ v coordinate (metres) [nrows, nant, nant]
        """
        return self.data['uvw'][..., 1]
    
    @property
    def w(self):
        """ w coordinate (metres) [nrows, nant, nant]
        """
        return self.data['uvw'][..., 2]
    
    @property
    def uvdist(self):
        """ uv distance (metres) [nrows, nant, nant]
        """
        return numpy.hypot(self.u, self.v)
    
    @property
    def uvwdist(self):
        """ uv distance (metres) [nrows, nant, nant]
        """
        return numpy.hypot(self.u, self.v, self.w)

    @property
    def vis(self):
        """ Complex visibility [nrows, nant, nant, ncha, npol]
        """
        return self.data['vis']

    @property
    def flagged_vis(self):
        """Flagged complex visibility [nrows, nant, nant, ncha, npol]
        """
        return self.data['vis'] * (1 - self.flags)

    @property
    def flags(self):
        """ Flags [nrows, nant, nant, nchan]
        """
        return self.data['flags']

    @property
    def weight(self):
        """ Weight[nrows, nant, nant, ncha, npol]
        """
        return self.data['weight']

    @property
    def flagged_weight(self):
        """Weight [: npol]
        """
        return self.data['weight'] * (1 - self.data['flags'])

    @property
    def imaging_weight(self):
        """ Imaging_weight[nrows, nant, nant, ncha, npol]
        """
        return self.data['imaging_weight']

    @property
    def flagged_imaging_weight(self):
        """ Flagged Imaging_weight[nrows, nant, nant, ncha, npol]
        """
        return self.data['imaging_weight'] * (1 - self.data['flags'])

    @property
    def time(self):
        """ Time (UTC) [nrows]
        """
        return self.data['time']
    
    @property
    def integration_time(self):
        """ Integration time [nrows]
        """
        return self.data['integration_time']
    
    @property
    def nvis(self):
        """ Number of visibilities (in total)
        """
        return self.data.size


class FlagTable:
    """ Flag table class

    Flags, time, integration_time, frequency, channel_bandwidth, pol,
    in a numpy structured array.

    The configuration is also an attribute
    """

    def __init__(self, data=None, flags=None, frequency=None, channel_bandwidth=None,
                 configuration=None, time=None, integration_time=None,
                 polarisation_frame=None):
        """FlagTable

        :param data: Structured data (used in copying)
        :param frequency: Frequency [nchan]
        :param channel_bandwidth: Channel bandwidth [nchan]
        :param configuration: Configuration
        :param time: Time (UTC) [ntimes]
        :param flags: Flags [ntimes, nant, nant, nchan]
        :param integration_time: Integration time [ntimes]
        """
        if data is None and flags is not None:
            ntimes, nants, _, nchan, npol = flags.shape
            assert len(frequency) == nchan
            assert len(channel_bandwidth) == nchan
            desc = [('time', 'f8'),
                    ('integration_time', 'f8'),
                    ('flags', 'i8', (nants, nants, nchan, npol))]
            data = numpy.zeros(shape=[ntimes], dtype=desc)
            data['time'] = time  # MJD in seconds
            data['integration_time'] = integration_time  # seconds
            data['flags'] = flags

        self.data = data  # numpy structured array
        self.frequency = frequency
        self.channel_bandwidth = channel_bandwidth
        self.polarisation_frame = polarisation_frame
        self.configuration = configuration  # Antenna/station configuration

    def __str__(self):
        """Default printer for FlagTable

        """
        s = "FlagTable:\n"
        s += "\tNumber of integrations: %s\n" % len(self.time)
        s += "\tFlags shape: %s\n" % str(self.flags.shape)
        s += "\tNumber of channels: %d\n" % len(self.frequency)
        s += "\tFrequency: %s\n" % self.frequency
        s += "\tChannel bandwidth: %s\n" % self.channel_bandwidth
        s += "\tNumber of polarisations: %s\n" % self.npol
        s += "\tPolarisation Frame: %s\n" % self.polarisation_frame.type
        s += "\tConfiguration: %s\n" % self.configuration.name

        return s

    def size(self):
        """ Return size in GB
        """
        size = 0
        for col in self.data.dtype.fields.keys():
            size += self.data[col].nbytes
        return size / 1024.0 / 1024.0 / 1024.0

    @property
    def npol(self):
        """ Number of polarisations
        """
        return self.data['flags'].shape[-1]

    @property
    def nchan(self):
        """ Number of channels
        """
        return self.data['flags'].shape[-1]

    @property
    def nants(self):
        """ Number of antennas
        """
        return self.data['vis'].shape[1]

    @property
    def flags(self):
        """ Flags [nrows, nant, nant, nchan]
        """
        return self.data['flags']

    @property
    def time(self):
        """ Time (UTC) [nrows]
        """
        return self.data['time']

    @property
    def integration_time(self):
        """ Integration time [nrows]
        """
        return self.data['integration_time']


class QA:
    """ Quality assessment

    :param origin: str, origin e.g. "continuum_imaging_pipeline"
    :param data: Data containing standard fields
    :param context: Context of QA e.g. "Cycle 5"

    """
    
    def __init__(self, origin=None, data=None, context=None):
        """QA

        :param origin:
        :param data:
        :param context:
        """
        self.origin = origin  # Name of function originating QA assessment
        self.data = data  # Dictionary containing standard fields
        self.context = context  # Context string
    
    def __str__(self):
        """Default printer for QA

        """
        s = "Quality assessment:\n"
        s += "\tOrigin: %s\n" % self.origin
        s += "\tContext: %s\n" % self.context
        s += "\tData:\n"
        for dataname in self.data.keys():
            s += "\t\t%s: %r\n" % (dataname, str(self.data[dataname]))
        return s


class ScienceDataModel:
    """ Science Data Model: not defined yet"""
    
    def __init__(self):
        pass
    
    def __str__(self):
        """ Default printer for Science Data Model
        """
        return ""


def assert_same_chan_pol(o1, o2):
    """
    Assert that two entities indexed over channels and polarisations
    have the same number of them.

    :param o1: Object 1 e.g. BlockVisibility
    :param o2: Object 1 e.g. BlockVisibility
    :return: Bool
    """
    assert o1.npol == o2.npol, \
        "%s and %s have different number of polarisations: %d != %d" % \
        (type(o1).__name__, type(o2).__name__, o1.npol, o2.npol)
    if isinstance(o1, BlockVisibility) and isinstance(o2, BlockVisibility):
        assert o1.nchan == o2.nchan, \
            "%s and %s have different number of channels: %d != %d" % \
            (type(o1).__name__, type(o2).__name__, o1.nchan, o2.nchan)


def assert_vis_gt_compatible(vis: Union[Visibility, BlockVisibility], gt: GainTable):
    """ Check if visibility and gaintable are compatible

    :param vis:
    :param gt:
    :return: Bool
    """
    assert vis.nchan == gt.nchan
    if vis.npol == 4:
        assert vis.npol == gt.nrec * gt.nrec
    elif vis.npol == 2:
        assert gt.nrec == 2
