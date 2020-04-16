"""Configuration definitions. A Configuration definition is read from a number of different formats.

"""

__all__ = ['create_configuration_from_file', 'create_configuration_from_MIDfile', 'create_configuration_from_SKAfile',
           'create_LOFAR_configuration', 'create_named_configuration', 'limit_rmax']

import numpy
from astropy import units as u
from astropy.coordinates import EarthLocation

from rascil.processing_components.util.coordinate_support import xyz_at_latitude
from rascil.data_models.memory_data_models import Configuration
from rascil.data_models.parameters import rascil_path, rascil_data_path, get_parameter
from rascil.processing_components.util.installation_checks import check_data_directory

import logging

log = logging.getLogger('logger')

def create_configuration_from_file(antfile: str, location: EarthLocation = None,
                                   mount: str = 'azel',
                                   names: str = "%d",
                                   diameter=35.0,
                                   rmax=None, name='') -> Configuration:
    """ Define configuration from a text file

    :param antfile: Antenna file name
    :param location: Earthlocation of array
    :param mount: mount type: 'azel', 'xy', 'equatorial'
    :param names: Antenna names e.g. "VLA%d"
    :param diameter: Effective diameter of station or antenna
    :param rmax: Maximum distance from array centre (m)
    :param name: Name of array
    :return: Configuration
    """
    check_data_directory()

    antxyz = numpy.genfromtxt(antfile, delimiter=",")
    assert antxyz.shape[1] == 3, ("Antenna array has wrong shape %s" % antxyz.shape)
    latitude = location.geodetic[1].to(u.rad).value
    antxyz = xyz_at_latitude(antxyz, latitude)
    antxyz += [location.geocentric[0].to(u.m).value,
               location.geocentric[1].to(u.m).value,
               location.geocentric[2].to(u.m).value]
    
    nants = antxyz.shape[0]
    diameters = diameter * numpy.ones(nants)
    anames = [names % ant for ant in range(nants)]
    mounts = numpy.repeat(mount, nants)
    antxyz, diameters, anames, mounts = limit_rmax(antxyz, diameters, anames, mounts, rmax)
    
    fc = Configuration(location=location, names=anames, mount=mounts, xyz=antxyz,
                       diameter=diameters, name=name)
    return fc


def create_configuration_from_SKAfile(antfile: str,
                                      mount: str = 'azel',
                                      names: str = "%d",
                                      rmax=None, name='', location=None) -> Configuration:
    """ Define configuration from a SKA format file

    :param antfile: Antenna file name
    :param location: Earthlocation of array
    :param mount: mount type: 'azel', 'xy', 'equatorial'
    :param names: Antenna names e.g. "VLA%d"
    :param rmax: Maximum distance from array centre (m)
    :param name: Name of array
    :return: Configuration
    """
    check_data_directory()

    antdiamlonglat = numpy.genfromtxt(antfile, usecols=[0, 1, 2], delimiter="\t")
    
    assert antdiamlonglat.shape[1] == 3, ("Antenna array has wrong shape %s" % antdiamlonglat.shape)
    antxyz = numpy.zeros([antdiamlonglat.shape[0] - 1, 3])
    diameters = numpy.zeros([antdiamlonglat.shape[0] - 1])
    for ant in range(antdiamlonglat.shape[0] - 1):
        loc = EarthLocation(lon=antdiamlonglat[ant, 1], lat=antdiamlonglat[ant, 2], height=0.0).geocentric
        antxyz[ant] = [loc[0].to(u.m).value, loc[1].to(u.m).value, loc[2].to(u.m).value]
        diameters[ant] = antdiamlonglat[ant, 0]

    nants = antxyz.shape[0]
    anames = [names % ant for ant in range(nants)]
    mounts = numpy.repeat(mount, nants)
    antxyz, diameters, anames, mounts = limit_rmax(antxyz, diameters, anames, mounts, rmax)
    
    fc = Configuration(location=location, names=anames, mount=mounts, xyz=antxyz,
                       diameter=diameters, name=name)
    return fc


def create_configuration_from_MIDfile(antfile: str, location=None,
                                      mount: str = 'azel',
                                      rmax=None, name='') -> Configuration:
    """ Define configuration from a SKA MID format file

    :param antfile: Antenna file name
    :param mount: mount type: 'azel', 'xy'
    :param rmax: Maximum distance from array centre (m)
    :param name: Name of array
    :return: Configuration
    """
    check_data_directory()


    # X Y Z Diam Station
    # 9.36976 35.48262 1052.99987 13.50 M001
    antxyz = numpy.genfromtxt(antfile, skip_header=5, usecols=[0, 1, 2], delimiter=" ")
    
    antxyz = xyz_at_latitude(antxyz, location.lat.rad)
    antxyz += [location.geocentric[0].to(u.m).value,
               location.geocentric[1].to(u.m).value,
               location.geocentric[2].to(u.m).value]

    nants = antxyz.shape[0]
    assert antxyz.shape[1] == 3, "Antenna array has wrong shape %s" % antxyz.shape

    anames = numpy.genfromtxt(antfile, dtype='str', skip_header=5, usecols=[4], delimiter=" ")
    mounts = numpy.repeat(mount, nants)
    diameters = numpy.genfromtxt(antfile, dtype='str', skip_header=5, usecols=[3], delimiter=" ")

    antxyz, diameters, anames, mounts = limit_rmax(antxyz, diameters, anames, mounts, rmax)

    fc = Configuration(location=location, names=anames, mount=mounts, xyz=antxyz,
                       diameter=diameters, name=name)

    return fc


def limit_rmax(antxyz, diameters, names, mounts, rmax):
    """ Select antennas with radius from centre < rmax
    
    :param antxyz: Geocentric coordinates
    :param diameters: diameters in metres
    :param names: Names
    :param mounts: Mount types
    :param rmax: Maximum radius (m)
    :return:
    """
    if rmax is not None:
        lantxyz = antxyz - numpy.average(antxyz, axis=0)
        r = numpy.sqrt(lantxyz[:, 0] ** 2 + lantxyz[:, 1] ** 2 + lantxyz[:, 2] ** 2)
        antxyz = antxyz[r < rmax]
        log.debug('create_configuration_from_file: Maximum radius %.1f m includes %d antennas/stations' %
                  (rmax, antxyz.shape[0]))
        diameters = diameters[r < rmax]
        names = numpy.array(names)[r < rmax]
        mounts = numpy.array(mounts)[r<rmax]
    else:
        log.debug('create_configuration_from_file: %d antennas/stations' % (antxyz.shape[0]))
    return antxyz, diameters, names, mounts


def create_LOFAR_configuration(antfile: str, location, rmax=1e6) -> Configuration:
    """ Define configuration from the LOFAR configuration file

    :param antfile:
    :param location: EarthLocation
    :param rmax: Maximum distance from array centre (m)
    :return: Configuration
    """
    check_data_directory()

    antxyz = numpy.genfromtxt(antfile, skip_header=2, usecols=[1, 2, 3], delimiter=",")
    nants = antxyz.shape[0]
    assert antxyz.shape[1] == 3, "Antenna array has wrong shape %s" % antxyz.shape
    anames = numpy.genfromtxt(antfile, dtype='str', skip_header=2, usecols=[0], delimiter=",")
    mounts = numpy.repeat('XY', nants)
    diameters = numpy.repeat(35.0, nants)
    
    antxyz, diameters, mounts, anames = limit_rmax(antxyz, diameters, anames, mounts, rmax)

    fc = Configuration(location=location, names=anames, mount=mounts, xyz=antxyz,
                       diameter=diameters, name='LOFAR')
    return fc


def create_named_configuration(name: str = 'LOWBD2', **kwargs) -> Configuration:
    """ Create standard configurations e.g. LOWBD2, MIDBD2

    Possible configurations are::
        LOWBD1
        LOWBD2
        LOWBD2-core
        LOW == LOWR3
        MID == MIDR5
        ASKAP
        LOFAR
        VLAA
        VLAA_north

    :param name: name of Configuration LOWBD2, LOWBD1, LOFAR, VLAA, ASKAP
    :param rmax: Maximum distance of station from the average (m)
    :return:
    
    For LOWBD2, setting rmax gives the following number of stations
    100.0       13
    300.0       94
    1000.0      251
    3000.0      314
    10000.0     398
    30000.0     476
    100000.0    512
    """
    
    check_data_directory()

    if name == 'LOWBD2':
        location = EarthLocation(lon="116.76444824", lat="-26.824722084", height=300.0)
        log.info("create_named_configuration: %s\n\t%s\n\t%s" % (name, location.geocentric, location.geodetic))
        fc = create_configuration_from_file(antfile=rascil_path("data/configurations/LOWBD2.csv"),
                                            location=location, mount='xy', names='LOWBD2_%d',
                                            diameter=35.0, name=name, **kwargs)
    elif name == 'LOWBD1':
        location = EarthLocation(lon="116.76444824", lat="-26.824722084", height=300.0)
        log.info("create_named_configuration: %s\n\t%s\n\t%s" % (name, location.geocentric, location.geodetic))
        fc = create_configuration_from_file(antfile=rascil_path("data/configurations/LOWBD1.csv"),
                                            location=location, mount='xy', names='LOWBD1_%d',
                                            diameter=35.0, name=name, **kwargs)
    elif name == 'LOWBD2-CORE':
        location = EarthLocation(lon="116.76444824", lat="-26.824722084", height=300.0)
        log.info("create_named_configuration: %s\n\t%s\n\t%s" % (name, location.geocentric, location.geodetic))
        fc = create_configuration_from_file(antfile=rascil_path("data/configurations/LOWBD2-CORE.csv"),
                                            location=location, mount='xy', names='LOWBD2_%d',
                                            diameter=35.0, name=name, **kwargs)
    elif (name == 'LOW') or (name == 'LOWR3'):
        location = EarthLocation(lon="116.76444824", lat="-26.824722084", height=300.0)
        log.info("create_named_configuration: %s\n\t%s\n\t%s" % (name, location.geocentric, location.geodetic))
        fc = create_configuration_from_MIDfile(antfile=rascil_path("data/configurations/ska1low_local.cfg"),
                                          mount='xy', name=name, location=location, **kwargs)
    elif (name == 'MID') or (name == "MIDR5"):
        location = EarthLocation(lon="21.443803", lat="-30.712925", height=0.0)
        log.info("create_named_configuration: %s\n\t%s\n\t%s" % (name, location.geocentric, location.geodetic))
        fc = create_configuration_from_MIDfile(antfile=rascil_path("data/configurations/ska1mid_local.cfg"),
            mount='azel', name=name, location=location, **kwargs)
    elif name == 'ASKAP':
        location = EarthLocation(lon="+116.6356824", lat="-26.7013006", height=377.0)
        log.info("create_named_configuration: %s\n\t%s\n\t%s" % (name, location.geocentric, location.geodetic))
        fc = create_configuration_from_file(antfile=rascil_path("data/configurations/A27CR3P6B.in.csv"),
                                            mount='equatorial', names='ASKAP_%d',
                                            diameter=12.0, name=name, location=location, **kwargs)
    elif name == 'LOFAR':
        location = EarthLocation(x=3826923.9 * u.m, y=460915.1 * u.m, z=5064643.2 * u.m)
        log.info("create_named_configuration: %s\n\t%s\n\t%s" % (name, location.geocentric, location.geodetic))
        assert get_parameter(kwargs, "meta", False) is False
        fc = create_LOFAR_configuration(antfile=rascil_path("data/configurations/LOFAR.csv"), location=location)
    elif name == 'VLAA':
        location = EarthLocation(lon="-107.6184", lat="34.0784", height=2124.0)
        log.info("create_named_configuration: %s\n\t%s\n\t%s" % (name, location.geocentric, location.geodetic))
        fc = create_configuration_from_file(antfile=rascil_path("data/configurations/VLA_A_hor_xyz.csv"),
                                            location=location,
                                            mount='azel',
                                            names='VLA_%d',
                                            diameter=25.0, name=name, **kwargs)
    elif name == 'VLAA_north':
        location = EarthLocation(lon="-107.6184", lat="90.000", height=0.0)
        log.info("create_named_configuration: %s\n\t%s\n\t%s" % (name, location.geocentric, location.geodetic))
        fc = create_configuration_from_file(antfile=rascil_path("data/configurations/VLA_A_hor_xyz.csv"),
                                            location=location,
                                            mount='azel',
                                            names='VLA_%d',
                                            diameter=25.0, name=name, **kwargs)
    else:
        raise ValueError("No such Configuration %s" % name)
    return fc
