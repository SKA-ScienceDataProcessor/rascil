""" Unit tests for geometry

This now uses the CASA measures classes via python-casacore. The interface is not as straightforward
as astropy.

    https://casa.nrao.edu/docs/CasaRef/measures.measure.html


"""

__all__ = ['calculate_transit_time', 'calculate_hourangles', 'calculate_azel']

import logging

import numpy

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, Angle

log = logging.getLogger('logger')


def calculate_hourangles(location, utc_time, direction):
    """ Return hour angles for location, utc_time, and direction

    :param utc_time:
    :param location:
    :param direction: Direction of source
    :return:
    """
    
    assert isinstance(location, EarthLocation)
    assert isinstance(utc_time, Time)
    assert isinstance(direction, SkyCoord)

    from casacore.measures import measures
    dm = measures()
    casa_location = dm.position('itrf', str(location.x), str(location.y), str(location.z))
    dm.doframe(casa_location)
    casa_direction = dm.direction('j2000', str(direction.ra), str(direction.dec))
    has = list()
    for utc in utc_time:
        casa_utc_time = dm.epoch('utc', str(utc.mjd) + 'd')
        dm.doframe(casa_utc_time)
        casa_hadec = dm.measure(casa_direction, 'hadec')
        ha = str(casa_hadec['m0']['value']) + casa_hadec['m0']['unit']
        has.append(Angle(ha))
    return has

    # from astroplan import Observer
    # site = Observer(location=location)
    # return site.target_hour_angle(utc_time, direction).wrap_at('180d')


def calculate_transit_time(location, utc_time, direction, fraction_day=0.01):
    """ Find the UTC time of the nearest transit

    :param fraction_day:
    :param utc_time:
    :param location:
    :param direction: Direction of source
    :return:
    """
    assert isinstance(location, EarthLocation)
    assert isinstance(utc_time, Time)
    assert isinstance(direction, SkyCoord)

    utc_times = Time(numpy.arange(0.0, 1.0, fraction_day) + utc_time.mjd, format='mjd', scale='utc')
    _, els = calculate_azel(location, utc_times, direction)
    elmax = - numpy.pi / 2.0
    best = None
    for i, el in enumerate(els):
        if el.value > elmax:
            elmax = el.value
            best = i
    if elmax < 0.0:
        log.warning("Source is always below horizon")
    return utc_times[best]
    
    # from astroplan import Observer
    # site = Observer(location)
    # return site.target_meridian_transit_time(utc_time, direction, which="next", n_grid_points=100)


def calculate_azel(location, utc_time, direction):
    """ Return az el for a location, utc_time, and direction

    :param utc_time:
    :param location:
    :param direction: Direction of source
    :return:
    """
    assert isinstance(location, EarthLocation)
    assert isinstance(utc_time, Time)
    assert isinstance(direction, SkyCoord)

    # Use the casa measures
    from casacore.measures import measures
    dm = measures()
    casa_location = dm.position('itrf', str(location.x), str(location.y), str(location.z))
    dm.doframe(casa_location)
    casa_direction = dm.direction('j2000', str(direction.ra), str(direction.dec))
    azs = list()
    els = list()
    for utc in utc_time:
        casa_utc_time = dm.epoch('utc', str(utc.mjd) + 'd')
        dm.doframe(casa_utc_time)
        casa_azel = dm.measure(casa_direction, 'azel')
        az = str(casa_azel['m0']['value']) + casa_azel['m0']['unit']
        el = str(casa_azel['m1']['value']) + casa_azel['m1']['unit']
        azs.append(Angle(az))
        els.append(Angle(el))
    return azs, els

    # from astroplan import Observer
    # site = Observer(location=location)
    # altaz = site.altaz(utc_time, direction)
    # return altaz.az.wrap_at('180d'), altaz.alt
