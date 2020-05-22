""" Unit tests for geometry

This now uses the CASA measures classes via python-casacore. The interface is not as straightforward
as astropy.

    https://casa.nrao.edu/docs/CasaRef/measures.measure.html


"""

__all__ = ['calculate_transit_time', 'calculate_hourangles', 'calculate_parallactic_angles',
           'calculate_azel']

import logging

import numpy

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, Angle

log = logging.getLogger('logger')


def angle_to_quanta(angle):
    return {"value": angle.rad, "unit": "rad"}


def calculate_parallactic_angles(location, utc_time, direction):
    """ Return hour angles for location, utc_time, and direction

    :param utc_time: Time(Iterable)
    :param location: EarthLocation
    :param direction: SkyCoord source
    :return: Angle
    """
    
    assert isinstance(location, EarthLocation)
    assert isinstance(utc_time, Time)
    assert isinstance(direction, SkyCoord)
    
    from casacore.measures import measures
    dm = measures()
    from casacore import quanta

    casa_location = dm.position('itrf', str(location.x), str(location.y), str(location.z))
    dm.doframe(casa_location)
    casa_direction = dm.direction('j2000', angle_to_quanta(direction.ra), angle_to_quanta(direction.dec))
    pas = list()
    unit = "rad"
    zenith = dm.direction('AZEL','0deg','90deg')

    for utc in utc_time:
        casa_utc_time = dm.epoch('utc', str(utc.mjd) + 'd')
        dm.doframe(casa_utc_time)
        casa_posangle = dm.posangle(casa_direction, zenith).canonical()
        pas.append(casa_posangle._get_value()[0])
        assert unit == casa_posangle.get_unit(), casa_posangle.get_unit()

    return Angle(pas, unit=unit)
    
    # from astroplan import Observer
    # site = Observer(location=location)
    # return site.target_hour_angle(utc_time, direction).wrap_at('180d')


def calculate_hourangles(location, utc_time, direction):
    """ Return hour angles for location, utc_time, and direction

    :param utc_time: Time(Iterable)
    :param location: EarthLocation
    :param direction: SkyCoord source
    :return: Angle
    """
    
    assert isinstance(location, EarthLocation)
    assert isinstance(utc_time, Time)
    assert isinstance(direction, SkyCoord)
    
    from casacore.measures import measures
    dm = measures()
    casa_location = dm.position('itrf', str(location.x), str(location.y), str(location.z))
    dm.doframe(casa_location)
    casa_direction = dm.direction('j2000', angle_to_quanta(direction.ra), angle_to_quanta(direction.dec))
    has = list()
    unit = "rad"
    for utc in utc_time:
        casa_utc_time = dm.epoch('utc', str(utc.mjd) + 'd')
        dm.doframe(casa_utc_time)
        casa_hadec = dm.measure(casa_direction, 'hadec')
        has.append(casa_hadec['m0']['value'])
        assert unit == casa_hadec['m0']['unit']
    return Angle(has, unit=unit)
    
    # from astroplan import Observer
    # site = Observer(location=location)
    # return site.target_hour_angle(utc_time, direction).wrap_at('180d')


def calculate_transit_time(location, utc_time, direction, fraction_day=0.01):
    """ Find the UTC time of the nearest transit

    :param fraction_day: Step in this fraction of day to find transit
    :param utc_time: Time(Iterable)
    :param location: EarthLocation
    :param direction: SkyCoord source
    :return: astropy Time
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

    :param utc_time: Time(Iterable)
    :param location: EarthLocation
    :param direction: SkyCoord source
    :return: astropy Angle, Angle
    """
    assert isinstance(location, EarthLocation)
    assert isinstance(utc_time, Time)
    assert isinstance(direction, SkyCoord)

    # Use the casa measures
    from casacore.measures import measures
    dm = measures()
    casa_location = dm.position('itrf', str(location.x), str(location.y), str(location.z))
    dm.doframe(casa_location)
    casa_direction = dm.direction('j2000', angle_to_quanta(direction.ra), angle_to_quanta(direction.dec))
    azs = list()
    els = list()
    unit0 = 'rad'
    unit1 = 'rad'
    for utc in utc_time:
        casa_utc_time = dm.epoch('utc', str(utc.mjd) + 'd')
        dm.doframe(casa_utc_time)
        casa_azel = dm.measure(casa_direction, 'azel')
        assert unit0 == casa_azel['m0']['unit']
        assert unit1 == casa_azel['m1']['unit']
        azs.append(casa_azel['m0']['value'])
        els.append(casa_azel['m1']['value'])
    return Angle(azs, unit=unit0), Angle(els, unit=unit1)

    # from astroplan import Observer
    # site = Observer(location=location)
    # altaz = site.altaz(utc_time, direction)
    # return altaz.az.wrap_at('180d'), altaz.alt
