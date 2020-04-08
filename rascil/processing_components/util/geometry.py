""" Unit tests for geometry

"""

__all__ = ['calculate_transit_time', 'calculate_hourangles', 'calculate_azel']

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation


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
    
    from astroplan import Observer
    site = Observer(location=location)
    return site.target_hour_angle(utc_time, direction).wrap_at('180d')


def calculate_transit_time(location, utc_time, direction):
    """ Find the UTC time of the nearest transit

    :param utc_time:
    :param location:
    :param direction: Direction of source
    :return:
    """
    assert isinstance(location, EarthLocation)
    assert isinstance(utc_time, Time)
    assert isinstance(direction, SkyCoord)

    from astroplan import Observer
    site = Observer(location)
    return site.target_meridian_transit_time(utc_time, direction, which="next", n_grid_points=100)


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

    from astroplan import Observer
    site = Observer(location=location)
    altaz = site.altaz(utc_time, direction)
    return altaz.az.wrap_at('180d'), altaz.alt
