""" Unit tests for geometry

"""

__all__ = ['calculate_blockvisibility_transit_time', 'calculate_blockvisibility_hourangles',
           'calculate_blockvisibility_azel']

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation

from rascil.data_models import BlockVisibility

def get_direction_time_location(bvis):
    location = bvis.configuration.location
    utc_time = Time(bvis.time / 86400.0, format='mjd', scale='utc')
    direction = bvis.phasecentre
    assert isinstance(bvis, BlockVisibility)
    assert isinstance(location, EarthLocation)
    assert isinstance(utc_time, Time)
    assert isinstance(direction, SkyCoord)
    return location, utc_time, direction


def calculate_blockvisibility_hourangles(bvis):
    """ Return hour angles for location, utc_time, and direction

    :param bvis:
    :return:
    """

    location, utc_time, direction = get_direction_time_location(bvis)

    from astroplan import Observer
    site = Observer(location=location)
    return site.target_hour_angle(utc_time, direction).wrap_at('180d')



def calculate_blockvisibility_transit_time(bvis):
    """ Find the UTC time of the nearest transit

    :param utc_time:
    :param location:
    :param direction: Direction of source
    :return:
    """
    location, utc_time, direction = get_direction_time_location(bvis)

    from astroplan import Observer
    site = Observer(location)
    return site.target_meridian_transit_time(utc_time, direction, which="next", n_grid_points=100)


def calculate_blockvisibility_azel(bvis):
    """ Return az el for a location, utc_time, and direction

    :param utc_time:
    :param location:
    :param direction: Direction of source
    :return:
    """
    location, utc_time, direction = get_direction_time_location(bvis)

    from astroplan import Observer
    site = Observer(location=location)
    altaz = site.altaz(utc_time, direction)
    return altaz.az.wrap_at('180d'), altaz.alt
