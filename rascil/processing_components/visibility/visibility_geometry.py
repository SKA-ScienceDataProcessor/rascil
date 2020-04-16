""" Unit tests for geometry

"""

__all__ = ['calculate_blockvisibility_transit_time',
           'calculate_blockvisibility_hourangles',
           'calculate_blockvisibility_azel']

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation

from rascil.data_models import BlockVisibility

from rascil.processing_components.util.geometry import calculate_azel, calculate_transit_time, \
    calculate_hourangles

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
    return calculate_hourangles(location, utc_time, direction)


def calculate_blockvisibility_transit_time(bvis, fraction_day=0.01):
    """ Find the UTC time of the nearest transit

    :param fraction_day:
    :param utc_time:
    :param location:
    :param direction: Direction of source
    :return:
    """
    location, utc_time, direction = get_direction_time_location(bvis)
    return calculate_transit_time(location, utc_time[0], direction, fraction_day=fraction_day)

def calculate_blockvisibility_azel(bvis):
    """ Return az el for a location, utc_time, and direction

    :param utc_time:
    :param location:
    :param direction: Direction of source
    :return:
    """
    location, utc_time, direction = get_direction_time_location(bvis)
    return calculate_azel(location, utc_time, direction)

