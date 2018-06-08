""" Helper functions for converting JSON to ARL objects

"""

from astropy.coordinates import SkyCoord
from astropy.units import Quantity, Unit
import numpy

def json_to_skycoord(d):
    """Convert JSON string to SkyCoord
    
    "phasecentre": {
      "ra": {
        "value": 30.0,
        "unit": "deg"
      },
      "dec": {
        "value": -60.0,
        "unit": "deg"
      },
      "frame": "icrs",
      "equinox": "j2000"
    }

    :param d:
    :return:
    """
    return SkyCoord(ra=json_to_quantity(d["ra"]),
                    dec=json_to_quantity(d["dec"]),
                    equinox=d["equinox"],
                    frame=d["frame"])


def json_to_quantity(q):
    """Convert JSON string to Quantity
    
    e.g.
    "cellsize": {
      "value": 0.001,
      "unit": "rad"
    },


    :param q:
    :return:
    """
    return Quantity(q["value"], Unit(q["unit"]))

def json_to_linspace(l):
    """Convert JSON string to numpy.linspace
    
    e.g.
    "frequency": {
        "start": 0.9e8,
        "stop": 1.1e8,
        "steps": 7
    },

    
    :param l:
    :return:
    """
    return numpy.linspace(l["start"], l["stop"], l["steps"])