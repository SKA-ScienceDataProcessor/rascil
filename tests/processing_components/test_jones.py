""" Unit tests for Jones matrix application

"""

import logging
import unittest

import numpy
from numpy.testing import assert_array_almost_equal

from rascil.data_models.polarisation import convert_pol_frame, PolarisationFrame
from rascil.processing_components.calibration import apply_jones

log = logging.getLogger('logger')

log.setLevel(logging.WARNING)


class TestJones(unittest.TestCase):
    def setUp(self):
        pass

    def test_apply_jones(self):
        nsucceeded = 0
        nfailures = 0
        for flux in (numpy.array([100.0, 0.0, 0.0, 0.0]),
                     numpy.array([100.0, 100.0, 0.0, 0.0]),
                     numpy.array([100.0, 0.0, 100.0, 0.0]),
                     numpy.array([100.0, 0.0, 0.0, 100.0]),
                     numpy.array([100.0, 1.0, -10.0, +60.0])):
            vpol = PolarisationFrame("linear")
            cpol = PolarisationFrame("stokesIQUV")
            cflux = convert_pol_frame(flux, cpol, vpol).reshape([2,2])

            diagonal = numpy.array([[1.0 + 0.0j, 0.0 + 0.0j],
                                    [0.0 + 0.0j, 1.0 + 0.0]])
            skew = numpy.array([[0.0 + 0.0j, 1.0 + 0.0j],
                                [1.0 + 0.0j, 0.0 + 0.0]])
            leakage = numpy.array([[1.0 + 0.0j, 0.0 + 0.1j],
                                   [0.0 - 0.1j, 1.0 + 0.0]])
            unbalanced = numpy.array([[100.0 + 0.0j, 0.0 + 0.0j],
                                      [0.0 + 0.0j, 0.03 + 0.0]])

            for ej in (diagonal, skew, leakage, unbalanced):
                try:
                    jflux = apply_jones(ej, cflux, inverse=False)
                    rflux = apply_jones(ej, jflux, inverse=True).reshape([4])
                    rflux = convert_pol_frame(rflux, vpol, cpol)
                    assert_array_almost_equal(flux, numpy.real(rflux), 12)
                    # print("{0} {1} {2} succeeded".format(vpol, str(ej), str(flux)))
                    nsucceeded += 1
                except AssertionError as e:
                    print(e)
                    print("{0} {1} {2} failed".format(vpol, str(ej), str(flux)))
                    nfailures += 1
        assert nfailures == 0, "{0} tests succeeded, {1} failed".format(nsucceeded, nfailures)


if __name__ == '__main__':
    unittest.main()
