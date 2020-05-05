""" Unit processing_components for polarisation


"""
import numpy
import unittest

from numpy import random
from numpy.testing import assert_array_almost_equal

from rascil.data_models.polarisation import PolarisationFrame, ReceptorFrame, congruent_polarisation, correlate_polarisation, \
    convert_pol_frame, convert_circular_to_stokes, convert_stokes_to_circular, convert_linear_to_stokes, \
    convert_stokes_to_linear


class TestPolarisation(unittest.TestCase):
    def test_polarisation_frame(self):
        for frame in ['circular', 'circularnp', 'linear', 'linearnp', 'stokesIQUV', 'stokesIV', 'stokesIQ', 'stokesI']:
            polarisation_frame = PolarisationFrame(frame)
            assert polarisation_frame.type == frame
            
        assert PolarisationFrame("circular").npol == 4
        assert PolarisationFrame("circularnp").npol == 2
        assert PolarisationFrame("linear").npol == 4
        assert PolarisationFrame("linearnp").npol == 2
        assert PolarisationFrame("circular").npol == 4
        assert PolarisationFrame("stokesI").npol == 1

        with self.assertRaises(ValueError):
            polarisation_frame = PolarisationFrame("circuloid")
            
        assert PolarisationFrame("linear") != PolarisationFrame("stokesI")
        assert PolarisationFrame("linear") != PolarisationFrame("circular")
        assert PolarisationFrame("circular") != PolarisationFrame("stokesI")

    def test_rec_frame(self):
        rec_frame = ReceptorFrame("linear")
        assert rec_frame.nrec == 2

        rec_frame = ReceptorFrame("circular")
        assert rec_frame.nrec == 2

        rec_frame = ReceptorFrame("stokesI")
        assert rec_frame.nrec == 1

        with self.assertRaises(ValueError):
            rec_frame = ReceptorFrame("circuloid")

    def test_correlate(self):
        for frame in ["linear", "circular", "stokesI"]:
            rec_frame = ReceptorFrame(frame)
            assert correlate_polarisation(rec_frame) == PolarisationFrame(frame)

    def test_congruent(self):
        for frame in ["linear", "circular", "stokesI"]:
            assert congruent_polarisation(ReceptorFrame(frame), PolarisationFrame(frame))
            assert not congruent_polarisation(ReceptorFrame(frame), PolarisationFrame("stokesIQUV"))

    def test_stokes_linear_conversion(self):
        stokes = numpy.array([1.0, 0.0, 0.0, 0.0])
        linear = convert_stokes_to_linear(stokes, 0)
        assert_array_almost_equal(linear, numpy.array([1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 1.0 + 0j]))

        stokes = numpy.array([1.0, 0.0])
        linearnp = convert_stokes_to_linear(stokes, 0)
        assert_array_almost_equal(linearnp, numpy.array([1.0 + 0j, 1.0 + 0j]))

        stokes = numpy.array([0.0, 1.0, 0.0, 0.0])
        linear = convert_stokes_to_linear(stokes, 0)
        assert_array_almost_equal(linear, numpy.array([1.0 + 0j, 0j, 0j, -1.0 + 0j]))
    
        stokes = numpy.array([0.0, 0.0, 1.0, 0.0])
        linear = convert_stokes_to_linear(stokes, 0)
        assert_array_almost_equal(linear, numpy.array([0.0 + 0j, 1.0 + 0j, 1.0 + 0j, 0.0 + 0j]))
    
        stokes = numpy.array([0.0, 0.0, 0.0, 1.0])
        linear = convert_stokes_to_linear(stokes, 0)
        assert_array_almost_equal(linear, numpy.array([0.0 + 0j, +1.0j, -1.0j, 0.0 + 0j]))
        
        stokes = numpy.array([1.0, -0.8, 0.2, 0.01])
        linear = convert_stokes_to_linear(stokes, 0)
        assert_array_almost_equal(linear, numpy.array([0.2+0.j, 0.2+0.01j, 0.2-0.01j, 1.8+0.j]))


    def test_stokes_circular_conversion(self):
        stokes = numpy.array([1.0, 0.0, 0.0, 0.0])
        circular = convert_stokes_to_circular(stokes, 0)
        assert_array_almost_equal(circular, numpy.array([1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 1.0 + 0j]))

        stokes = numpy.array([1.0, 0.0])
        circularcp = convert_stokes_to_circular(stokes, 0)
        assert_array_almost_equal(circularcp, numpy.array([1.0 + 0j, 1.0 + 0j]))

        stokes = numpy.array([0.0, 1.0, 0.0, 0.0])
        circular = convert_stokes_to_circular(stokes, 0)
        assert_array_almost_equal(circular, numpy.array([0.0 + 0j, -1j, -1j, 0.0 + 0j]))
    
        stokes = numpy.array([0.0, 0.0, 1.0, 0.0])
        circular = convert_stokes_to_circular(stokes, 0)
        assert_array_almost_equal(circular, numpy.array([0.0 + 0j, 1.0 + 0j, -1.0 + 0j, 0.0 + 0j]))
    
        stokes = numpy.array([0.0, 0.0, 0.0, 1.0])
        circular = convert_stokes_to_circular(stokes, 0)
        assert_array_almost_equal(circular, numpy.array([1.0 + 0j, +0.0j, 0.0j, -1.0 + 0j]))
        
        stokes = numpy.array([1.0, -0.8, 0.2, 0.01])
        linear = convert_stokes_to_circular(stokes, 0)
        assert_array_almost_equal(linear, numpy.array([1.01+0.j, 0.2 + 0.8j, -0.2 + 0.8j, 0.99+0.j]))


    def test_stokes_linear_stokes_conversion(self):
        stokes = numpy.array([1, 0.5, 0.2, -0.1])
        linear = convert_stokes_to_linear(stokes, 0)
        assert_array_almost_equal(convert_linear_to_stokes(linear, 0).real, stokes, 15)

    def test_stokes_linearnp_stokesIQ_conversion(self):
        stokes = numpy.array([1, 0.5])
        linearnp = convert_stokes_to_linear(stokes, 0)
        assert_array_almost_equal(convert_linear_to_stokes(linearnp, 0).real, stokes, 15)

    def test_stokes_circular_stokes_conversion(self):
        stokes = numpy.array([1, 0.5, 0.2, -0.1])
        circular = convert_stokes_to_circular(stokes, 0)
        assert_array_almost_equal(convert_circular_to_stokes(circular, 0).real, stokes, 15)

    def test_stokes_circularnp_stokesIV_conversion(self):
        stokes = numpy.array([1, 0.5])
        circularnp = convert_stokes_to_circular(stokes, 0)
        assert_array_almost_equal(convert_circular_to_stokes(circularnp, 0).real, stokes, 15)

    def test_image_conversion(self):
        stokes = numpy.array(random.uniform(-1.0, 1.0, [3, 4, 128, 128]))
        cir = convert_stokes_to_circular(stokes, 1)
        st = convert_circular_to_stokes(cir)
        assert_array_almost_equal(st.real, stokes, 15)

    def test_image_auto_conversion_circular(self):
        stokes = numpy.array(random.uniform(-1.0, 1.0, [3, 4, 128, 128]))
        ipf = PolarisationFrame('stokesIQUV')
        opf = PolarisationFrame('circular')
        cir = convert_pol_frame(stokes, ipf, opf, polaxis=1)
        st = convert_pol_frame(cir, opf, ipf, polaxis=1)
        assert_array_almost_equal(st.real, stokes, 15)

    def test_image_auto_conversion_circularnp(self):
        stokes = numpy.array(random.uniform(-1.0, 1.0, [3, 2, 128, 128]))
        ipf = PolarisationFrame('stokesIV')
        opf = PolarisationFrame('circularnp')
        cir = convert_pol_frame(stokes, ipf, opf, polaxis=1)
        st = convert_pol_frame(cir, opf, ipf, polaxis=1)
        assert_array_almost_equal(st.real, stokes, 15)

    def test_image_auto_conversion_linear(self):
        stokes = numpy.array(random.uniform(-1.0, 1.0, [3, 4, 128, 128]))
        ipf = PolarisationFrame('stokesIQUV')
        opf = PolarisationFrame('linear')
        cir = convert_pol_frame(stokes, ipf, opf, polaxis=1)
        st = convert_pol_frame(cir, opf, ipf, polaxis=1)
        assert_array_almost_equal(st.real, stokes, 15)

    def test_image_auto_conversion_linearnp(self):
        stokes = numpy.array(random.uniform(-1.0, 1.0, [3, 2, 128, 128]))
        ipf = PolarisationFrame('stokesIQ')
        opf = PolarisationFrame('linearnp')
        cir = convert_pol_frame(stokes, ipf, opf, polaxis=1)
        st = convert_pol_frame(cir, opf, ipf, polaxis=1)
        assert_array_almost_equal(st.real, stokes, 15)

    def test_image_auto_conversion_I(self):
        stokes = numpy.array(random.uniform(-1.0, 1.0, [3, 4, 128, 128]))
        ipf = PolarisationFrame('stokesI')
        opf = PolarisationFrame('stokesI')
        cir = convert_pol_frame(stokes, ipf, opf, polaxis=1)
        st = convert_pol_frame(cir, opf, ipf, polaxis=1)
        assert_array_almost_equal(st.real, stokes, 15)

    def test_vis_conversion(self):
        stokes = numpy.array(random.uniform(-1.0, 1.0, [1000, 3, 4]))
        cir = convert_stokes_to_circular(stokes, polaxis=2)
        st = convert_circular_to_stokes(cir, polaxis=2)
        assert_array_almost_equal(st.real, stokes, 15)

    def test_vis_auto_conversion(self):
        stokes = numpy.array(random.uniform(-1.0, 1.0, [1000, 3, 4]))
        ipf = PolarisationFrame('stokesIQUV')
        opf = PolarisationFrame('circular')
        cir = convert_pol_frame(stokes, ipf, opf, polaxis=2)
        st = convert_pol_frame(cir, opf, ipf, polaxis=2)
        assert_array_almost_equal(st.real, stokes, 15)

    def test_vis_auto_conversion_I(self):
        stokes = numpy.array(random.uniform(-1.0, 1.0, [1000, 3, 1]))
        ipf = PolarisationFrame('stokesI')
        opf = PolarisationFrame('stokesI')
        cir = convert_pol_frame(stokes, ipf, opf, polaxis=2)
        st = convert_pol_frame(cir, opf, ipf, polaxis=2)
        assert_array_almost_equal(st.real, stokes, 15)
        
    def test_circular_to_linear_should_fail(self):
        stokes = numpy.array(random.uniform(-1.0, 1.0, [3, 4, 128, 128]))
        ipf = PolarisationFrame('stokesIQUV')
        opf = PolarisationFrame('circular')
        cir = convert_pol_frame(stokes, ipf, opf)
        wrong_pf = PolarisationFrame('linear')
        with self.assertRaises(ValueError):
            convert_pol_frame(cir, opf, wrong_pf)


if __name__ == '__main__':
    unittest.main()
