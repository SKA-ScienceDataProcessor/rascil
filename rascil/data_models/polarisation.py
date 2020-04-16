"""
Functions for defining polarisation conventions. These include definitions via classes and
conversion functions.

For example::

    stokes = numpy.array(random.uniform(-1.0, 1.0, [3, 4, 128, 128]))
    ipf = PolarisationFrame('stokesIQUV')
    opf = PolarisationFrame('circular')
    cir = convert_pol_frame(stokes, ipf, opf)
    st = convert_pol_frame(cir, opf, ipf)

or::

    stokes = numpy.array([1, 0.5, 0.2, -0.1])
    circular = convert_stokes_to_circular(stokes)
    
These function operate on Numpy arrays. These are packaged for use in Images. The Image functions are
probably more useful.

"""

import logging

import numpy

log = logging.getLogger('logger')


class ReceptorFrame:
    """Define polarisation frames for receptors

    circular, linear, and stokesI. The latter is non-physical but useful for some types of testing.
    """
    
    rec_frames = {
        'circular': {'R': 0, 'L': 1},
        'circularnp': {'R': 0, 'L': 1},
        'linear': {'X': 0, 'Y': 1},
        'linearnp': {'X': 0, 'Y': 1},
        'stokesI': {'I': 0}
    }
    
    def __init__(self, name):
        """ create ReceptorFrame

        :param name:
        """
        
        if name in self.rec_frames.keys():
            self.type = name
            self.translations = self.rec_frames[name]
        else:
            raise ValueError("Unknown receptor frame %s" % str(name))
    
    @property
    def nrec(self):
        """ Number of receptors (should be 2)
        """
        return len(list(self.translations.keys()))
    
    def valid(self, name):
        return name in self.rec_frames.keys()
    
    @property
    def names(self):
        """ Names
        
        :return:
        """
        return list(self.translations.keys())
    
    def __eq__(self, a):
        return self.type == a.type


class PolarisationFrame:
    """ Define polarisation frames post correlation

    stokesI, stokesIQUV, linear, circular

    """
    fits_codes = {
        'circular': [-1, -2, -3, -4],
        'circularnp': [-1, -4],
        'linear': [-5, -6, -7, -8],
        'linearnp': [-5, -8],
        'stokesIQUV': [1, 2, 3, 4],
        'stokesIV': [1, 4],
        'stokesIQ': [1, 2],
        'stokesI': [1]
    }
    polarisation_frames = {
        'circular': {'RR': 0, 'RL': 1, 'LR': 2, 'LL': 3},
        'circularnp': {'RR': 0, 'LL': 1},
        'linear': {'XX': 0, 'XY': 1, 'YX': 2, 'YY': 3},
        'linearnp': {'XX': 0, 'YY': 1},
        'stokesIQUV': {'I': 0, 'Q': 1, 'U': 2, 'V': 3},
        'stokesIV': {'I': 0, 'V': 1},
        'stokesIQ': {'I': 0, 'Q': 1},
        'stokesI': {'I': 0}
    }
    
    def __init__(self, name):
        """ create PolarisationFrame

        :param name:
        """
        
        if name in self.polarisation_frames.keys():
            self.type = name
            self.translations = self.polarisation_frames[name]
        else:
            raise ValueError("Unknown polarisation frame %s" % str(name))
    
    def __eq__(self, a):
        if a is None:
            return False
        return self.type == a.type
    
    def __str__(self):
        """Default printer for Polarisation"""
        return self.type
    
    @property
    def npol(self):
        """ Number of correlated polarisations
        """
        return len(list(self.translations.keys()))
    
    @property
    def names(self):
        """ Names

        :return:
        """
        return list(self.translations.keys())


def polmatrixmultiply(cm, vec, polaxis=0):
    """Matrix multiply of appropriate axis of vec [...,:] by cm

    For an image vec has axes [nchan, npol, ny, nx] and polaxis=1
    For visibility vec has axes [row, nchan, npol] and polaxis=2
    For blockvisibility vec has axes [row, ant, ant, nchan, npol] and polaxis=4

    :param cm: matrix to apply
    :param vec: array to be multiplied [...,:]
    :param polaxis: which axis contains the polarisation
    :return: multiplied vec
    """
    if vec.shape[polaxis] == 1:
        return numpy.dot(cm, vec)
    elif vec.shape[polaxis] == 2:
        assert cm.shape == (2, 2)
    
    elif vec.shape[polaxis] == 4:
        assert cm.shape == (4, 4)
    
    else:
        raise ValueError("Unknown polarisation conversion {} {}".format(str(cm), str(vec)))
    
    # This tensor swaps the first two axes so we need to tranpose back
    # e.g. if polaxis=2 1000, 3, 4 becomes 4, 1000, 3
    if polaxis == -1:
        polaxis = len(vec.shape) - 1
    result = numpy.tensordot(cm, vec, axes=(1, polaxis))
    permut = list(range(len(vec.shape)))
    assert 5 > polaxis >= 0, "Error in polarisation conversion logic"
    if polaxis == 1:
        permut[0], permut[1] = permut[1], permut[0]
    elif polaxis == 2:
        permut[0], permut[1], permut[2] = permut[1], permut[2], permut[0]
    elif polaxis == 3:
        permut[0], permut[1], permut[2], permut[3] = permut[1], permut[2], permut[3], permut[0]
    elif polaxis == 4:
        permut[0], permut[1], permut[2], permut[3], permut[4] = permut[1], permut[2], permut[3], permut[4], \
                                                                permut[0]
    transposed = numpy.transpose(result, axes=permut)
    assert transposed.shape == vec.shape
    return transposed


def convert_stokes_to_linear(stokes, polaxis=1):
    """ Convert Stokes IQUV to Linear (complex image)

    :param stokes: [...,4] Stokes vector in I,Q,U,V (can be complex)
    :param polaxis: Axis of stokes with polarisation (default 1)
    :return: linear vector in XX, XY, YX, YY sequence

    Equation 4.58 TMS
    """
    if stokes.shape[polaxis] == 2:
        conversion_matrix = numpy.array([[1, 1],
                                         [1, -1]])
    
    else:
        conversion_matrix = numpy.array([[1, 1, 0, 0],
                                         [0, 0, 1, 1j],
                                         [0, 0, 1, -1j],
                                         [1, -1, 0, 0]])
    
    return polmatrixmultiply(conversion_matrix, stokes, polaxis)


def convert_linear_to_stokes(linear, polaxis=1):
    """ Convert Linear to Stokes IQUV (complex image)

    :param linear: [...,4] linear vector in XX, XY, YX, YY sequence
    :param polaxis: Axis of linear with polarisation (default 1)
    :return: Complex I,Q,U,V

    Equation 4.58 TMS, inverted with numpy.linalg.inv
    """
    
    if linear.shape[polaxis] == 2:
        conversion_matrix = numpy.array([[0.5 + 0.j, 0.5 + 0.j],
                                         [0.5 + 0.j, -0.5 - 0.j]])
    else:
        conversion_matrix = numpy.array([[0.5 + 0.j, 0.0 + 0.j, 0.0 + 0.j, 0.5 + 0.j],
                                         [0.5 + 0.j, 0.0 + 0.j, 0.0 + 0.j, -0.5 - 0.j],
                                         [0.0 + 0.j, 0.5 + 0.j, 0.5 + 0.j, 0.0 + 0.j],
                                         [0.0 + 0.j, 0.0 - 0.5j, 0.0 + 0.5j, 0.0 + 0.j]])
    
    return polmatrixmultiply(conversion_matrix, linear, polaxis)


def convert_linear_to_stokesI(linear):
    """ Convert Linear to Stokes I

    :param linear: [...,4] linear vector in XX, XY, YX, YY sequence
    :return: Complex I

    Equation 4.58 TMS, inverted with numpy.linalg.inv
    """
    if linear.shape[-1] == 2:
        return 0.5 * (linear[..., 0] + linear[..., 1])[..., numpy.newaxis]
    else:
        return 0.5 * (linear[..., 0] + linear[..., 3])[..., numpy.newaxis]


def convert_stokes_to_circular(stokes, polaxis=1):
    """ Convert Stokes IQUV to Circular (complex image)

    :param stokes: [...,4] Stokes vector in I,Q,U,V (can be complex)
    :param polaxis: Axis of stokes with polarisation (default 1)
    :return: circular vector in RR, RL, LR, LL sequence

    Equation 4.59 TMS
    """
    if stokes.shape[polaxis] == 2:
        conversion_matrix = numpy.array([[1, 1],
                                         [1, -1]])

    else:
        conversion_matrix = numpy.array([[1, 0, 0, 1],
                                         [0, -1j, 1, 0],
                                         [0, -1j, -1, 0],
                                         [1, 0, 0, -1]])
    return polmatrixmultiply(conversion_matrix, stokes, polaxis)


def convert_circular_to_stokes(circular, polaxis=1):
    """ Convert Circular to Stokes IQUV (complex image)

    :param circular: [...,4] linear vector in RR, RL, LR, LL sequence
    :param polaxis: Axis of circular with polarisation (default 1)
    :return: Complex I,Q,U,V

    Equation 4.58 TMS, inverted with numpy.linalg.inv
    """
    
    if circular.shape[polaxis] == 2:
        conversion_matrix = numpy.array([[0.5 + 0.j, 0.5 + 0.j],
                                         [0.5 + 0.j, -0.5 - 0.j]])
    else:
        conversion_matrix = numpy.array([[0.5 + 0.j, 0.0 + 0.j, 0.0 + 0.j, 0.5 + 0.j],
                                         [0.0 + 0.j, -0.0 + 0.5j, -0.0 + 0.5j, 0.0 + 0.j],
                                         [0.0 + 0.j, 0.5 + 0.j, -0.5 - 0.j, 0.0 + 0.j],
                                         [0.5 + 0.j, 0.0 + 0.j, 0.0 + 0.j, -0.5 - 0.j]])
    
    return polmatrixmultiply(conversion_matrix, circular, polaxis)


def convert_circular_to_stokesI(circular):
    """ Convert Circular to Stokes I

    :param circular: [...,4] linear vector in RR, RL, LR, LL sequence
    :return: Complex I

    Equation 4.58 TMS, inverted with numpy.linalg.inv
    """
    
    if circular.shape[-1] == 2:
        return 0.5 * (circular[..., 0] + circular[..., 1])[..., numpy.newaxis]
    else:
        return 0.5 * (circular[..., 0] + circular[..., 3])[..., numpy.newaxis]


def convert_pol_frame(polvec, ipf: PolarisationFrame, opf: PolarisationFrame, polaxis=1):
    if ipf == opf:
        return polvec

    if ipf == PolarisationFrame("linear"):
        if opf == PolarisationFrame("stokesIQUV"):
            return convert_linear_to_stokes(polvec, polaxis)
        elif opf == PolarisationFrame("stokesI"):
            return convert_linear_to_stokesI(polvec)

    if ipf == PolarisationFrame("linearnp"):
        if opf == PolarisationFrame("stokesIQ"):
            return convert_linear_to_stokes(polvec, polaxis)
        elif opf == PolarisationFrame("stokesI"):
            return convert_linear_to_stokesI(polvec)

    if ipf == PolarisationFrame("circular"):
        if opf == PolarisationFrame("stokesIQUV"):
            return convert_circular_to_stokes(polvec, polaxis)
        elif opf == PolarisationFrame("stokesI"):
            return convert_circular_to_stokesI(polvec)

    if ipf == PolarisationFrame("circularnp"):
        if opf == PolarisationFrame("stokesIV"):
            return convert_circular_to_stokes(polvec, polaxis)
        elif opf == PolarisationFrame("stokesI"):
            return convert_circular_to_stokesI(polvec)

    if ipf == PolarisationFrame("stokesIQUV"):
        if opf == PolarisationFrame("linear"):
            return convert_stokes_to_linear(polvec, polaxis)
        elif opf == PolarisationFrame("circular"):
            return convert_stokes_to_circular(polvec, polaxis)

    if ipf == PolarisationFrame("stokesIQ"):
        if opf == PolarisationFrame("linearnp"):
            return convert_stokes_to_linear(polvec, polaxis)

    if ipf == PolarisationFrame("stokesIV"):
        if opf == PolarisationFrame("circularnp"):
            return convert_stokes_to_linear(polvec, polaxis)

    if ipf == PolarisationFrame("stokesI"):
        if opf == PolarisationFrame("stokesI"):
            return polvec
    
    raise ValueError("Unknown polarisation conversion: {} to {}".format(ipf, opf))


def correlate_polarisation(rec_frame: ReceptorFrame):
    """ Gives the polarisation frame corresponding to a receptor frame

    :param rec_frame: Receptor frame
    :return: PolarisationFrame
    """
    if rec_frame == ReceptorFrame("circular"):
        correlation = PolarisationFrame("circular")
    elif rec_frame == ReceptorFrame("linear"):
        correlation = PolarisationFrame("linear")
    elif rec_frame == ReceptorFrame("stokesI"):
        correlation = PolarisationFrame("stokesI")
    else:
        raise ValueError("Unknown receptor frame %s for correlation" % rec_frame)
    
    return correlation


def congruent_polarisation(rec_frame: ReceptorFrame, polarisation_frame: PolarisationFrame):
    """Are these receptor and polarisation frames congruent?

    """
    if rec_frame.type == "linear":
        return polarisation_frame.type in ["linear", "linearnp"]
    elif rec_frame.type == "circular":
        return polarisation_frame.type in ["circular", "circularnp"]
    elif rec_frame.type == "stokesI":
        return polarisation_frame.type == "stokesI"
    
    return False
