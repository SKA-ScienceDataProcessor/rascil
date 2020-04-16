""" Support for coordinates in FFTs

All grids and images are considered quadratic and centered around
`npixel//2`, where `npixel` is the pixel width/height. This means that `npixel//2` is
the zero frequency for FFT purposes, as is convention. Note that this
means that for even `npixel` the grid is not symmetrical, which means that
e.g. for convolution kernels odd image sizes are preferred.

"""

__all__ = ['w_beam', 'grdsf', 'coordinates']

import logging

import numpy

log = logging.getLogger('logger')


def coordinateBounds(npixel):
    r""" Returns lowest and highest coordinates of an image/grid given:

    1. Step size is :math:`1/npixel`:

       .. math:: \frac{high-low}{npixel-1} = \frac{1}{npixel}

    2. The coordinate :math:`\lfloor npixel/2\rfloor` falls exactly on zero:

       .. math:: low + \left\lfloor\frac{npixel}{2}\right\rfloor * (high-low) = 0

    This is the coordinate system for shifted FFTs.
    """
    if npixel % 2 == 0:
        return -0.5, 0.5 * (npixel - 2) / npixel
    else:
        return -0.5 * (npixel - 1) / npixel, 0.5 * (npixel - 1) / npixel


def coordinates(npixel: int):
    """ 1D array which spans [-.5,.5[ with 0 at position npixel/2
    
    """
    return (numpy.arange(npixel) - npixel // 2) / npixel


def coordinates2(npixel: int):
    """Two dimensional grids of coordinates spanning -1 to 1 in each dimension

    1. a step size of 2/npixel and
    2. (0,0) at pixel (floor(n/2),floor(n/2))
    """
    return (numpy.mgrid[0:npixel, 0:npixel] - npixel // 2) / npixel


def coordinates2Offset(npixel: int, cx: int, cy: int, quadrant=False):
    """Two dimensional grids of coordinates centred on an arbitrary point.

    This is used for A and w beams.

    1. a step size of 2/npixel and
    2. (0,0) at pixel (cx, cy,floor(n/2))
    """
    if cx is None:
        cx = npixel // 2
    if cy is None:
        cy = npixel // 2
    if not quadrant:
        mg = numpy.mgrid[0:npixel, 0:npixel]
    else:
        # If npixel is even, we should create a grid with npixel//2+1
        mg = numpy.mgrid[0:npixel//2+1, 0:npixel//2+1]
    return (mg[0] - cy) / npixel, (mg[1] - cx) / npixel

def grdsf(nu):
    """Calculate PSWF using an old SDE routine re-written in Python

    Find Spheroidal function with M = 6, alpha = 1 using the rational
    approximations discussed by Fred Schwab in 'Indirect Imaging'.
    This routine was checked against Fred's SPHFN routine, and agreed
    to about the 7th significant digit.
    The griddata function is (1-NU**2)*GRDSF(NU) where NU is the distance
    to the edge. The grid correction function is just 1/GRDSF(NU) where NU
    is now the distance to the edge of the image.
    """
    p = numpy.array([[8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1],
                     [4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2]])
    q = numpy.array([[1.0000000e0, 8.212018e-1, 2.078043e-1],
                     [1.0000000e0, 9.599102e-1, 2.918724e-1]])

    _, np = p.shape
    _, nq = q.shape

    nu = numpy.abs(nu)

    nuend = numpy.zeros_like(nu)
    part = numpy.zeros(len(nu), dtype='int')
    part[(nu >= 0.0) & (nu < 0.75)] = 0
    part[(nu >= 0.75) & (nu <= 1.0)] = 1
    nuend[(nu >= 0.0) & (nu < 0.75)] = 0.75
    nuend[(nu >= 0.75) & (nu <= 1.0)] = 1.0

    delnusq = nu ** 2 - nuend ** 2

    top = p[part, 0]
    for k in range(1, np):
        top += p[part, k] * numpy.power(delnusq, k)

    bot = q[part, 0]
    for k in range(1, nq):
        bot += q[part, k] * numpy.power(delnusq, k)

    grdsf = numpy.zeros_like(nu)
    ok = (bot > 0.0)
    grdsf[ok] = top[ok] / bot[ok]
    ok = numpy.abs(nu > 1.0)
    grdsf[ok] = 0.0

    # Return the griddata function and the grid correction function
    return grdsf, (1 - nu ** 2) * grdsf


def w_beam(npixel, field_of_view, w, cx=None, cy=None, remove_shift=False):
    """ W beam, the fresnel diffraction pattern arising from non-coplanar baselines
    
    :param npixel: Size of the grid in pixels
    :param field_of_view: Field of view
    :param w: Baseline distance to the projection plane
    :param cx: location of delay centre def :npixel//2
    :param cy: location of delay centre def :npixel//2
    :param remove_shift: Remove overall phase shift at the centre of the image
    :return: npixel x npixel array with the far field
    """
    if cx is None:
        cx = npixel // 2
    if cy is None:
        cy = npixel // 2

    # Original codes
    # ly, mx = coordinates2Offset(npixel, cx, cy)
    # r2 = field_of_view**2*(ly ** 2 + mx ** 2)
    # ph = numpy.zeros_like(r2)
    # ph[r2 < 1.0] = -2 * numpy.pi * w * (1 - numpy.sqrt(1.0 - r2[r2 < 1.0]))
    # cp = numpy.zeros_like(r2, dtype='complex')
    # cp[r2 < 1.0] = numpy.exp(1j * ph[r2 < 1.0])
    # cp[r2 == 0] = 1.0 + 0j
    # if remove_shift:
    #     cp /= cp[npixel // 2, npixel // 2]

    # numpy.putmask
    # ly, mx = coordinates2Offset(npixel, cx, cy)
    # r2 = field_of_view**2*(ly ** 2 + mx ** 2)
    # ph = numpy.zeros_like(r2)
    # m = r2 < 1.0
    # cp = numpy.zeros_like(r2, dtype='complex')
    # numpy.putmask(ph, m, -2 * numpy.pi * w * (1 - numpy.sqrt(1.0 - r2)))
    # numpy.putmask(cp, m, numpy.exp(1j * ph))
    # numpy.putmask(cp, r2 == 0, 1.0 + 0j)
    # if remove_shift:
    #     cp /= cp[npixel // 2, npixel // 2]

    # numpy.putmask - 2
    # ly, mx = coordinates2Offset(npixel, cx, cy)
    # r2 = field_of_view ** 2 * (ly ** 2 + mx ** 2)
    # ph = -2 * numpy.pi * w * (1 - numpy.sqrt(1.0 - r2))
    # numpy.putmask(ph, r2 >= 1.0, 0)
    # cp = numpy.zeros_like(r2, dtype='complex')
    # cp = numpy.exp(1j * ph)
    # numpy.putmask(cp, r2 >= 1.0, 0 + 0j)
    # numpy.putmask(cp, r2 == 0, 1.0 + 0j)
    # if remove_shift:
    #     cp /= cp[npixel // 2, npixel // 2]

    # SubArray Copy Symmetrically
    ly, mx = coordinates2Offset(npixel, cx, cy, quadrant=True)
    r2 = field_of_view ** 2 * (ly ** 2 + mx ** 2)
    ph = -2 * numpy.pi * w * (1 - numpy.sqrt(1.0 - r2))
    numpy.putmask(ph, r2 >= 1.0, 0)
    cp = numpy.zeros_like(r2, dtype='complex')
    cp = numpy.exp(1j * ph)
    numpy.putmask(cp, r2 >= 1.0, 0 + 0j)
    numpy.putmask(cp, r2 == 0, 1.0 + 0j)
    # Correct for linear phase shift in faceting
    if remove_shift:
        cp /= cp[-1, -1]

    cp = numpy.pad(cp, ((0, int(cx) + npixel % 2 - 1), (0, int(cy) + npixel % 2 - 1)), 'reflect')

    # assert((cp==cp1).all())

    return cp


