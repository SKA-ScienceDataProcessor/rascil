#
"""
Functions that define and manipulate ConvolutionFunctions.

The griddata has axes [chan, pol, z, dy, dx, y, x] where z, y, x are spatial axes in either sky or Fourier plane. The
order in the WCS is reversed so the grid_WCS describes UU, VV, DUU, DVV, WW, STOKES, FREQ axes.

GridData can be used to hold the Fourier transform of an Image or gridded visibilities. In addition, the convolution
function can be stored in a GridData, most probably with finer spatial sampling.


"""

__all__ = ['create_convolutionfunction_from_image', 'copy_convolutionfunction', 'create_convolutionfunction_from_array',
           'convolutionfunction_sizeof', 'calculate_bounding_box_convolutionfunction', 'convert_convolutionfunction_to_image',
           'apply_bounding_box_convolutionfunction', 'qa_convolutionfunction']
import copy
import logging

import numpy
from astropy.wcs import WCS

from rascil.data_models.memory_data_models import ConvolutionFunction
from rascil.data_models.memory_data_models import QA
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.image.operations import create_image_from_array

log = logging.getLogger('logger')


def convolutionfunction_sizeof(cf: ConvolutionFunction):
    """ Return size in GB
    """
    return cf.size()


def create_convolutionfunction_from_array(data: numpy.array, grid_wcs: WCS, projection_wcs: WCS,
                                          polarisation_frame: PolarisationFrame) -> ConvolutionFunction:
    """ Create a convolution function from an array and wcs's
    
    The cf has axes [chan, pol, z, dy, dx, y, x] where z, y, x are spatial axes in either sky or Fourier plane. The
    order in the WCS is reversed so the grid_WCS describes UU, VV, WW, STOKES, FREQ axes
    
    The axes UU,VV have the same physical stride as the image, The axes DUU, DVV are subsampled.
    
    Convolution function holds the original sky plane projection in the projection_wcs.

    :param data: Numpy.array
    :param grid_wcs: Grid world coordinate system
    :param projection_wcs: Projection world coordinate system
    :param polarisation_frame: Polarisation Frame
    :return: GridData
    
    """
    fconvfunc = ConvolutionFunction()
    fconvfunc.polarisation_frame = polarisation_frame
    
    fconvfunc.data = data
    fconvfunc.grid_wcs = grid_wcs.deepcopy()
    fconvfunc.projection_wcs = projection_wcs.deepcopy()
    
    assert isinstance(fconvfunc, ConvolutionFunction), "Type is %s" % type(fconvfunc)
    return fconvfunc


def create_convolutionfunction_from_image(im: numpy.array, nz=1, zstep=1e15, ztype='WW', oversampling=8, support=16,
                                          grid_reference=1.0):
    """ Create a convolution function from an image

    The griddata has axes [chan, pol, z, dy, dx, y, x] where z, y, x are spatial axes in either sky or Fourier plane. The
    order in the WCS is reversed so the grid_WCS describes UU, VV, WW, STOKES, FREQ axes

    The axes UU,VV have the same physical stride as the image, The axes DUU, DVV are subsampled.

    Convolution function holds the original sky plane projection in the projection_wcs.

    :param grid_reference:
    :param im: Template Image
    :param nz: Number of z axes, usually z is W
    :param zstep: Step in z, usually z is W
    :param ztype: Type of z, usually 'WW'
    :param oversampling: Oversampling (size of dy, dx axes)
    :param support: Support of final convolution function (size of y, x axes)
    :return: Convolution Function

    """
    assert len(im.shape) == 4
    assert im.wcs.wcs.ctype[0] == 'RA---SIN'
    assert im.wcs.wcs.ctype[1] == 'DEC--SIN'
    
    d2r = numpy.pi / 180.0
    
    # WCS Coords are [x, y, dy, dx, z, pol, chan] where x, y, z are spatial axes in real space or Fourier space
    # Array Coords are [chan, pol, z, dy, dx, y, x] where x, y, z are spatial axes in real space or Fourier space
    cf_wcs = WCS(naxis=7)
    
    cf_wcs.wcs.ctype[0] = 'UU'
    cf_wcs.wcs.ctype[1] = 'VV'
    cf_wcs.wcs.ctype[2] = 'DUU'
    cf_wcs.wcs.ctype[3] = 'DVV'
    cf_wcs.wcs.ctype[4] = ztype
    cf_wcs.wcs.ctype[5] = im.wcs.wcs.ctype[2]
    cf_wcs.wcs.ctype[6] = im.wcs.wcs.ctype[3]
    
    cf_wcs.wcs.axis_types[0] = 0
    cf_wcs.wcs.axis_types[1] = 0
    cf_wcs.wcs.axis_types[2] = 0
    cf_wcs.wcs.axis_types[3] = 0
    cf_wcs.wcs.axis_types[4] = 0
    cf_wcs.wcs.axis_types[5] = im.wcs.wcs.axis_types[2]
    cf_wcs.wcs.axis_types[6] = im.wcs.wcs.axis_types[3]
    
    cf_wcs.wcs.crval[0] = 0.0
    cf_wcs.wcs.crval[1] = 0.0
    cf_wcs.wcs.crval[2] = 0.0
    cf_wcs.wcs.crval[3] = 0.0
    cf_wcs.wcs.crval[4] = 0.0
    cf_wcs.wcs.crval[5] = im.wcs.wcs.crval[2]
    cf_wcs.wcs.crval[6] = im.wcs.wcs.crval[3]
    
    cf_wcs.wcs.crpix[0] = float(support // 2) + grid_reference
    cf_wcs.wcs.crpix[1] = float(support // 2) + grid_reference
    cf_wcs.wcs.crpix[2] = float(oversampling // 2) + 1.0
    cf_wcs.wcs.crpix[3] = float(oversampling // 2) + 1.0
    cf_wcs.wcs.crpix[4] = float(nz // 2 + 1)
    cf_wcs.wcs.crpix[5] = im.wcs.wcs.crpix[2]
    cf_wcs.wcs.crpix[6] = im.wcs.wcs.crpix[3]
    
    # The sampling on the UU and VV axes should be the same as for the image.
    # The sampling on the DUU and DVV axes should be oversampling times finer.
    cf_wcs.wcs.cdelt[0] = 1.0 / (im.shape[3] * d2r * im.wcs.wcs.cdelt[0])
    cf_wcs.wcs.cdelt[1] = 1.0 / (im.shape[2] * d2r * im.wcs.wcs.cdelt[1])
    cf_wcs.wcs.cdelt[2] = cf_wcs.wcs.cdelt[0] / oversampling
    cf_wcs.wcs.cdelt[3] = cf_wcs.wcs.cdelt[1] / oversampling
    cf_wcs.wcs.cdelt[4] = zstep
    cf_wcs.wcs.cdelt[5] = im.wcs.wcs.cdelt[2]
    cf_wcs.wcs.cdelt[6] = im.wcs.wcs.cdelt[3]
    
    grid_data = im.data[..., numpy.newaxis, :, :].astype('complex')
    grid_data[...] = 0.0
    
    nchan, npol, ny, nx = im.shape
    
    fconvfunc = ConvolutionFunction()
    fconvfunc.polarisation_frame = im.polarisation_frame
    
    fconvfunc.data = numpy.zeros([nchan, npol, nz, oversampling, oversampling, support, support], dtype='complex')
    fconvfunc.grid_wcs = cf_wcs.deepcopy()
    fconvfunc.projection_wcs = im.wcs.deepcopy()
    
    assert isinstance(fconvfunc, ConvolutionFunction), "Type is %s" % type(fconvfunc)
    
    return fconvfunc


def convert_convolutionfunction_to_image(cf):
    """ Convert ConvolutionFunction to an image
    
    :param cf:
    :return:
    """
    return create_image_from_array(cf.data, cf.grid_wcs, cf.polarisation_frame)


def apply_bounding_box_convolutionfunction(cf, fractional_level=1e-4):
    """Apply a bounding box to a convolution function

    :param cf:
    :param fractional_level:
    :return: bounded convolution function
    """
    newcf = copy_convolutionfunction(cf)
    nx = newcf.data.shape[-1]
    ny = newcf.data.shape[-2]
    mask = numpy.max(numpy.abs(newcf.data), axis=(0, 1, 2, 3, 4))
    coords = numpy.argwhere(mask > fractional_level * numpy.max(numpy.abs(cf.data)))
    crpx = int(numpy.round(cf.grid_wcs.wcs.crpix[0]))
    crpy = int(numpy.round(cf.grid_wcs.wcs.crpix[1]))
    x0, y0 = coords.min(axis=0, initial=cf.data.shape[-1])
    dx = crpx - x0
    dy = crpy - y0
    x0 -= 1
    y0 -= 1
    x1 = crpx + dx - 1
    y1 = crpy + dy - 1
    newcf.data = newcf.data[..., y0:y1, x0:x1]
    nny, nnx = newcf.data.shape[-2], newcf.data.shape[-1]
    newcf.grid_wcs.wcs.crpix[0] += nnx / 2 - nx / 2
    newcf.grid_wcs.wcs.crpix[1] += nny / 2 - ny / 2
    return newcf


def calculate_bounding_box_convolutionfunction(cf, fractional_level=1e-4):
    """Calculate bounding boxes
    
    Returns a list of bounding boxes where each element is
    (z, (y0, y1), (x0, x1))
    
    These can be used in griddata/degridding.

    :param cf:
    :param fractional_level:
    :return: list of bounding boxes
    """
    bboxes = list()
    threshold = fractional_level * numpy.max(numpy.abs(cf.data))
    for z in range(cf.data.shape[2]):
        mask = numpy.max(numpy.abs(cf.data[:, :, z, ...]), axis=(0, 1, 2, 3))
        coords = numpy.argwhere(mask > threshold)
        x0, y0 = coords.min(axis=0, initial=cf.data.shape[-1])
        x1, y1 = coords.max(axis=0, initial=cf.data.shape[-1])
        bboxes.append((z, (y0, y1), (x0, x1)))
    return bboxes


def qa_convolutionfunction(cf, context="") -> QA:
    """Assess the quality of a convolutionfunction

    :param cf:
    :return: QA
    """
    assert isinstance(cf, ConvolutionFunction), cf
    data = {'shape': str(cf.data.shape),
            'max': numpy.max(cf.data),
            'min': numpy.min(cf.data),
            'rms': numpy.std(cf.data),
            'sum': numpy.sum(cf.data),
            'medianabs': numpy.median(numpy.abs(cf.data)),
            'median': numpy.median(cf.data)}
    
    qa = QA(origin="qa_image", data=data, context=context)
    return qa

def copy_convolutionfunction(cf):
    """Make a copy of a convolution function
    
    :param cf:
    :return:
    """
    assert isinstance(cf, ConvolutionFunction), cf
    fcf = ConvolutionFunction()
    fcf.polarisation_frame = cf.polarisation_frame
    fcf.data = copy.deepcopy(cf.data)
    fcf.projection_wcs = copy.deepcopy(cf.projection_wcs)
    fcf.grid_wcs = copy.deepcopy(cf.grid_wcs)
    if convolutionfunction_sizeof(fcf) >= 1.0:
        log.debug("copy_convolutionfunction: copied %s convolution function of shape %s, size %.3f (GB)" %
                  (fcf.data.dtype, str(fcf.shape), convolutionfunction_sizeof(fcf)))
    assert isinstance(fcf, ConvolutionFunction), fcf
    return fcf
