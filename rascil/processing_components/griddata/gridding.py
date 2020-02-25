""" Imaging is based on used of the FFT to perform Fourier transforms efficiently. Since the observed visibility data_models
do not arrive naturally on grid points, the sampled points are resampled on the FFT grid using a convolution function to
smear out the sample points. The resulting grid points are then FFT'ed. The result can be corrected for the griddata
convolution function by division in the image plane of the transform.

This module contains functions for performing the griddata process and the inverse degridding process.

The GridData data model is used to hold the specification of the desired result.
"""

__all__ = ['convolution_mapping_visibility', 'grid_visibility_to_griddata', 'grid_visibility_to_griddata_fast',
           'grid_visibility_weight_to_griddata', 'griddata_merge_weights', 'griddata_visibility_reweight',
           'fft_griddata_to_image',
           'degrid_visibility_from_griddata', 'fft_image_to_griddata', 'grid_blockvisibility_weight_to_griddata',
           'griddata_blockvisibility_reweight']

import logging

import astropy.constants as constants
import numpy
import numpy.testing

from rascil.data_models.memory_data_models import BlockVisibility, Visibility, GridData
from rascil.processing_components.fourier_transforms import ifft, fft
from rascil.processing_components.griddata.operations import copy_griddata
from rascil.processing_components.image.operations import create_image_from_array
from rascil.processing_components.visibility.base import copy_visibility

log = logging.getLogger('logger')


def convolution_mapping_visibility(vis, griddata, frequency, cf, channel_tolerance=1e-8):
    """Find the mappings between visibility, griddata, and convolution function

    :param vis:
    :param griddata:
    :param cf:
    :param channel_tolerance:
    :return:
    """
    
    assert isinstance(vis, Visibility), vis
    
    u = vis.uvw[:, 0]
    v = vis.uvw[:, 1]
    w = vis.uvw[:, 2]
    
    pu_grid, pu_offset, pv_grid, pv_offset, pwc_fraction, pwc_grid, pwg_fraction, pwg_grid = \
        spatial_mapping(cf, griddata, u, v, w)
    
    ###### Frequency mapping
    pfreq_pixel = griddata.grid_wcs.sub([5]).wcs_world2pix(frequency, 0)[0]
    # Find the nearest grid point
    pfreq_grid = numpy.round(pfreq_pixel).astype('int')
    assert numpy.min(pfreq_grid) >= 0, "Frequency axis underflows: %f" % numpy.max(pfreq_grid)
    assert numpy.max(pfreq_grid) < cf.shape[0], "Frequency axis overflows: %f" % numpy.max(pfreq_grid)
    pfreq_fraction = pfreq_pixel - pfreq_grid
    # If we are doing spectral imaging, check the tolerances
    is_spectral = cf.shape[0] > 1
    if is_spectral and (numpy.max(numpy.abs(pfreq_fraction)) > channel_tolerance):
        print("convolution_mapping_visibility: alignment of visibility and image frequency grids exceeds tolerance %s" %
              (numpy.max(pfreq_fraction)))
        log.warning(
            "convolution_mapping_visibility: alignment of visibility and image frequency grids exceeds tolerance %s" %
            (numpy.max(pfreq_fraction)))
    
    ######  TODO: Polarisation mapping
    
    return pu_grid, pu_offset, pv_grid, pv_offset, pwg_grid, pwg_fraction, pwc_grid, pwc_fraction, pfreq_grid


def convolution_mapping_blockvisibility(vis, griddata, frequency, cf, channel_tolerance=1e-8):
    """Find the mappings between visibility, griddata, and convolution function

    :param vis:
    :param griddata:
    :param cf:
    :param channel_tolerance:
    :return:
    """
    
    assert isinstance(vis, BlockVisibility), vis
    
    k = frequency / constants.c.value
    u = vis.uvw[..., 0].flat * k
    v = vis.uvw[..., 1].flat * k
    w = vis.uvw[..., 2].flat * k
    
    pu_grid, pu_offset, pv_grid, pv_offset, pwc_fraction, pwc_grid, pwg_fraction, pwg_grid = \
        spatial_mapping(cf, griddata, u, v, w)
    
    return pu_grid, pu_offset, pv_grid, pv_offset, pwg_grid, pwg_fraction, pwc_grid, pwc_fraction


def spatial_mapping(cf, griddata, u, v, w):
    """ Map u,v,w per row into coordinates in the grid
    
    :param cf:
    :param griddata:
    :param vis:
    :return:
    """
    
    numpy.testing.assert_almost_equal(griddata.grid_wcs.wcs.cdelt[0], cf.grid_wcs.wcs.cdelt[0], 7)
    numpy.testing.assert_almost_equal(griddata.grid_wcs.wcs.cdelt[1], cf.grid_wcs.wcs.cdelt[1], 7)
    ####### UV mapping
    # We use the grid_wcs's to do the coordinate conversion
    # Find the nearest grid points
    pu_grid, pv_grid = \
        numpy.round(griddata.grid_wcs.sub([1, 2]).wcs_world2pix(u, v, 0)).astype('int')
    assert numpy.min(pu_grid) >= 0, "U axis underflows: %f" % numpy.min(pu_grid)
    assert numpy.max(pu_grid) < griddata.shape[3], "U axis overflows: %f" % numpy.max(pu_grid)
    assert numpy.min(pv_grid) >= 0, "V axis underflows: %f" % numpy.min(pv_grid)
    assert numpy.max(pv_grid) < griddata.shape[4], "V axis overflows: %f" % numpy.max(pv_grid)
    # We now have the location of grid points, convert back to uv space and find the remainder (in wavelengths). We
    # then use this to calculate the subsampling indices (DUU, DVV)
    wu_grid, wv_grid = griddata.grid_wcs.sub([1, 2]).wcs_pix2world(pu_grid, pv_grid, 0)
    wu_subsample, wv_subsample = u - wu_grid, v - wv_grid
    pu_offset, pv_offset = \
        numpy.floor(cf.grid_wcs.sub([3, 4]).wcs_world2pix(wu_subsample, wv_subsample, 0)).astype('int')
    ###### W mapping for Grid
    # nchan, npol, w, v, u
    pwg_pixel = griddata.grid_wcs.sub([3]).wcs_world2pix(w, 0)[0]
    # Find the nearest grid point
    pwg_grid = numpy.round(pwg_pixel).astype('int')
    if numpy.min(pwg_grid) < 0:
        print(w[0:10, 2])
        print(cf.grid_wcs.sub([5]).__repr__())
    assert numpy.min(pwg_grid) >= 0, "W axis underflows: %f" % numpy.min(pwg_grid)
    assert numpy.max(pwg_grid) < cf.shape[2], "W axis overflows: %f" % numpy.max(pwg_grid)
    pwg_fraction = pwg_pixel - pwg_grid
    ###### W mapping for CF
    # nchan, npol, w, dv, du, v, u
    pwc_pixel = cf.grid_wcs.sub([5]).wcs_world2pix(w, 0)[0]
    pwc_grid = numpy.round(pwc_pixel).astype('int')
    if numpy.min(pwc_grid) < 0:
        print(w[0:10, 2])
        print(cf.grid_wcs.sub([5]).__repr__())
    assert numpy.min(pwc_grid) >= 0, "W axis underflows: %f" % numpy.min(pwc_grid)
    assert numpy.max(pwc_grid) < cf.shape[2], "W axis overflows: %f" % numpy.max(pwc_grid)
    pwc_fraction = pwc_pixel - pwc_grid
    return pu_grid, pu_offset, pv_grid, pv_offset, pwc_fraction, pwc_grid, pwg_fraction, pwg_grid


def grid_blockvisibility_to_griddata(vis, griddata, cf):
    """Grid Visibility onto a GridData

    :param vis: Visibility to be gridded
    :param griddata: GridData
    :param cf: Convolution function
    :return: GridData
    """
    
    assert isinstance(vis, BlockVisibility), vis
    
    vis_to_im = numpy.round(
        griddata.grid_wcs.sub([3]).wcs_world2pix(vis.frequency, 0)[0]).astype('int')
    
    griddata.data[...] = 0.0
    
    nrows, nants, _, nvchan, nvpol = vis.vis.shape
    fvist = vis.flagged_vis.reshape([nrows * nants * nants, nvchan, nvpol]).T
    fwtt = vis.flagged_imaging_weight.reshape([nrows * nants * nants, nvchan, nvpol]).T
    fvist *= fwtt
    
    # Do this in place to avoid creating a new copy. Doing the conjugation outside the loop
    # reduces run time immensely
    cf.data = numpy.conjugate(cf.data)
    _, _, _, _, _, gv, gu = cf.shape
    du = gu // 2
    dv = gv // 2
    
    for vchan in range(nvchan):
        imchan = vis_to_im[vchan]
        pu_grid, pu_offset, pv_grid, pv_offset, pwg_grid, pwg_fraction, pwc_grid, pwc_fraction = \
            convolution_mapping_blockvisibility(vis, griddata, vis.frequency[vchan], cf)
        for pol in range(nvpol):
            for row in range(nrows* nants * nants):
                griddata.data[imchan, \
                pol, \
                pwg_grid[row], \
                (pv_grid[row] - dv):(pv_grid[row] + dv), \
                (pu_grid[row] - du):(pu_grid[row] + du)] += \
                    cf.data[imchan, pol, pwc_grid[row], pv_offset[row], pu_offset[row], :, :] * fvist[pol, vchan, row]
    sumwt = numpy.sum(fwtt[..., :], axis=2).T
    
    cf.data = numpy.conjugate(cf.data)
    return griddata, sumwt


def grid_visibility_to_griddata(vis, griddata, cf):
    """Grid Visibility onto a GridData

    :param vis: Visibility to be gridded
    :param griddata: GridData
    :param cf: Convolution function
    :return: GridData
    """
    
    assert isinstance(vis, Visibility), vis
    
    nchan, npol, nz, oversampling, _, support, _ = cf.shape
    sumwt = numpy.zeros([nchan, npol])
    pu_grid, pu_offset, pv_grid, pv_offset, pwg_grid, pwg_fraction, pwc_grid, pwc_fraction, pfreq_grid = \
        convolution_mapping_visibility(vis, griddata, vis.frequency, cf)
    _, _, _, _, _, gv, gu = cf.shape
    coords = zip(vis.vis * vis.flagged_imaging_weight, vis.flagged_imaging_weight, pfreq_grid, pu_grid,
                 pu_offset, pv_grid, pv_offset, pwg_grid, pwc_grid)
    griddata.data[...] = 0.0
    
    # Do this in place to avoid creating a new copy. Doing the conjugation outside the loop
    # reduces run time immensely
    cf.data = numpy.conjugate(cf.data)
    
    du = gu // 2
    dv = gv // 2
    for v, vwt, chan, uu, uuf, vv, vvf, zzg, zzc in coords:
        griddata.data[chan, :, zzg, (vv - dv):(vv + dv), (uu - du):(uu + du)] += \
            cf.data[chan, :, zzc, vvf, uuf, :, :] * v[:, numpy.newaxis, numpy.newaxis]
        sumwt[chan, :] += vwt
    
    cf.data = numpy.conjugate(cf.data)
    return griddata, sumwt


def grid_visibility_to_griddata_fast(vis, griddata, cf, gcf):
    """Grid Visibility onto a GridData

    :param vis: Visibility to be gridded
    :param griddata: GridData
    :return: GridData
    """
    assert isinstance(vis, Visibility), vis
    
    nchan, npol, nz, ny, nx = griddata.shape
    sumwt = numpy.zeros([nchan, npol])
    pu_grid, pu_offset, pv_grid, pv_offset, pwg_grid, pwg_fraction, pwc_grid, pwc_fraction, pfreq_grid = \
        convolution_mapping_visibility(vis, griddata, vis.frequency, cf)
    _, _, _, _, _, gv, gu = cf.shape
    coords = zip(vis.vis, vis.flagged_imaging_weight, pfreq_grid, pu_grid, pv_grid, pwg_grid)
    griddata.data[...] = 0.0
    
    for v, vwt, chan, xx, yy, zzg in coords:
        griddata.data[chan, :, zzg, yy, xx] += v * vwt
        sumwt[chan, :] += vwt
    
    return griddata, sumwt


def grid_blockvisibility_weight_to_griddata(vis, griddata: GridData, cf):
    """Grid BlockVisibility weight onto a GridData

    :param vis: BlockVisibility to be gridded
    :param griddata: GridData
    :param cf: Convolution function
    :return: GridData
    """
    assert isinstance(vis, BlockVisibility), vis
    
    nchan, npol, nz, ny, nx = griddata.shape
    sumwt = numpy.zeros([nchan, npol])
    
    _, _, _, _, _, gv, gu = cf.shape
    vis_to_im = numpy.round(
        griddata.grid_wcs.sub([3]).wcs_world2pix(vis.frequency, 0)[0]).astype('int')
    
    griddata.data[...] = 0.0
    real_gd = numpy.real(griddata.data)
    
    nrows, nants, _, nvchan, nvpol = vis.vis.shape
    
    # Transpose to get row varying fastest
    fwtt = vis.flagged_imaging_weight.reshape([nrows * nants * nants, nvchan, nvpol]).T
    
    for vchan in range(nvchan):
        imchan = vis_to_im[vchan]
        pu_grid, pu_offset, pv_grid, pv_offset, pwg_grid, _, _, _ = \
            convolution_mapping_blockvisibility(vis, griddata, vis.frequency[vchan], cf)
        for pol in range(nvpol):
            for row in range(nrows * nants * nants):
                real_gd[imchan, pol, pwg_grid[row], pv_grid[row], pu_grid[row]] += fwtt[pol, vchan, row]
                sumwt[imchan, pol] += fwtt[pol, vchan, row]
    
    griddata.data = real_gd.astype("complex")
    
    return griddata, sumwt


def grid_visibility_weight_to_griddata(vis, griddata: GridData, cf):
    """Grid Visibility weight onto a GridData

    :param vis: Visibility to be gridded
    :param griddata: GridData
    :return: GridData
    """
    assert isinstance(vis, Visibility), vis
    
    nchan, npol, nz, ny, nx = griddata.shape
    sumwt = numpy.zeros([nchan, npol])
    pu_grid, pu_offset, pv_grid, pv_offset, pwg_grid, pwg_fraction, pwc_grid, pwc_fraction, pfreq_grid = \
        convolution_mapping_visibility(vis, griddata, vis.frequency, cf)
    _, _, _, _, _, gv, gu = cf.shape
    coords = zip(vis.flagged_imaging_weight, pfreq_grid, pu_grid, pv_grid, pwg_grid)
    griddata.data[...] = 0.0
    
    real_gd = numpy.real(griddata.data)
    for vwt, chan, xx, yy, zzg in coords:
        real_gd[chan, :, zzg, yy, xx] += vwt
        sumwt[chan, :] += vwt
    
    griddata.data = real_gd.astype("complex")
    
    return griddata, sumwt


def griddata_merge_weights(gd_list, algorithm='uniform'):
    """ Merge weights into one grid
    
    :param gd_list:
    :param gd:
    :param algorithm:
    :return:
    """
    centre = len(gd_list) // 2
    gd = copy_griddata(gd_list[centre][0])
    sumwt = gd_list[centre][1]
    
    frequency = 0.0
    bandwidth = 0.0
    
    for i, g in enumerate(gd_list):
        if i != centre:
            gd.data += g[0].data
            sumwt += g[1]
        frequency += g[0].grid_wcs.wcs.crval[4]
        bandwidth += g[0].grid_wcs.wcs.cdelt[4]
    
    gd.grid_wcs.wcs.cdelt[4] = bandwidth
    gd.grid_wcs.wcs.crval[4] = frequency / len(gd_list)
    return (gd, sumwt)


def griddata_visibility_reweight(vis, griddata, cf):
    """Reweight visibility weight using the weights in griddata

    :param vis: Visibility to be reweighted
    :param griddata: GridData holding gridded weights
    :param cf: Convolution function
    :return: Visibility with imaging_weights corrected
    """
    real_gd = numpy.real(griddata.data)
    
    vis_to_im = numpy.round(
        griddata.grid_wcs.sub([3]).wcs_world2pix(vis.frequency, 0)[0]).astype('int')
    
    nrows, nvpol = vis.vis.shape
    fwtt = vis.flagged_imaging_weight.T
    nvchan = len(numpy.unique(vis.frequency))
    for pol in range(nvpol):
        for vchan in range(nvchan):
            imchan = vis_to_im[vchan]
            frequency = vis.frequency[vchan]
            pu_grid, pu_offset, pv_grid, pv_offset, pwg_grid, pwg_fraction, pwc_grid, pwc_fraction, pfreq_grid = \
                convolution_mapping_visibility(vis, griddata, frequency, cf)
            for row in range(nrows):
                wt = real_gd[imchan, pol, pwg_grid[row], pv_grid[row], pu_grid[row]]
                if wt > 0.0:
                    fwtt[pol, row] /= wt

    vis.data['imaging_weight'][...] = fwtt.T

    return vis


def griddata_blockvisibility_reweight(vis, griddata, cf):
    """Reweight visibility weight using the weights in griddata

    :param vis: Visibility to be reweighted
    :param griddata: GridData holding gridded weights
    :param cf: Convolution function
    :return: visibility with imaging_weights corrected
    """
    nchan, npol, nz, ny, nx = griddata.shape
    nrows, nants, _, nvchan, nvpol = vis.vis.shape
    sumwt = numpy.zeros([nchan, npol])
    _, _, _, _, _, gv, gu = cf.shape
    vis_to_im = numpy.round(
        griddata.grid_wcs.sub([3]).wcs_world2pix(vis.frequency, 0)[0]).astype('int')
    
    real_gd = numpy.real(griddata.data)
    wgtt = vis.flagged_imaging_weight.reshape([nrows * nants * nants, nvchan, nvpol]).T
    
    for pol in range(nvpol):
        for vchan in range(nvchan):
            imchan = vis_to_im[vchan]
            frequency = vis.frequency[vchan]
            pu_grid, pu_offset, pv_grid, pv_offset, pwg_grid, pwg_fraction, pwc_grid, pwc_fraction = \
                convolution_mapping_blockvisibility(vis, griddata, frequency, cf)
            for row in range(nrows):
                wt = real_gd[imchan, pol, pwg_grid[row], pv_grid[row], pu_grid[row]]
                if wt > 0.0:
                    wgtt[pol, vchan, row] /= wt
    
    vis.data['imaging_weight'] = wgtt.T.reshape([nrows, nants, nants, nvchan, nvpol])
    
    return vis


def degrid_blockvisibility_from_griddata(vis, griddata, cf, **kwargs):
    """Degrid blockVisibility from a GridData

    :param vis: Visibility to be degridded
    :param griddata: GridData containing image
    :param cf: Convolution function (as GridData)
    :param kwargs:
    :return: Visibility
    """
    nchan, npol, nz, oversampling, _, support, _ = cf.shape
    vis_to_im = numpy.round(
        griddata.grid_wcs.sub([3]).wcs_world2pix(vis.frequency, 0)[0]).astype('int')
    
    nrows, nants, _, nvchan, nvpol = vis.vis.shape
    fwt = vis.flagged_imaging_weight.reshape([nrows * nants * nants, nvchan, nvpol])
    
    _, _, _, _, _, gv, gu = cf.shape
    
    newvis = copy_visibility(vis, zero=True)
    
    # coords = zip(pfreq_grid, pu_grid, pu_offset, pv_grid, pv_offset, pw_grid)
    
    du = gu // 2
    dv = gv // 2
    
    nvis = vis.vis.shape[0]
    
    fvis = vis.vis.reshape([nrows * nants * nants, nvchan, nvpol])
    
    for vchan in range(nvchan):
        imchan = vis_to_im[vchan]
        frequency = vis.frequency[vchan]
        pu_grid, pu_offset, pv_grid, pv_offset, pwg_grid, pwg_fraction, pwc_grid, pwc_fraction = \
            convolution_mapping_blockvisibility(vis, griddata, frequency, cf)
        coords = zip(fvis, pu_grid, pu_offset, pv_grid, pv_offset, pwg_grid, pwc_grid)
        for v, uu, uuf, vv, vvf, zzg, zzc in coords:
            v += numpy.einsum('ijk,ijk->i',
                              griddata.data[imchan, :, zzg, (vv - dv):(vv + dv), (uu - du):(uu + du)],
                              cf.data[imchan, :, zzc, vvf, uuf, :, :])
    
    return newvis


def degrid_visibility_from_griddata(vis, griddata, cf, **kwargs):
    """Degrid Visibility from a GridData

    :param vis: Visibility to be degridded
    :param griddata: GridData containing image
    :param cf: Convolution function (as GridData)
    :param kwargs:
    :return: Visibility
    """
    nchan, npol, nz, oversampling, _, support, _ = cf.shape
    pu_grid, pu_offset, pv_grid, pv_offset, pwg_grid, pwg_fraction, pwc_grid, pwc_fraction, pfreq_grid = \
        convolution_mapping_visibility(vis, griddata, vis.frequency, cf)
    _, _, _, _, _, gv, gu = cf.shape
    
    newvis = copy_visibility(vis, zero=True)
    
    # coords = zip(pfreq_grid, pu_grid, pu_offset, pv_grid, pv_offset, pw_grid)
    
    du = gu // 2
    dv = gv // 2
    
    nvis = vis.vis.shape[0]
    
    for ivis in range(nvis):
        chan, uu, uuf, vv, vvf, zzg, zzc = pfreq_grid[ivis], pu_grid[ivis], pu_offset[ivis], pv_grid[ivis], \
                                           pv_offset[ivis], pwg_grid[ivis], pwc_grid[ivis]
        # Use einsum to replace the following:
        # newvis.vis[i,:] = numpy.sum(griddata.data[chan, :, zzg, (vv - dv):(vv + dv), (uu - du):(uu + du)] *
        #                              cf.data[chan, :, zzc, vvf, uuf, :, :], axis=(1, 2))
        
        newvis.vis[ivis, :] += numpy.einsum('ijk,ijk->i',
                                            griddata.data[chan, :, zzg, (vv - dv):(vv + dv), (uu - du):(uu + du)],
                                            cf.data[chan, :, zzc, vvf, uuf, :, :])
    
    return newvis


def fft_griddata_to_image(griddata, gcf=None, imaginary=False):
    """ FFT griddata after applying gcf

    :param griddata:
    :param gcf: Grid correction image
    :return:
    """
    
    projected = numpy.sum(griddata.data, axis=2)
    ny, nx = projected.data.shape[-2], projected.data.shape[-1]
    
    if gcf is None:
        im_data = ifft(projected) * float(nx) * float(ny)
    else:
        im_data = ifft(projected) * gcf.data * float(nx) * float(ny)
    
    im_real = create_image_from_array(im_data.real, griddata.projection_wcs, griddata.polarisation_frame)
    
    if imaginary:
        im_imag = create_image_from_array(im_data.imag, griddata.projection_wcs, griddata.polarisation_frame)
        return im_real, im_imag
    else:
        return im_real


def fft_image_to_griddata(im, griddata, gcf=None):
    """Fill griddata with transform of im

    :param griddata:
    :param gcf: Grid correction image
    :return:
    """
    # chan, pol, z, u, v, w
    if gcf is None:
        griddata.data[:, :, :, ...] = fft(im.data)[:, :, numpy.newaxis, ...]
    else:
        griddata.data[:, :, :, ...] = fft(im.data * gcf.data)[:, :, numpy.newaxis, ...]
    
    return griddata
