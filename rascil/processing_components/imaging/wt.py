"""
Functions that implement prediction of and imaging from visibilities using the simple, w-projection and w-towers gridder.

https://gitlab.com/ska-telescope/py-wtowers

"""

__all__ = ['vis2dirty', 'gcf2wkern', 'predict_wt', 'invert_wt']

import logging
from typing import Union

import numpy

from rascil.data_models.memory_data_models import Visibility, BlockVisibility, Image
from rascil.data_models.parameters import get_parameter
from rascil.data_models.polarisation import convert_pol_frame
from rascil.processing_components.griddata import convert_kernel_to_list
from rascil.processing_components.image.operations import copy_image, image_is_canonical, export_image_to_fits
from rascil.processing_components.imaging.base import shift_vis_to_image, normalize_sumwt, create_image_from_array
from rascil.processing_components.visibility.base import copy_visibility
from rascil.processing_components.fourier_transforms import ifft, fft

log = logging.getLogger(__name__)

try:
    import wtowers.wtowers as wtowers
    
    
    def vis2dirty(grid_size, theta, wtvis, wtkern=None, subgrid_size=None, margin=None, winc=None):
        uvgrid = numpy.zeros(grid_size * grid_size, dtype=numpy.complex128)
        if wtkern is None:
            # Simple gridding
            flops = wtowers.grid_simple_func(uvgrid, grid_size, theta, wtvis)
        else:
            if subgrid_size == None or margin == None or winc == None:
                # W-projection gridding
                flops = wtowers.grid_wprojection_func(uvgrid, grid_size, theta, wtvis, wtkern)
            else:
                # W-towers gridding
                flops = wtowers.grid_wtowers_func(uvgrid, grid_size, theta, wtvis, wtkern, subgrid_size, margin, winc)
        # Fill a hermitian conjugated part of the uv_grid plane
        wtowers.make_hermitian_func(uvgrid, grid_size)
        # Create a dirty image and show
        uvgrid = uvgrid.reshape((grid_size, grid_size))
        img = ifft(uvgrid) * float(grid_size)**2
        return numpy.real(img)
    
    
    def dirty2vis(dirty, grid_size, theta, wtvis, wtkern=None, subgrid_size=None, margin=None, winc=None):
        # 1. Make uvgrid
        uvgrid = fft(dirty)
        # 2. Make degridding
        if wtkern is None:
            # Simple degridding
            flops = wtowers.degrid_simple_func(wtvis, uvgrid.reshape((grid_size * grid_size)), grid_size, theta)
        else:
            if subgrid_size == None or margin == None or winc == None:
                # W-projection degridding
                flops = wtowers.degrid_wprojection_func(wtvis, uvgrid.reshape((grid_size * grid_size)), grid_size,
                                                        theta, wtkern)
            else:
                # W-towers degridding
                flops = wtowers.degrid_wtowers_func(wtvis, uvgrid.reshape((grid_size * grid_size)), grid_size, theta,
                                                    wtkern, subgrid_size, margin, winc)
        # 3. Return wtvis
        return wtvis
    
    
    def gcf2wkern(gcfcf, wtkern, crocodile=False, im=None, NpixFF=1024, conjugate=False):
        # Get data and metadata from the input gcfcf
        plane_count, wplanes, wmin, wmax, wstep, size_y, size_x, oversampling = convert_kernel_to_list(gcfcf)
        # Allocate memory for wtkern structure
        status = wtowers.wkernel_allocate_func(wtkern, plane_count, size_x, size_y, oversampling)
        # Copy the rest of the metadata to wtkern
        wtkern.w_min = wmin[0]
        wtkern.w_max = wmax[0]
        wtkern.w_step = wstep
        wtkern.oversampling = oversampling
        
        if NpixFF is None:
            NpixFF = 1024
        if im is not None and crocodile == True:
            npixdirty = im.nwidth
            pixsize = numpy.abs(numpy.radians(im.wcs.wcs.cdelt[0]))
            # theta = numpy.cos(numpy.pi / 2 - npixdirty * pixsize)
            theta = npixdirty * pixsize
            # print(numpy.cos(numpy.pi / 2 - npixdirty * pixsize), theta)

        # Copy w-kernels into wtkern.kern_by_w
        for i in range(plane_count):
            w = wplanes[i][0][0]
            wtkern.kern_by_w[i].w = wplanes[i][0][0]
            if crocodile == True:
                kern = w_kernel(theta, wtkern.kern_by_w[i].w, NpixFF=NpixFF, NpixKern=size_x, Qpx=oversampling)
                wplanes_f = kern
            else:
                wplanes_f = numpy.asarray(wplanes[i][1])

            #       	for iy in range(size_y):
            #           		for ix in range(size_x):
            #               		for offy in range(oversampling):
            #                   			for offx in range(oversampling):
            #               				idx = ix + iy*size_x
            #               				wtkern.kern_by_w[i].data[2*idx]   = numpy.real(wplanes[i][1][offy][offx][ix][iy])
            #               				wtkern.kern_by_w[i].data[2*idx+1] = numpy.imag(wplanes[i][1][offy][offx][ix][iy])
            do_differences = False
            if do_differences:
                npixdirty = im.nwidth
                pixsize = numpy.abs(numpy.radians(im.wcs.wcs.cdelt[0]))
                # theta = numpy.cos(numpy.pi / 2 - npixdirty * pixsize)
                theta = npixdirty * pixsize
                newshape = [oversampling * size_y, oversampling * size_x]
                kern_croc = w_kernel(theta, wtkern.kern_by_w[i].w, NpixFF=NpixFF, NpixKern=size_x, Qpx=oversampling)
                kern_rascil = numpy.asarray(wplanes[i][1])
                kern_diff = kern_croc - kern_rascil
                print(i, plane_count, w, numpy.max(numpy.abs(kern_croc)), numpy.max(numpy.abs(kern_rascil)),
                      numpy.max(numpy.abs(kern_diff)))
                import matplotlib.pyplot as plt
                kern_croc_2d = unpack(kern_croc, oversampling, size_x, size_y)
                kern_rascil_2d = unpack(kern_rascil, oversampling, size_x, size_y)
                kern_diff_2d = unpack(kern_diff, oversampling, size_x, size_y)
                if w == 0.0:
                    print(numpy.unravel_index(numpy.abs(kern_croc_2d).argmax(), kern_croc_2d.shape))
                    print(numpy.unravel_index(numpy.abs(kern_rascil_2d).argmax(), kern_rascil_2d.shape))
                plt.clf()
                plt.subplot(231)
                plt.imshow(numpy.real(kern_croc_2d))
                plt.colorbar(shrink=0.25)
                plt.title('Crocodile Real {:.1f}'.format(w))
                plt.subplot(232)
                plt.imshow(numpy.real(kern_rascil_2d))
                plt.colorbar(shrink=0.25)
                plt.title('RASCIL Real {:.1f}'.format(w))
                plt.subplot(233)
                plt.imshow(numpy.real(kern_diff_2d))
                plt.colorbar(shrink=0.25)
                plt.title('Diff Real {:.1f}'.format(w))
                plt.subplot(234)
                plt.imshow(numpy.imag(kern_croc_2d))
                plt.colorbar(shrink=0.25)
                plt.title('Crocodile Imag {:.1f}'.format(w))
                plt.subplot(235)
                plt.imshow(numpy.imag(kern_rascil_2d))
                plt.colorbar(shrink=0.25)
                plt.title('RASCIL Imag {:.1f}'.format(w))
                plt.subplot(236)
                plt.imshow(numpy.imag(kern_diff_2d))
                plt.colorbar(shrink=0.25)
                plt.title('Diff Imag {:.1f}'.format(w))
                plt.savefig("test_results/kernels_{:.1f}.png".format(w))
                plt.show(block=False)

            wplanes_f = wplanes_f.flatten()

            if conjugate:
                for idx in range(wplanes_f.shape[0]):
                    wtkern.kern_by_w[i].data[2 * idx] = numpy.real(wplanes_f[idx])
                    wtkern.kern_by_w[i].data[2 * idx + 1] = - numpy.imag(wplanes_f[idx])
            else:
                for idx in range(wplanes_f.shape[0]):
                    wtkern.kern_by_w[i].data[2 * idx] = numpy.real(wplanes_f[idx])
                    wtkern.kern_by_w[i].data[2 * idx + 1] = numpy.imag(wplanes_f[idx])

        return wtkern


    def retile(kern_diff, oversampling, size_x, size_y):
        wplanes_f_2d = numpy.zeros([size_y * oversampling, size_x * oversampling], dtype='complex')
        for offy in range(oversampling):
            for offx in range(oversampling):
                wplanes_f_2d[(offy * size_y):((offy + 1) * size_y),
                (offx * size_x):((offx + 1) * size_x)] = kern_diff[offy, offx, :, :]
        return wplanes_f_2d

    def unpack(kern_diff, oversampling, size_x, size_y):
        wplanes_f_2d = numpy.zeros([size_y * oversampling, size_x * oversampling], dtype='complex')
        for y in range(size_y):
            for x in range(size_x):
                for offy in range(oversampling):
                    for offx in range(oversampling):
                        wplanes_f_2d[offy + oversampling * y, offx + oversampling * x] = \
                            kern_diff[offy, offx, size_y - y - 1, size_x - x - 1]
        return wplanes_f_2d


    # A part of crocodile.synthesis for wkernel calculation
    
    def coordinates2(N):
        """Two dimensional grids of coordinates spanning -1 to 1 in each
        dimension, with

        1. a step size of 2/N and
        2. (0,0) at pixel (floor(n/2),floor(n/2))

        :returns: pair (cx,cy) of 2D coordinate arrays
        """
        
        N2 = N // 2
        if N % 2 == 0:
            return numpy.mgrid[-N2:N2, -N2:N2][::-1] / N
        else:
            return numpy.mgrid[-N2:N2 + 1, -N2:N2 + 1][::-1] / N
    
    
    def extract_oversampled(a, xf, yf, Qpx, N):
        """
        Extract the (xf-th,yf-th) w-kernel from the oversampled parent

        Offsets are suitable for correcting of fractional coordinates,
        e.g. an offset of (xf,yf) results in the kernel for an (-xf,-yf)
        sub-grid offset.

        We do not want to make assumptions about the source grid's symetry
        here, which means that the grid's side length must be at least
        Qpx*(N+2) to contain enough information in all circumstances

        :param a: grid from which to extract
        :param ox: x offset
        :param oy: y offset
        :param Qpx: oversampling factor
        :param N: size of section
        """
        
        assert xf >= 0 and xf < Qpx
        assert yf >= 0 and yf < Qpx
        # Determine start offset.
        Na = a.shape[0]
        my = Na // 2 - Qpx * (N // 2) - yf
        mx = Na // 2 - Qpx * (N // 2) - xf
        assert mx >= 0 and my >= 0
        # Extract every Qpx-th pixel
        mid = a[my: my + Qpx * N: Qpx,
              mx: mx + Qpx * N: Qpx]
        # normalise
        return Qpx * Qpx * mid
    
    
    def pad_mid(ff, N):
        """
        Pad a far field image with zeroes to make it the given size.

        Effectively as if we were multiplying with a box function of the
        original field's size, which is equivalent to a convolution with a
        sinc pattern in the uv-grid.
    
        :param ff: The input far field. Should be smaller than NxN.
        :param N:  The desired far field size

        """
        
        N0, N0w = ff.shape
        if N == N0: return ff
        assert N > N0 and N0 == N0w
        return numpy.pad(ff,
                         pad_width=2 * [(N // 2 - N0 // 2, (N + 1) // 2 - (N0 + 1) // 2)],
                         mode='constant',
                         constant_values=0.0)
    
    
    def crocodile_ifft(a):
        """ Fourier transformation from grid to image space

        :param a: `uv` grid to transform
        :returns: an image in `lm` coordinate space
        """
        return numpy.fft.fftshift(numpy.fft.ifft2(numpy.fft.ifftshift(a)))
    
    
    def kernel_coordinates(N, theta, dl=0, dm=0, T=None):
        """
        Returns (l,m) coordinates for generation of kernels
        in a far-field of the given size.

        If coordinate transformations are passed, they must be inverse to
        the transformations applied to the visibilities using
        visibility_shift/uvw_transform.

        :param N: Desired far-field resolution
        :param theta: Field of view size (directional cosines)
        :param dl: Pattern horizontal shift (see visibility_shift)
        :param dm: Pattern vertical shift (see visibility_shift)
        :param T: Pattern transformation matrix (see uvw_transform)
        :returns: Pair of (m,l) coordinates
        """
        
        l, m = coordinates2(N) * theta
        if not T is None:
            l, m = T[0, 0] * l + T[1, 0] * m, T[0, 1] * l + T[1, 1] * m
        return l + dl, m + dm
    
    
    def w_kernel_function(l, m, w, dl=0, dm=0, T=numpy.eye(2)):
        """W beam, the fresnel diffraction pattern arising from non-coplanar baselines

        For the w-kernel, shifting the kernel pattern happens to also
        shift the kernel depending on w. To counter this effect, `dl` or
        `dm` can be passed so that the kernel ends up approximately
        centered again. This means that kernels will have to be used at an
        offset to get the same result, use `visibility_recentre` to
        achieve this.

        :param l: Horizontal image coordinates
        :param m: Vertical image coordinates
        :param N: Size of the grid in pixels
        :param w: Baseline distance to the projection plane
        :param dl: Shift the kernel by `dl w` to re-center it after a pattern shift.
        :param dm: Shift the kernel by `dm w` to re-center it after a pattern shift.
        :returns: N x N array with the far field
        """
        
        r2 = l ** 2 + m ** 2
        assert numpy.all(r2 < 1.0), "Error in image coordinate system: l %s, m %s" % (l, m)
        ph = 1 - numpy.sqrt(1.0 - r2) - dl * l - dm * m
        cp = numpy.exp(2j * numpy.pi * w * ph)
        return cp
    
    
    def kernel_oversample(ff, N, Qpx, s):
        """
        Takes a farfield pattern and creates an oversampled convolution
        function.

        If the far field size is smaller than N*Qpx, we will pad it. This
        essentially means we apply a sinc anti-aliasing kernel by default.

        :param ff: Far field pattern
        :param N:  Image size without oversampling
        :param Qpx: Factor to oversample by -- there will be Qpx x Qpx convolution arl
        :param s: Size of convolution function to extract
        :returns: Numpy array of shape [ov, ou, v, u], e.g. with sub-pixel
          offsets as the outer coordinates.
        """
        
        # Pad the far field to the required pixel size
        padff = pad_mid(ff, N * Qpx)
        
        # Obtain oversampled uv-grid
        af = crocodile_ifft(padff)
        
        # Extract kernels
        res = [[extract_oversampled(af, x, y, Qpx, s) for x in range(Qpx)] for y in range(Qpx)]
        return numpy.array(res)
    
    
    def w_kernel(theta, w, NpixFF, NpixKern, Qpx, **kwargs):
        """
        The middle s pixels of W convolution kernel. (W-KERNel-Aperture-Function)

        :param theta: Field of view (directional cosines)
        :param w: Baseline distance to the projection plane
        :param NpixFF: Far field size. Must be at least NpixKern+1 if Qpx > 1, otherwise NpixKern.
        :param NpixKern: Size of convolution function to extract
        :param Qpx: Oversampling, pixels will be Qpx smaller in aperture
          plane than required to minimially sample theta.

        :returns: [Qpx,Qpx,s,s] shaped oversampled convolution kernels
        """
        assert NpixFF > NpixKern or (NpixFF == NpixKern and Qpx == 1)
        
        l, m = kernel_coordinates(NpixFF, theta, **kwargs)
        kern = w_kernel_function(l, m, w)
        return kernel_oversample(kern, NpixFF, Qpx, NpixKern)
    
    
    def predict_wt(bvis: BlockVisibility, model: Image, gcfcf=None, **kwargs) -> \
            BlockVisibility:
        """ Predict using convolutional degridding.
        
        Wtowers version. https://gitlab.com/ska-telescope/py-wtowers
    
        :param gcfcf:
        :param bvis: BlockVisibility to be predicted
        :param model: model image
        :return: resulting BlockVisibility (in place works)
        """
        
        assert isinstance(bvis, BlockVisibility), bvis
        assert image_is_canonical(model)
        
        if model is None:
            return bvis
        
        newbvis = copy_visibility(bvis, zero=True)
        
        # Read wtowers-related parameters
        subgrid_size = get_parameter(kwargs, "subgrid_size")
        margin = get_parameter(kwargs, "margin")
        winc = get_parameter(kwargs, "winc")
        crocodile = get_parameter(kwargs, "crocodile")
        NpixFF = get_parameter(kwargs, "NpixFF")
        
        # Create an empty vis_data structure
        wtvis = wtowers.VIS_DATA()
        
        # Define the data to copy to wtvis
        antenna_count = bvis.data['uvw'].shape[1]
        bl_count = int(bvis.data.shape[0] * antenna_count * (
                    antenna_count - 1) / 2)  # bl_count = ntimes*nbases, each block will contain vis for one time count and one baseline
        # nfreq = bvis.frequency.shape[0] # the same number of frequencies for each block
        
        # Allocate memory for wtvis structure, time_count = 1, freq_count = 1
        status = wtowers.fill_vis_data_func(wtvis, antenna_count, bl_count, 1, 1)
        
        ibl = 0
        # Loop over ntime and and antennas in BlockVis, copy metadata
        for itime in range(bvis.data.shape[0]):
            for i1 in range(antenna_count):
                for i2 in range(i1 + 1, antenna_count):
                    wtvis.bl[ibl].antenna1 = i1
                    wtvis.bl[ibl].antenna2 = i2
                    wtvis.bl[ibl].time[0] = bvis.data['time'][itime]
                    wtvis.bl[ibl].freq[0] = 0.0
                    wtvis.bl[ibl].vis[0] = 0.0
                    wtvis.bl[ibl].vis[1] = 0.0
                    wtvis.bl[ibl].uvw[0] = - bvis.data['uvw'][itime, i2, i1, 0]
                    wtvis.bl[ibl].uvw[1] = bvis.data['uvw'][itime, i2, i1, 1]
                    wtvis.bl[ibl].uvw[2] = bvis.data['uvw'][itime, i2, i1, 2]
                    ibl += 1
                    # Fill stats
        status = wtowers.fill_stats_func(wtvis)
        
        # Fill wkern structure if gcfcf is provided
        wtkern = None
        if gcfcf is not None:
            wtkern = wtowers.W_KERNEL_DATA()
            wtkern = gcf2wkern(gcfcf, wtkern, crocodile, model, NpixFF, conjugate=True)
            
            # Extracting data from BlockVisibility
        freq = bvis.frequency  # frequency, Hz
        nrows, nants, _, vnchan, vnpol = bvis.vis.shape
        
        # uvw = newbvis.data['uvw'].reshape([nrows * nants * nants, 3])
        # vis = newbvis.data['vis'].reshape([nrows * nants * nants, vnchan, vnpol])
        
        # vis[...] = 0.0 + 0.0j  # Make all vis data equal to 0 +0j
        
        # Get the image properties
        m_nchan, m_npol, ny, nx = model.data.shape
        # Check if the number of frequency channels matches in bvis and a model
        #        assert (m_nchan == v_nchan)
        assert (m_npol == vnpol)
        
        # fuvw = uvw.copy()
        # We need to flip the u and w axes. The flip in w is equivalent to the conjugation of the
        # convolution function grid_visibility to griddata
        # fuvw[:, 0] *= -1.0
        # fuvw[:, 2] *= -1.0
        
        # Find out the image size/resolution
        npixdirty = model.nwidth
        pixsize = numpy.abs(numpy.radians(model.wcs.wcs.cdelt[0]))
        
        # Define WTowers FoV in direction cosine units
        # theta = numpy.cos(numpy.pi / 2 - npixdirty * pixsize)
        theta = npixdirty * pixsize
        grid_size = nx
        
        # Make de-gridding over a frequency range and pol fields
        vis_to_im = numpy.round(model.wcs.sub([4]).wcs_world2pix(freq, 0)[0]).astype('int')
        for vchan in range(vnchan):
            imchan = vis_to_im[vchan]
            for vpol in range(vnpol):
                # Fill the frequency
                for ibl in range(wtvis.bl_count):
                    wtvis.bl[ibl].freq[0] = freq[vchan]
                wtvis = dirty2vis(model.data[imchan, vpol, :, :].astype(numpy.float64), grid_size, theta, wtvis,
                                  wtkern=wtkern, subgrid_size=subgrid_size, margin=margin, winc=winc)
                
                # Fill the vis and frequency data in wtvis
                ibl = 0
                # Loop over ntime and and antennas in BlockVis, copy metadata
                for itime in range(bvis.data.shape[0]):
                    for i1 in range(antenna_count):
                        for i2 in range(i1 + 1, antenna_count):
                            newbvis.data['vis'][itime, i2, i1, vchan, vpol] = wtvis.bl[ibl].vis[0] + 1j * \
                                                                              wtvis.bl[ibl].vis[1]
                            ibl += 1
        
        assert numpy.max(numpy.abs(newbvis.data['vis'])) > 0.0
        newbvis.data['vis'] = convert_pol_frame(newbvis.data['vis'], model.polarisation_frame, bvis.polarisation_frame,
                                                polaxis=4)
        assert numpy.max(numpy.abs(newbvis.data['vis'])) > 0.0

        # Now we can shift the visibility from the image frame to the original visibility frame
        return shift_vis_to_image(newbvis, model, tangent=True, inverse=True)
    
    
    def invert_wt(bvis: BlockVisibility, model: Image, dopsf: bool = False, normalize: bool = True, gcfcf=None,
                  **kwargs) -> (Image, numpy.ndarray):
        """ Invert using py-wtowers module
        
        https://gitlab.com/ska-telescope/py-wtowers
    
        Use the image im as a template. Do PSF in a separate call.
    
        This is at the bottom of the layering i.e. all transforms are eventually expressed in terms
        of this function. . Any shifting needed is performed here.
    
        :param bvis: BlockVisibility to be inverted
        :param im: image template (not changed)
        :param normalize: Normalize by the sum of weights (True)
        :param gcfcf:
        :param dopsf: Make a PSF
        :return: (resulting image, sum of the weights for each frequency and polarization)
    
        """
        assert image_is_canonical(model)
        
        assert isinstance(bvis, BlockVisibility), bvis
        
        im = copy_image(model)
        
        # Read wtowers-related parameters
        subgrid_size = get_parameter(kwargs, "subgrid_size")
        margin = get_parameter(kwargs, "margin")
        winc = get_parameter(kwargs, "winc")
        crocodile = get_parameter(kwargs, "crocodile", False)
        NpixFF = get_parameter(kwargs, "NpixFF")
        
        sbvis = copy_visibility(bvis)
        sbvis = shift_vis_to_image(sbvis, im, tangent=True, inverse=False)
        
        # Create an empty vis_data structure
        wtvis = wtowers.VIS_DATA()
        
        # Define the data to copy to wtvis
        antenna_count = bvis.data['uvw'].shape[1]
        bl_count = int(bvis.data.shape[0] * antenna_count * (
                    antenna_count - 1) / 2)  # bl_count = ntimes*nbases, each block will contain vis for one time count and one baseline
        # nfreq = bvis.frequency.shape[0] # the same number of frequencies for each block
        
        # Allocate memory for wtvis structure, time_count = 1, freq_count = 1
        status = wtowers.fill_vis_data_func(wtvis, antenna_count, bl_count, 1, 1)
        
        ibl = 0
        # Loop over ntime and and antennas in BlockVis, copy metadata
        for itime in range(bvis.data.shape[0]):
            for i1 in range(antenna_count):
                for i2 in range(i1 + 1, antenna_count):
                    wtvis.bl[ibl].antenna1 = i1
                    wtvis.bl[ibl].antenna2 = i2
                    wtvis.bl[ibl].time[0] = bvis.data['time'][itime]
                    wtvis.bl[ibl].freq[0] = 0.0
                    wtvis.bl[ibl].vis[0] = 0.0
                    wtvis.bl[ibl].vis[1] = 0.0
                    wtvis.bl[ibl].uvw[0] = - bvis.data['uvw'][itime, i2, i1, 0]
                    wtvis.bl[ibl].uvw[1] = bvis.data['uvw'][itime, i2, i1, 1]
                    wtvis.bl[ibl].uvw[2] = bvis.data['uvw'][itime, i2, i1, 2]
                    ibl += 1
                    # Fill stats
        status = wtowers.fill_stats_func(wtvis)
        
        # Fill wkern structure if gcfcf is provided
        wtkern = None
        if gcfcf is not None:
            wtkern = wtowers.W_KERNEL_DATA()
            wtkern = gcf2wkern(gcfcf, wtkern, crocodile, im, NpixFF)
        
        vis = bvis.vis
        
        freq = sbvis.frequency  # frequency, Hz
        
        nrows, nants, _, vnchan, vnpol = vis.shape
        
        # NG-related
        flags = sbvis.flags.reshape([nrows * nants * nants, vnchan, vnpol])
        uvw = sbvis.uvw.reshape([nrows * nants * nants, 3])
        ms = sbvis.flagged_vis.reshape([nrows * nants * nants, vnchan, vnpol])
        wgt = sbvis.flagged_imaging_weight.reshape([nrows * nants * nants, vnchan, vnpol])
        ##########################
        
        # Re-write for the wt vis_data structure
        if dopsf:
            ms[...] = (1 - flags).astype('complex')
        
        # NG-related
        # if epsilon > 5.0e-6:
        #    ms = ms.astype("c8")
        #    wgt = wgt.astype("f4")
        ###########################
        
        # Find out the image size/resolution
        npixdirty = im.nwidth
        pixsize = numpy.abs(numpy.radians(im.wcs.wcs.cdelt[0]))
        
        # Define WTowers FoV in direction cosine units
        # theta = numpy.cos(numpy.pi / 2 - npixdirty * pixsize)
        theta = npixdirty * pixsize
        # Find the grid size in uvlambda
        uvlambda_init = numpy.maximum(numpy.abs(numpy.amin(bvis.data['uvw'])),
                                      numpy.abs(numpy.amax(bvis.data['uvw'])))  # m
        uvlambda_init = 2.1 * uvlambda_init
        freq_max = numpy.max(bvis.frequency)  # Hz
        clight = 299792458.  # m/s
        uvlambda_init = uvlambda_init * freq_max / clight
        uvlambda = numpy.double(npixdirty / theta)
        grid_size = int(theta * uvlambda)
        assert (uvlambda >= uvlambda_init)
        #################################
        
        nchan, npol, ny, nx = im.shape
        im.data[...] = 0.0
        sumwt = numpy.zeros([nchan, npol])
        
        ms = convert_pol_frame(ms, bvis.polarisation_frame, im.polarisation_frame, polaxis=2)
        # There's a latent problem here with the weights.
        # wgt = numpy.real(convert_pol_frame(wgt, bvis.polarisation_frame, im.polarisation_frame, polaxis=2))
        
        # Set up the conversion from visibility channels to image channels
        vis_to_im = numpy.round(model.wcs.sub([4]).wcs_world2pix(freq, 0)[0]).astype('int')
        for vchan in range(vnchan):
            ichan = vis_to_im[vchan]
            for pol in range(npol):
                # Nifty gridder likes to receive contiguous arrays
                # ms_1d = numpy.array([ms[row, vchan:vchan+1, pol] for row in range(nrows * nants * nants)], dtype='complex')
                # ms_1d.reshape([ms_1d.shape[0], 1])
                # wgt_1d = numpy.array([wgt[row, vchan:vchan+1, pol] for row in range(nrows * nants * nants)])
                # wgt_1d.reshape([wgt_1d.shape[0], 1])
                
                # Fill the vis and frequency data in wtvis
                ibl = 0
                # Loop over ntime and and antennas in BlockVis, copy metadata
                for itime in range(bvis.data.shape[0]):
                    for i1 in range(antenna_count):
                        for i2 in range(i1 + 1, antenna_count):
                            wtvis.bl[ibl].freq[0] = freq[vchan]
                            wtvis.bl[ibl].vis[0] = numpy.real(bvis.data['vis'][itime, i2, i1, vchan, pol])
                            wtvis.bl[ibl].vis[1] = numpy.imag(bvis.data['vis'][itime, i2, i1, vchan, pol])
                            ibl += 1
                            # Fill stats
                status = wtowers.fill_stats_func(wtvis)
                
                # Get dirty image for this frequency
                dirty = vis2dirty(grid_size, theta, wtvis, wtkern=wtkern, subgrid_size=subgrid_size, margin=margin,
                                  winc=winc)
                
                # dirty = ng.ms2dirty(
                #    fuvw, freq[vchan:vchan+1], ms_1d, wgt_1d,
                #    npixdirty, npixdirty, pixsize, pixsize, epsilon, do_wstacking=do_wstacking,
                #    nthreads=nthreads, verbosity=verbosity)
                
                sumwt[ichan, pol] += numpy.sum(wgt[:, vchan, pol])
                im.data[ichan, pol] += dirty
        
        if normalize:
            im = normalize_sumwt(im, sumwt)
        
        return im, sumwt


except ImportError:
    import warnings
    
    warnings.warn('Cannot import wtowers, wt disabled', ImportWarning)
    
    
    def predict_wt(bvis: Union[BlockVisibility, Visibility], model: Image, **kwargs) -> \
            Union[BlockVisibility, Visibility]:
        log.error("Wtowers gridder not available")
        return bvis
    
    
    def invert_wt(bvis: BlockVisibility, model: Image, dopsf: bool = False, normalize: bool = True,
                  **kwargs) -> (Image, numpy.ndarray):
        log.error("Wtowers gridder not available")
        return model, None
