"""Unit tests for primary beam application with polarisation


"""

import logging
import os
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models.polarisation import PolarisationFrame, convert_pol_frame
from rascil.processing_components.image import export_image_to_fits
from rascil.processing_components.image.operations import qa_image
from rascil.processing_components.imaging import create_image_from_visibility, advise_wide_field
from rascil.processing_components.imaging.dft import dft_skycomponent_visibility, idft_visibility_skycomponent
from rascil.processing_components.imaging.primary_beams import create_vp
from rascil.processing_components.simulation import create_named_configuration, create_test_skycomponents_from_s3
from rascil.processing_components.skycomponent import apply_voltage_pattern_to_skycomponent, \
    filter_skycomponents_by_flux
from rascil.processing_components.visibility import create_blockvisibility, vis_timeslice_iter, \
    create_visibility_from_rows
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute
from rascil.workflows.rsexecute.pipelines import continuum_imaging_list_rsexecute_workflow
from rascil.workflows.rsexecute.imaging import weight_list_rsexecute_workflow

log = logging.getLogger('logger')

log.setLevel(logging.DEBUG)


class TestPrimaryBeamsPolGraph(unittest.TestCase):
    def setUp(self):
        rsexecute.set_client(use_dask=True, processes=False, threads_per_worker=1)
    
        from rascil.data_models.parameters import rascil_path
        self.dir = rascil_path('test_results')
        self.persist = os.getenv("RASCIL_PERSIST", True)

    def tearDown(self):
        rsexecute.close()

    def createVis(self, config='MID', dec=-35.0, rmax=1e2, freq=1.3e9):
        self.frequency = numpy.array([freq])
        self.channel_bandwidth = numpy.array([1e6])
        self.flux = numpy.array([[100.0, 60.0, -10.0, +1.0]])
        self.phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=dec * u.deg, frame='icrs',
                                    equinox='J2000')
        self.config = create_named_configuration(config, rmax=rmax)
        self.times = numpy.linspace(-300.0, 300.0, 3) * numpy.pi / 43200.0
        nants = self.config.xyz.shape[0]
        self.npixel = 512
        self.fov = 4
        self.cellsize = numpy.pi * self.fov / (self.npixel * 180.0)
        assert nants > 1
        assert len(self.config.names) == nants
        assert len(self.config.mount) == nants

    @unittest.skip("Too large for CI/CD")
    def test_apply_voltage_pattern_image_s3(self):
        self.createVis(rmax=1e3)
        telescope = 'MID_FEKO_B2'
        vpol = PolarisationFrame("linear")
        self.times = numpy.linspace(-4, +4, 8) * numpy.pi / 12.0
        bvis = create_blockvisibility(self.config, self.times, self.frequency,
                                      channel_bandwidth=self.channel_bandwidth,
                                      phasecentre=self.phasecentre, weight=1.0,
                                      polarisation_frame=vpol, zerow=True)
        cellsize = advise_wide_field(bvis)['cellsize']

        pbmodel = create_image_from_visibility(bvis, cellsize=self.cellsize, npixel=self.npixel,
                                               override_cellsize=False,
                                               polarisation_frame=PolarisationFrame("stokesIQUV"))
        vpbeam = create_vp(pbmodel, telescope=telescope, use_local=False)
        vpbeam.wcs.wcs.ctype[0] = 'RA---SIN'
        vpbeam.wcs.wcs.ctype[1] = 'DEC--SIN'
        vpbeam.wcs.wcs.crval[0] = pbmodel.wcs.wcs.crval[0]
        vpbeam.wcs.wcs.crval[1] = pbmodel.wcs.wcs.crval[1]

        s3_components = create_test_skycomponents_from_s3(flux_limit=0.1,
                                                          phasecentre=self.phasecentre,
                                                          frequency=self.frequency,
                                                          polarisation_frame=PolarisationFrame('stokesI'),
                                                          radius=1.5 * numpy.pi / 180.0)

        for comp in s3_components:
            comp.polarisation_frame = PolarisationFrame('stokesIQUV')
            comp.flux = numpy.array([[comp.flux[0, 0], 0.0, 0.0, 0.0]])

        s3_components = filter_skycomponents_by_flux(s3_components, 0.0, 10.0)

        from rascil.processing_components.image import show_image
        import matplotlib.pyplot as plt
        plt.clf()
        show_image(vpbeam, components=s3_components)
        plt.show(block=False)

        vpcomp = apply_voltage_pattern_to_skycomponent(s3_components, vpbeam)
        bvis.data['vis'][...] = 0.0 + 0.0j
        bvis = dft_skycomponent_visibility(bvis, vpcomp)

        rec_comp = idft_visibility_skycomponent(bvis, vpcomp)[0]

        stokes_comp = list()
        for comp in rec_comp:
            stokes_comp.append(convert_pol_frame(comp.flux[0], PolarisationFrame("linear"),
                                                 PolarisationFrame("stokesIQUV")))

        stokesI = numpy.abs(numpy.array([comp_flux[0] for comp_flux in stokes_comp]).real)
        stokesQ = numpy.abs(numpy.array([comp_flux[1] for comp_flux in stokes_comp]).real)
        stokesU = numpy.abs(numpy.array([comp_flux[2] for comp_flux in stokes_comp]).real)
        stokesV = numpy.abs(numpy.array([comp_flux[3] for comp_flux in stokes_comp]).real)
        plt.clf()
        plt.loglog(stokesI, stokesQ, '.', label='Q')
        plt.loglog(stokesI, stokesU, '.', label='U')
        plt.loglog(stokesI, stokesV, '.', label='V')
        plt.xlabel("Stokes Flux I (Jy)")
        plt.ylabel("Flux (Jy)")
        plt.legend()
        plt.savefig('%s/test_primary_beams_pol_rsexecute_stokes_errors.png' % self.dir)
        plt.show(block=False)

        split_times = False
        if split_times:
            bvis_list = list()
            for rows in vis_timeslice_iter(bvis, vis_slices=8):
                bvis_list.append(create_visibility_from_rows(bvis, rows))
        else:
            bvis_list = [bvis]

        bvis_list = rsexecute.scatter(bvis_list)

        model_list = \
            [rsexecute.execute(create_image_from_visibility, nout=1)(bv, cellsize=cellsize, npixel=4096,
                                                                     phasecentre=self.phasecentre,
                                                                     override_cellsize=False,
                                                                     polarisation_frame=PolarisationFrame("stokesIQUV"))
             for bv in bvis_list]

        model_list = rsexecute.persist(model_list)
        bvis_list = weight_list_rsexecute_workflow(bvis_list, model_list)

        continuum_imaging_list = \
            continuum_imaging_list_rsexecute_workflow(bvis_list, model_list,
                                                      context='2d',
                                                      algorithm='hogbom',
                                                      facets=1,
                                                      niter=1000,
                                                      fractional_threshold=0.1,
                                                      threshold=1e-4,
                                                      nmajor=5, gain=0.1,
                                                      deconvolve_facets=4,
                                                      deconvolve_overlap=32,
                                                      deconvolve_taper='tukey',
                                                      psf_support=64,
                                                      restore_facets=4, psfwidth=1.0)
        clean, residual, restored = rsexecute.compute(continuum_imaging_list, sync=True)
        centre = 0
        if self.persist:
            export_image_to_fits(clean[centre], '%s/test_primary_beams_pol_rsexecute_clean.fits' %
                                 self.dir)
            export_image_to_fits(residual[centre][0],
                                 '%s/test_primary_beams_pol_rsexecute_residual.fits' % self.dir)
            export_image_to_fits(restored[centre],
                                 '%s/test_primary_beams_pol_rsexecute_restored.fits' % self.dir)

        plt.clf()
        show_image(restored[centre])
        plt.show(block=False)

        qa = qa_image(restored[centre])
        assert numpy.abs(qa.data['max'] - 0.9953017707113947) < 1.0e-7, str(qa)
        assert numpy.abs(qa.data['min'] + 0.0036396480874570846) < 1.0e-7, str(qa)


if __name__ == '__main__':
    unittest.main()
