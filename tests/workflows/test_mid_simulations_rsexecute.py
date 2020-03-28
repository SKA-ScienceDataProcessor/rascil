"""Simulation of the effect of errors on MID observations

This measures the change in a dirty imagethe induced by various errors:
    - The sky can be a point source at the half power point or a realistic sky constructed from S3-SEX catalog.
    - The observation is by MID over a range of hour angles
    - Processing can be divided into chunks of time (default 1800s)
    - Dask is used to distribute the processing over a number of workers.
    - Various plots are produced, The primary output is a csv file containing information about the statistics of
    the residual images.

"""
import logging
import os
import sys
import unittest

import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models.parameters import rascil_path, rascil_data_path
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components.image.operations import qa_image, export_image_to_fits
from rascil.processing_components.imaging.base import create_image_from_visibility, advise_wide_field
from rascil.processing_components.imaging.primary_beams import create_vp
from rascil.processing_components.simulation.simulation_helpers import find_pb_width_null, create_simulation_components
from rascil.processing_components.visibility.coalesce import convert_blockvisibility_to_visibility
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute
from rascil.workflows.rsexecute.imaging.imaging_rsexecute import sum_invert_results_rsexecute, \
    weight_list_rsexecute_workflow
from rascil.workflows.rsexecute.simulation.simulation_rsexecute import \
    calculate_residual_from_gaintables_rsexecute_workflow, create_surface_errors_gaintable_rsexecute_workflow, \
    create_pointing_errors_gaintable_rsexecute_workflow, create_standard_mid_simulation_rsexecute_workflow, \
    create_polarisation_gaintable_rsexecute_workflow

results_dir = rascil_path('test_results')

log = logging.getLogger()
log.setLevel(logging.WARNING)
log.addHandler(logging.StreamHandler(sys.stdout))
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)


class TestPointingSimulation(unittest.TestCase):
    
    def setUp(self) -> None:
        rsexecute.set_client(use_dask=True, n_workers=4, threads_per_worker=1)
        self.persist = True
    
    def simulation(self, args, time_series='wind', band='B2',
                   image_polarisation_frame=PolarisationFrame("stokesI"),
                   vis_polarisation_frame=PolarisationFrame("stokesI")):
        
        context = args.context
        ra = args.ra
        declination = args.declination
        use_radec = args.use_radec == "True"
        integration_time = args.integration_time
        time_range = args.time_range
        time_chunk = args.time_chunk
        offset_dir = args.offset_dir
        pbtype = args.pbtype
        pbradius = args.pbradius
        rmax = args.rmax
        flux_limit = args.flux_limit
        npixel = args.npixel
        shared_directory = args.shared_directory
        vp_directory = args.vp_directory
        
        # Simulation specific parameters
        global_pe = numpy.array(args.global_pe)
        static_pe = numpy.array(args.static_pe)
        dynamic_pe = args.dynamic_pe
        
        seed = args.seed
        basename = os.path.basename(os.getcwd())
        
        # Set up details of simulated observation
        nfreqwin = 1
        if band == 'B1':
            frequency = [0.765e9]
        elif band == 'B2':
            frequency = [1.36e9]
        elif band == 'Ku':
            frequency = [12.179e9]
        else:
            raise ValueError("Unknown band %s" % band)
        
        phasecentre = SkyCoord(ra=ra * u.deg, dec=declination * u.deg, frame='icrs', equinox='J2000')
        
        bvis_graph = create_standard_mid_simulation_rsexecute_workflow(band, rmax, phasecentre, time_range, time_chunk,
                                                                       integration_time,
                                                                       shared_directory,
                                                                       polarisation_frame=vis_polarisation_frame)
        future_bvis_list = rsexecute.persist(bvis_graph)
        
        # We need the HWHM of the primary beam, and the location of the nulls
        HWHM_deg, null_az_deg, null_el_deg = find_pb_width_null(pbtype, frequency)
        
        HWHM = HWHM_deg * numpy.pi / 180.0
        
        FOV_deg = 8.0 * 1.36e9 / frequency[0]
        
        advice_list = rsexecute.execute(advise_wide_field)(future_bvis_list[0], guard_band_image=1.0, delA=0.02,
                                                           verbose=False)
        advice = rsexecute.compute(advice_list, sync=True)
        pb_npixel = 1024
        d2r = numpy.pi / 180.0
        pb_cellsize = d2r * FOV_deg / pb_npixel
        cellsize = advice['cellsize']
        
        # Now construct the components
        original_components, offset_direction = create_simulation_components(context, phasecentre, frequency,
                                                                             pbtype, offset_dir, flux_limit,
                                                                             pbradius * HWHM, pb_npixel, pb_cellsize,
                                                                             polarisation_frame=image_polarisation_frame,
                                                                             filter_by_primary_beam=True)
        
        print("There are {} components".format(len(original_components)))
        
        vp_list = [rsexecute.execute(create_image_from_visibility)(bv, npixel=pb_npixel, frequency=frequency,
                                                                   nchan=nfreqwin, cellsize=pb_cellsize,
                                                                   phasecentre=phasecentre,
                                                                   polarisation_frame=image_polarisation_frame,
                                                                   override_cellsize=False) for bv in future_bvis_list]
        vp_list = [rsexecute.execute(create_vp)(vp, pbtype, pointingcentre=phasecentre, use_local=not use_radec)
                   for vp in vp_list]
        future_vp_list = rsexecute.persist(vp_list)
        
        a2r = numpy.pi / (3600.0 * 1800)
        
        if time_series == '':
            # Random pointing errors
            global_pointing_error = global_pe
            static_pointing_error = static_pe
            pointing_error = dynamic_pe
            
            no_error_gtl, error_gtl = \
                create_pointing_errors_gaintable_rsexecute_workflow(future_bvis_list, original_components,
                                                                    sub_vp_list=future_vp_list,
                                                                    use_radec=use_radec,
                                                                    pointing_error=a2r * pointing_error,
                                                                    static_pointing_error=a2r * static_pointing_error,
                                                                    global_pointing_error=a2r * global_pointing_error,
                                                                    seed=seed,
                                                                    show=False, basename=basename)
        elif time_series == 'wind':
            # Wind-induced pointing errors
            no_error_gtl, error_gtl = \
                create_pointing_errors_gaintable_rsexecute_workflow(future_bvis_list, original_components,
                                                                    sub_vp_list=future_vp_list,
                                                                    use_radec=use_radec,
                                                                    time_series=time_series,
                                                                    time_series_type='precision',
                                                                    seed=seed,
                                                                    show=False, basename=basename)
        elif time_series == 'gravity':
            # Dish surface sag due to gravity
            no_error_gtl, error_gtl = \
                create_surface_errors_gaintable_rsexecute_workflow(band, future_bvis_list, original_components,
                                                                   vp_directory=vp_directory, use_radec=use_radec,
                                                                   show=False, basename=basename)
        elif time_series == 'polarisation':
            # Polarised beams
            no_error_gtl, error_gtl = \
                create_polarisation_gaintable_rsexecute_workflow(band, future_bvis_list, original_components,
                                                                 basename="Polarisation gain table", show=True)
        else:
            raise ValueError("Unknown type of error %s" % time_series)
        
        # Perform uniform weighting
        future_remodel_list = [rsexecute.execute(create_image_from_visibility)(v, npixel=npixel,
                                                                               frequency=frequency,
                                                                               nchan=nfreqwin, cellsize=cellsize,
                                                                               phasecentre=offset_direction,
                                                                               polarisation_frame=image_polarisation_frame)
                               for v in future_bvis_list]
        
        future_bvis_list = weight_list_rsexecute_workflow(future_bvis_list, future_remodel_list)
        
        # Now make all the residual images
        # Make one image per component
        future_model_list = [rsexecute.execute(create_image_from_visibility)(future_bvis_list[0], npixel=npixel,
                                                                             frequency=frequency,
                                                                             nchan=nfreqwin, cellsize=cellsize,
                                                                             phasecentre=offset_direction,
                                                                             polarisation_frame=image_polarisation_frame)
                             for i, _ in enumerate(original_components)]
        vis_comp_chunk_dirty_list = \
            calculate_residual_from_gaintables_rsexecute_workflow(future_bvis_list, original_components,
                                                                  future_model_list,
                                                                  no_error_gtl, error_gtl)
        
        # Add the resulting images
        error_dirty_list = sum_invert_results_rsexecute(vis_comp_chunk_dirty_list)
        
        # Actually compute the graph assembled above
        error_dirty, sumwt = rsexecute.compute(error_dirty_list, sync=True)
        
        return error_dirty, sumwt
    
    def get_args(self):
        
        # Get command line inputs
        import argparse
        
        parser = argparse.ArgumentParser(description='Simulate pointing errors')
        parser.add_argument('--context', type=str, default='s3sky', help='s3sky or singlesource or null')
        
        # Observation definition
        parser.add_argument('--ra', type=float, default=+15.0, help='Right ascension (degrees)')
        parser.add_argument('--declination', type=float, default=-45.0, help='Declination (degrees)')
        parser.add_argument('--rmax', type=float, default=2e3, help='Maximum distance of station from centre (m)')
        parser.add_argument('--band', type=str, default='B2', help="Band")
        parser.add_argument('--integration_time', type=float, default=600, help='Integration time (s)')
        parser.add_argument('--time_range', type=float, nargs=2, default=[-4.0, 4.0], help='Time range in hours')
        
        parser.add_argument('--npixel', type=int, default=1536, help='Number of pixels in image')
        parser.add_argument('--use_natural', type=str, default='True', help='Use natural weighting?')
        
        parser.add_argument('--snapshot', type=str, default='False', help='Do snapshot only?')
        parser.add_argument('--opposite', type=str, default='False',
                            help='Move source to opposite side of pointing centre')
        parser.add_argument('--offset_dir', type=float, nargs=2, default=[1.0, 0.0], help='Multipliers for null offset')
        parser.add_argument('--pbradius', type=float, default=1.0, help='Radius of sources to include (in HWHM)')
        parser.add_argument('--pbtype', type=str, default='MID', help='Primary beam model: MID or MID_GAUSS')
        parser.add_argument('--seed', type=int, default=18051955, help='Random number seed')
        parser.add_argument('--flux_limit', type=float, default=0.003, help='Flux limit (Jy)')
        
        # Control parameters
        parser.add_argument('--use_radec', type=str, default="False", help='Calculate in RADEC (false)?')
        parser.add_argument('--shared_directory', type=str, default=rascil_data_path('configurations'),
                            help='Location of configuration files')
        
        # Dask parameters
        parser.add_argument('--nnodes', type=int, default=1, help='Number of nodes')
        parser.add_argument('--nthreads', type=int, default=4, help='Number of threads')
        parser.add_argument('--memory', type=int, default=8, help='Memory per worker (GB)')
        parser.add_argument('--nworkers', type=int, default=8, help='Number of workers')
        parser.add_argument('--serial', type=str, default='False', help='Use serial processing?')
        
        # Simulation parameters
        parser.add_argument('--time_chunk', type=float, default=8*3600.0, help="Time for a chunk (s)")
        parser.add_argument('--time_series', type=str, default='wind', help="Type of time series")
        parser.add_argument('--global_pe', type=float, nargs=2, default=[0.0, 0.0], help='Global pointing error')
        parser.add_argument('--static_pe', type=float, nargs=2, default=[0.0, 0.0],
                            help='Multipliers for static errors')
        parser.add_argument('--dynamic_pe', type=float, default=1.0, help='Multiplier for dynamic errors')
        parser.add_argument('--pointing_file', type=str, default=None, help="Pointing file")
        parser.add_argument('--pointing_directory', type=str, default=rascil_data_path('models'),
                            help='Location of pointing files')
        parser.add_argument('--vp_directory', type=str, default=rascil_data_path('models/interpolated'),
                            help='Location of pointing files')
        
        args = parser.parse_args([])
        
        return args
    
    @unittest.skip("Needs reworking")
    def test_wind(self):
        
        error_dirty, sumwt = self.simulation(self.get_args(), 'wind')
        
        qa = qa_image(error_dirty)
        
        numpy.testing.assert_almost_equal(qa.data['max'], 1.2440965953105578e-06, 12)
        numpy.testing.assert_almost_equal(qa.data['min'], -3.836051633637655e-06, 12)
        numpy.testing.assert_almost_equal(qa.data['rms'], 5.840397284050296e-07, 12)
    
    @unittest.skip("Needs reworking")
    def test_random(self):
        
        error_dirty, sumwt = self.simulation(self.get_args(), '')
        
        qa = qa_image(error_dirty)
        
        numpy.testing.assert_almost_equal(qa.data['max'], 2.2055849698035616e-06, 12)
        numpy.testing.assert_almost_equal(qa.data['min'], -6.838117387793031e-07, 12)
        numpy.testing.assert_almost_equal(qa.data['rms'], 3.7224203394509413e-07, 12)
    
    def test_gravity(self):
        
        if os.path.isdir(rascil_path('models/interpolated')):
            error_dirty, sumwt = self.simulation(self.get_args(), 'gravity')
            
            qa = qa_image(error_dirty)
            
            numpy.testing.assert_almost_equal(qa.data['max'], 2.2055849698035616e-06, 12)
            numpy.testing.assert_almost_equal(qa.data['min'], -6.838117387793031e-07, 12)
            numpy.testing.assert_almost_equal(qa.data['rms'], 3.7224203394509413e-07, 12)
    
    def test_polarisation(self):
        
        error_dirty, sumwt = self.simulation(self.get_args(), 'polarisation',
                                             image_polarisation_frame=PolarisationFrame("stokesIQUV"),
                                             vis_polarisation_frame=PolarisationFrame("linear"))
        qa = qa_image(error_dirty)
        print(qa)
        
        if self.persist:
            export_image_to_fits(error_dirty, "{}/test_mid_simulation_polarisation.fits".format(results_dir))
