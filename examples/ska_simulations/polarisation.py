"""Simulation of the effect of uncorrected beam polarisation on MID observations

This measures the change in a dirty image induced by various errors:
    - The sky can be a point source at the half power point or a realistic sky constructed from S3-SEX catalog.
    - The observation is by MID over a range of hour angles
    - Processing can be divided into chunks of time (default 1800s)
    - Dask is used to distribute the processing over a number of workers.
    - Various plots are produced, The primary output is a csv file containing information about the statistics of
    the residual images.

"""
import csv
import logging
import os
import socket
import sys
import time

import numpy
import seqfile
from astropy import units as u
from astropy.coordinates import SkyCoord

from rascil.data_models.parameters import rascil_path
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components import show_image, qa_image, export_image_to_fits, \
    create_vp, copy_image, \
    create_image_from_visibility, advise_wide_field, plot_azel, \
    plot_uvcoverage, find_pb_width_null, create_simulation_components
from rascil.workflows import invert_list_rsexecute_workflow, sum_invert_results_rsexecute, \
    weight_list_rsexecute_workflow, calculate_residual_from_gaintables_rsexecute_workflow, \
    create_polarisation_gaintable_rsexecute_workflow, \
    create_standard_mid_simulation_rsexecute_workflow, \
    sum_invert_results
from rascil.processing_components.imaging.ng import invert_ng
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute, \
    get_dask_client

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

import pprint

pp = pprint.PrettyPrinter()


def cli_parser():
    # Get command line inputs
    import argparse
    
    par = argparse.ArgumentParser(
        description='Distributed simulation of uncorrected beam polarisation for SKA-MID')
    par.add_argument('--context', type=str, default='singlesource',
                     help='Type of sky: s3sky or singlesource or null')
    par.add_argument('--imaging_context', type=str, default='2d',
                     help='Type of imaging transforms to use: 2d, wstack, or ng')
    
    # Observation definition
    par.add_argument('--ra', type=float, default=+15.0,
                     help='Right ascension of target source (degrees)')
    par.add_argument('--declination', type=float, default=-45.0,
                     help='Declination  of target source (degrees)')
    par.add_argument('--frequency', type=float, default=1.36e9,
                     help='Frequency of observation (Hz)')
    par.add_argument('--rmax', type=float, default=1e4,
                     help='Maximum distance of dish from array centre (m)')
    par.add_argument('--band', type=str, default='B2', help="Band: B1, B2 or Ku")
    par.add_argument('--integration_time', type=float, default=60,
                     help='Duration of single integration (s)')
    par.add_argument('--time_range', type=float, nargs=2, default=[-4.0, 4.0],
                     help='Hour angle of observation (hours)')
    par.add_argument('--npixel', type=int, default=1024,
                     help='Number of pixels in dirty image used for statistics')
    par.add_argument('--use_natural', type=str, default='False',
                     help='Use natural weighting?')
    par.add_argument('--offset_dir', type=float, nargs=2, default=[1.0, 0.0],
                     help='Multipliers for null offset')
    par.add_argument('--pbradius', type=float, default=2.0,
                     help='Radius of s3sky sources to include (in HWHM)')
    par.add_argument('--pbtype', type=str, default='MID',
                     help='Primary beam model: MID, MID_GAUSS, MID_FEKO_B1, MID_FEKO_B2, MID_FEKO_Ku')
    par.add_argument('--flux_limit', type=float, default=1.0,
                     help='Flux limit in selecting sources for s3sky (Jy)')
    # Control parameters
    par.add_argument('--show', type=str, default='True', help='Show images?')
    par.add_argument('--export_images', type=str, default='True',
                     help='Export images in fits format?')
    par.add_argument('--use_agg', type=str, default="True",
                     help='Use Agg matplotlib backend?')
    par.add_argument('--use_radec', type=str, default="False",
                     help='Calculate primary beams in RADEC?')
    default_shared_path = rascil_path("data/configurations")
    par.add_argument('--shared_directory', type=str, default=default_shared_path,
                     help='Location of configuration files (default is RASCIL data/configurations)')
    # Dask parameters; matched to P3
    par.add_argument('--serial', type=str, default='False',
                     help='Use serial processing (very slow)')
    par.add_argument('--nthreads', type=int, default=1,
                     help='Number of threads per Dask worker')
    par.add_argument('--memory', type=int, default=64,
                     help='Memory per Dask worker (GB)')
    par.add_argument('--nworkers', type=int, default=16, help='Number of Dask workers')
    # Simulation parameters
    par.add_argument('--time_chunk', type=float, default=1800.0,
                     help="Time for a chunk (s)")
    par.add_argument('--elevation_sampling', type=float, default=1.0,
                     help='Elevation sampling 1 deg or coarser (deg)')
    return par


if __name__ == '__main__':
    
    start_epoch = time.asctime()
    log.info(
        "\nDistributed simulation of unmodelled dish beam polarisation for SKA-MID\nStarted at %s\n" % start_epoch)
    
    memory_use = dict()
    
    # Get command line inputs
    parser = cli_parser()
    
    args = parser.parse_args()
    pp.pprint(vars(args))
    
    use_agg = args.use_agg == "True"
    if use_agg:
        import matplotlib as mpl
        
        mpl.use('Agg')
    from matplotlib import pyplot as plt
    
    band = args.band
    ra = args.ra
    declination = args.declination
    use_radec = args.use_radec == "True"
    use_natural = args.use_natural == "True"
    export_images = args.export_images == "True"
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
    
    # Simulation specific parameters
    elevation_sampling = args.elevation_sampling
    
    show = args.show == 'True'
    context = args.context
    nworkers = args.nworkers
    threads_per_worker = args.nthreads
    memory = args.memory
    serial = args.serial == "True"
    
    basename = os.path.basename(os.getcwd())
    
    if serial:
        print("Will use serial processing")
        use_serial_invert = True
        use_serial_predict = True
        rsexecute.set_client(use_dask=False)
        print(rsexecute.client)
        nworkers = 1
    else:
        print("Will use dask processing")
        if nworkers > 0:
            client = get_dask_client(n_workers=nworkers,
                                     memory_limit=memory * 1024 * 1024 * 1024,
                                     threads_per_worker=threads_per_worker)
            rsexecute.set_client(client=client)
        else:
            client = get_dask_client()
            rsexecute.set_client(client=client)
        
        actualnworkers = len(rsexecute.client.scheduler_info()['workers'])
        nworkers = actualnworkers
        print("Using %s Dask workers" % nworkers)
    
    time_started = time.time()
    
    # Set up details of simulated observation
    nfreqwin = 1
    diameter = 15.0
    if band == 'B1':
        frequency = [0.765e9]
    elif band == 'B2':
        frequency = [1.36e9]
    elif band == 'Ku':
        frequency = [12.179e9]
    else:
        raise ValueError("Unknown band %s" % band)
    
    channel_bandwidth = [1e7]
    phasecentre = SkyCoord(ra=ra * u.deg, dec=declination * u.deg, frame='icrs',
                           equinox='J2000')
    
    bvis_graph = create_standard_mid_simulation_rsexecute_workflow(band, rmax, phasecentre,
                                                                   time_range, time_chunk,
                                                                   integration_time,
                                                                   shared_directory=shared_directory,
                                                                   polarisation_frame=PolarisationFrame("linear"),
                                                                   zerow=args.imaging_context=="2d")
    bvis_graph = rsexecute.persist(bvis_graph)
    
    # We need the HWHM of the primary beam, and the location of the nulls
    HWHM_deg, null_az_deg, null_el_deg = find_pb_width_null(pbtype, frequency)
    
    HWHM = HWHM_deg * numpy.pi / 180.0
    
    FOV_deg = 8.0 * 1.36e9 / frequency[0]
    print('%s: HWHM beam = %g deg' % (pbtype, HWHM_deg))
    
    advice_list = rsexecute.execute(advise_wide_field)(bvis_graph[0], guard_band_image=1.0, delA=0.02, verbose=False)
    advice = rsexecute.compute(advice_list, sync=True)
    guard_band_image = npixel / advice['npixels']
    advice_list = rsexecute.execute(advise_wide_field)(bvis_graph[0], guard_band_image=guard_band_image, delA=0.02)
    advice = rsexecute.compute(advice_list, sync=True)
    vis_slices = advice['vis_slices']
    if args.imaging_context == "ng":
        vis_slices = 1

    pb_npixel = 1024
    d2r = numpy.pi / 180.0
    pb_cellsize = d2r * FOV_deg / pb_npixel
    cellsize = advice['cellsize']
    
    if show:
        vis_list = rsexecute.compute(bvis_graph, sync=True)
        plot_uvcoverage(vis_list, title=basename)
        plt.savefig('uvcoverage.png')
        plt.show(block=False)
        
        bvis_list = rsexecute.compute(bvis_graph, sync=True)
        plot_azel(bvis_list, title=basename)
        plt.savefig('azel.png')
        plt.show(block=False)
    
    # Now construct the components
    original_components, offset_direction = create_simulation_components(context,
                                                                         phasecentre,
                                                                         frequency,
                                                                         pbtype,
                                                                         offset_dir,
                                                                         flux_limit,
                                                                         pbradius * HWHM,
                                                                         pb_npixel,
                                                                         pb_cellsize)
    
    for comp in original_components:
        comp.polarisation_frame = PolarisationFrame('stokesIQUV')
        comp_flux = numpy.zeros([nfreqwin, 4])
        for chan in range(nfreqwin):
            comp_flux[chan, :] = numpy.array([[comp.flux[0, 0], 0.0, 0.0, 0.0]])
        comp.flux = comp_flux
    
    scenarios = ['']
    
    if rsexecute.using_dask:
        nworkers = len(rsexecute.client.scheduler_info()['workers'])
        print("    Using %s Dask workers" % nworkers)
    else:
        nworkers = 0
    
    # Uniform weighting
    psf_list = [rsexecute.execute(create_image_from_visibility)(v, npixel=npixel,
                                                                nchan=1,
                                                                cellsize=cellsize,
                                                                phasecentre=phasecentre,
                                                                polarisation_frame=PolarisationFrame("stokesIQUV"))
                for v in bvis_graph]
    future_psf_list = rsexecute.persist(psf_list)
    del psf_list
    
    if use_natural:
        print("Using natural weighting")
    else:
        print("Using uniform weighting")
        bvis_graph = weight_list_rsexecute_workflow(bvis_graph, future_psf_list)
        bvis_graph = rsexecute.persist(bvis_graph)
    
    print("Inverting to get PSF")
    psf_list = invert_list_rsexecute_workflow(bvis_graph, future_psf_list, args.imaging_context, vis_slices=vis_slices,
                                              dopsf=True, do_wstacking=False)
    psf_list = rsexecute.compute(psf_list, sync=True)
    psf, sumwt = sum_invert_results(psf_list)
    print("PSF sumwt ", sumwt)
    if export_images:
        export_image_to_fits(psf, 'PSF_rascil.fits')
    if show:
        show_image(psf, cm='gray_r', title='%s PSF' % basename, vmin=-0.01, vmax=0.1)
        plt.savefig('PSF_rascil.png')
        plt.show(block=False)
    del psf_list
    del future_psf_list
    
    # ### Calculate the voltage pattern without errors
    vp_list = [rsexecute.execute(create_image_from_visibility)(bv, npixel=pb_npixel,
                                                               frequency=frequency,
                                                               nchan=1,
                                                               cellsize=pb_cellsize,
                                                               phasecentre=phasecentre,
                                                               override_cellsize=False)
               for bv in bvis_graph]
    print("Constructing voltage pattern")
    vp_list = [rsexecute.execute(create_vp)(vp, pbtype, pointingcentre=phasecentre,
                                            use_local=not use_radec)
               for vp in vp_list]
    future_vp_list = rsexecute.persist(vp_list)
    del vp_list
    
    future_model_list = [
        rsexecute.execute(create_image_from_visibility)(bvis, npixel=npixel,
                                                        frequency=frequency,
                                                        nchan=1, cellsize=cellsize,
                                                        phasecentre=offset_direction,
                                                        polarisation_frame=PolarisationFrame("stokesIQUV"))
        for bvis in bvis_graph]
    
    filename = seqfile.findNextFile(
        prefix='polarisation_simulation_%s_' % socket.gethostname(), suffix='.csv')
    print('Saving results to %s' % filename)
    
    epoch = time.strftime("%Y-%m-%d %H:%M:%S")
    
    time_started = time.time()
    
    # Now loop over all scenarios
    print("")
    print("***** Starting loop over scenarios ******")
    print("")
    results = []
    
    for scenario in scenarios:
        
        result = dict()
        result['context'] = context
        result['nb_name'] = sys.argv[0]
        result['hostname'] = socket.gethostname()
        result['epoch'] = epoch
        result['basename'] = basename
        result['npixel'] = npixel
        result['pb_npixel'] = pb_npixel
        result['flux_limit'] = flux_limit
        result['pbtype'] = pbtype
        result['offset_dir'] = offset_dir
        result['ra'] = ra
        result['declination'] = declination
        result['use_radec'] = use_radec
        result['use_natural'] = use_natural
        result['integration_time'] = integration_time
        result['se'] = scenario
        result['band'] = band
        result['frequency'] = frequency
        
        a2r = numpy.pi / (3600.0 * 180.0)
        
        rsexecute.init_statistics()
        
        no_error_gtl, error_gtl = \
            create_polarisation_gaintable_rsexecute_workflow(band, bvis_graph,
                                                             original_components,
                                                             use_radec=use_radec,
                                                             show=show,
                                                             basename=basename)
        
        # Now make all the residual images
        vis_comp_chunk_dirty_list = \
            calculate_residual_from_gaintables_rsexecute_workflow(bvis_graph,
                                                                  original_components,
                                                                  future_model_list,
                                                                  no_error_gtl, error_gtl,
                                                                  context=args.imaging_context,
                                                                  vis_slices=vis_slices,
                                                                  do_wstacking=False)
        
        # Add the resulting images
        error_dirty_list = sum_invert_results_rsexecute(vis_comp_chunk_dirty_list)
        
        # Actually compute the graph assembled above
        error_dirty, sumwt = rsexecute.compute(error_dirty_list, sync=True)
        print("Dirty image sumwt", sumwt)
        del error_dirty_list
        
        if export_images:
            export_image_to_fits(error_dirty, 'residual_rascil.fits')
        
        if show:
            for ipol, pol in enumerate(["I", "Q", "U", "V"]):
                show_image(error_dirty, cm='gray_r', pol=ipol)
                plt.title("Stokes " + pol)
                plt.savefig('residual_image_Stokes_' + pol + '.png')
            plt.show(block=False)

        _, _, ny, nx = error_dirty.shape
        for ipol, pol in enumerate(["I", "Q", "U", "V"]):
            polimage = copy_image(error_dirty)
            polimage.data = polimage.data[:,ipol,...][:, numpy.newaxis, ...]
            qa = qa_image(polimage, context="Stokes " + pol)
            for field in ['maxabs', 'rms', 'medianabs']:
                result["onsource_" + pol + "_" + field] = qa.data[field]
        
        qa_psf = qa_image(psf)
        _, _, ny, nx = psf.shape
        for field in ['maxabs', 'rms', 'medianabs']:
            result["psf_" + field] = qa_psf.data[field]
        
        result['elapsed_time'] = time.time() - time_started
        print('Elapsed time = %.1f (s)' % result['elapsed_time'])
        
        results.append(result)
    
    pp.pprint(results)
    
    rsexecute.save_statistics(name='polarisation')
    
    with open(filename, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results[0].keys(), delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
        csvfile.close()
    
    rsexecute.close()
    
    log.info("\nDistributed simulation of unmodelled beam polarisation for SKA-MID")
    log.info("Started at  %s" % start_epoch)
    log.info("Finished at %s" % time.asctime())
