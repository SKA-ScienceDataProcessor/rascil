"""Simulation of the effect of non-isoplanatic pahse errore on MID observations

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

from rascil.data_models.parameters import rascil_path, rascil_data_path
from rascil.data_models.polarisation import PolarisationFrame
from rascil.processing_components import plot_azel, \
    plot_uvcoverage, find_pb_width_null, create_simulation_components, \
    convert_blockvisibility_to_visibility, \
    convert_visibility_to_blockvisibility, show_image, qa_image, export_image_to_fits, \
    create_image_from_visibility, advise_wide_field, \
    apply_beam_to_skycomponent, create_pb, import_image_from_fits
from rascil.workflows import invert_list_rsexecute_workflow, \
    sum_invert_results_rsexecute, weight_list_rsexecute_workflow, \
    create_atmospheric_errors_gaintable_rsexecute_workflow, \
    create_standard_mid_simulation_rsexecute_workflow, \
    calculate_selfcal_residual_from_gaintables_rsexecute_workflow,\
    sum_invert_results
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
        description='Distributed simulation of atmosphere induced errors for SKA-MID')
    par.add_argument('--context', type=str, default='singlesource',
                     help='Type of sky: s3sky or doublesource or singlesource or null')
    par.add_argument('--imaging_context', type=str, default='2d',
                     help='Type of imaging transforms to use: 2d or ng. '
                          'The latter requires that Nifty Gridder be installed')
    par.add_argument('--telescope', type=str, default='MID',
                     help='Telescope: MID or LOW')

    # Observation definition
    par.add_argument('--ra', type=float, default=+15.0,
                     help='Right ascension of target source (degrees)')
    par.add_argument('--declination', type=float, default=-45.0,
                     help='Declination  of target source (degrees)')
    par.add_argument('--frequency', type=float, default=1.36e9,
                     help='Frequency of observation (Hz)')
    par.add_argument('--rmax', type=float, default=1e5,
                     help='Maximum distance of dish from array centre (m)')
    par.add_argument('--band', type=str, default='B2', help="Band: B1, B2 or Ku")
    par.add_argument('--integration_time', type=float, default=600,
                     help='Duration of single integration (s)')
    par.add_argument('--time_range', type=float, nargs=2, default=[-6.0, 6.0],
                     help='Hour angle of observation (hours)')
    par.add_argument('--npixel', type=int, default=512,
                     help='Number of pixels in dirty image used for statistics')
    par.add_argument('--use_natural', type=str, default='False',
                     help='Use natural weighting?')
    par.add_argument('--pbradius', type=float, default=2.0,
                     help='Radius of s3sky sources to include (in HWHM)')
    par.add_argument('--pbtype', type=str, default='MID',
                     help='Primary beam model: MID, MID_GAUSS, MID_FEKO_B1, MID_FEKO_B2, MID_FEKO_Ku')
    par.add_argument('--flux_limit', type=float, default=1.0,
                     help='Flux limit in selecting sources for s3sky (Jy)')
    par.add_argument('--make_residual', type=str, default='True',
                     help='Make residual image?')
    par.add_argument('--selfcal', type=str, default='True',
                     help='Selfcalibrate?')
    par.add_argument('--zerow', type=str, default='True',
                     help='Set w to zero? Use together with --imaging_context 2d')
    # Control parameters
    par.add_argument('--show', type=str, default='False', help='Show images?')
    par.add_argument('--export_images', type=str, default='False',
                     help='Export images in fits format?')
    par.add_argument('--use_agg', type=str, default="True",
                     help='Use Agg matplotlib backend?')
    default_shared_path = rascil_data_path("configurations")
    par.add_argument('--shared_directory', type=str, default=default_shared_path,
                     help='Location of configuration files (default is RASCIL data/configurations)')
    # Dask parameters; matched to P3
    par.add_argument('--serial', type=str, default='False',
                     help='Use serial processing (very slow)')
    par.add_argument('--nthreads', type=int, default=4,
                     help='Number of threads in Nifty Gridder')
    par.add_argument('--memory', type=int, default=64,
                     help='Memory per Dask worker (GB)')
    par.add_argument('--nworkers', type=int, default=16, help='Number of Dask workers')
    par.add_argument('--dask_worker_space', type=str, default=".",
                     help='Location for dask worker files')
    # Simulation parameters
    par.add_argument('--time_chunk', type=float, default=1800.0,
                     help="Chunking of time for simulation (s)")
    par.add_argument('--type_atmosphere', type=str, default='troposphere',
                     help="Type: troposphere or ionosphere")
    par.add_argument('--screen', type=str, default=None,
                     help="Screen as fits file")
    par.add_argument('--height', type=float, default=3e3,
                     help="Nominal height of screen")
    par.add_argument('--r0', nargs='+', type=float, default=[5e3,15e3,45e3],
                     help="List of r0 values to test (meters)")
    return par


if __name__ == '__main__':

    start_epoch = time.asctime()
    log.info(
        "\nDistributed simulation of atmosphere induced errors\nStarted at %s\n" % start_epoch)

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
    telescope = args.telescope
    ra = args.ra
    declination = args.declination
    use_natural = args.use_natural == "True"
    export_images = args.export_images == "True"
    integration_time = args.integration_time
    time_range = args.time_range
    time_chunk = args.time_chunk
    pbtype = args.pbtype
    pbradius = args.pbradius
    rmax = args.rmax
    flux_limit = args.flux_limit
    npixel = args.npixel
    shared_directory = args.shared_directory
    type_atmosphere = args.type_atmosphere
    height = args.height
    make_residual = args.make_residual == "True"
    selfcal = args.selfcal == "True"
    zerow = args.zerow == "True"

    screen = args.screen
    screen_image = import_image_from_fits(screen)
    # Simulation specific parameters

    show = args.show == 'True'
    context = args.context
    nworkers = args.nworkers
    threads_per_worker = 1
    nthreads = args.nthreads
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
            print("{nworkers} workers each with {memory}GB, worker files in {local_dir}".format(nworkers=nworkers,
                                                                       memory=memory,
                                                                       local_dir=args.dask_worker_space))
            client = get_dask_client(n_workers=nworkers,
                                     memory_limit=memory * 1024 * 1024 * 1024,
                                     threads_per_worker=threads_per_worker,
                                     local_dir=args.dask_worker_space)
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

    frequency = numpy.array(frequency)

    channel_bandwidth = [1e7]
    phasecentre = SkyCoord(ra=ra * u.deg, dec=declination * u.deg, frame='icrs',
                           equinox='J2000')

    bvis_graph = create_standard_mid_simulation_rsexecute_workflow(band, rmax,
                                                                   phasecentre,
                                                                   time_range, time_chunk,
                                                                   integration_time,
                                                                   shared_directory,
                                                                   zerow=zerow)
    future_bvis_list = rsexecute.persist(bvis_graph)
    bvis_list0 = rsexecute.compute(bvis_graph[0], sync=True)
    nchunks = len(bvis_graph)
    memory_use['bvis_list'] = nchunks * bvis_list0.size()

    memory_use['vis_list'] = nchunks * bvis_list0.size()

    # We need the HWHM of the primary beam, and the location of the nulls
    HWHM_deg, null_az_deg, null_el_deg = find_pb_width_null(pbtype, frequency)

    HWHM = HWHM_deg * numpy.pi / 180.0

    FOV_deg = 4.0 * 1.36e9 / frequency[0]
    print('%s: HWHM beam = %g deg' % (pbtype, HWHM_deg))

    advice_list = rsexecute.execute(advise_wide_field)(bvis_list0,
                                                       guard_band_image=1.0,
                                                       delA=0.02)
    advice = rsexecute.compute(advice_list, sync=True)
    pb_npixel = 1024
    d2r = numpy.pi / 180.0
    pb_cellsize = d2r * FOV_deg / pb_npixel
    cellsize = advice['cellsize']

    if show:
        bvis_list = rsexecute.compute(bvis_graph, sync=True)
        plot_azel(bvis_list, title=basename)
        plt.savefig('azel.png')
        plt.show(block=False)

        vis_list = rsexecute.compute(bvis_graph, sync=True)
        plot_uvcoverage(vis_list, title=basename)
        plt.savefig('uvcoverage.png')
        plt.show(block=False)

   # Now construct the components
    print("Constructing components")
    original_components, offset_direction = create_simulation_components(context,
                                                                         phasecentre,
                                                                         frequency,
                                                                         pbtype,
                                                                         [0.0, 0.0],
                                                                         flux_limit,
                                                                         pbradius * HWHM,
                                                                         pb_npixel,
                                                                         pb_cellsize,
                                                                         fov=10)

    scenarios = args.r0
    print("Testing r0 values: {scenarios}".format(scenarios=scenarios))

    # Estimate resource usage
    nants = len(bvis_list0.configuration.names)
    ntimes = len(bvis_list0.time)
    nbaselines = nants * (nants - 1) // 2

    memory_use['model_list'] = 8 * npixel * npixel * len(frequency) * len(
        original_components) / 1024 / 1024 / 1024
    memory_use['vp_list'] = 16 * npixel * npixel * len(
        frequency) * nchunks / 1024 / 1024 / 1024
    print("Memory use (GB)")
    pp.pprint(memory_use)
    total_memory_use = numpy.sum([memory_use[key] for key in memory_use.keys()])

    print("Summary of processing:")
    print("    There are %d workers" % nworkers)
    print("    There are %d separate visibility time chunks being processed" % len(
        future_bvis_list))
    print("    The integration time within each chunk is %.1f (s)" % integration_time)
    print("    There are a total of %d integrations per chunk" % ntimes)
    print("    There are %d baselines" % nbaselines)
    print("    There are %d components" % len(original_components))
    print("    %d scenario(s) will be tested" % len(scenarios))
    ntotal = ntimes * nbaselines * len(original_components) * len(scenarios)
    print("    Total processing %g times-baselines-components-scenarios" % ntotal)
    print("    Approximate total memory use for data = %.3f GB" % total_memory_use)
    if not serial:
        nworkers = len(rsexecute.client.scheduler_info()['workers'])
        print("    Using %s Dask workers" % nworkers)

    # Uniform weighting
    psf_list = [
        rsexecute.execute(create_image_from_visibility)(v, npixel=npixel,
                                                        frequency=frequency,
                                                        nchan=nfreqwin, cellsize=cellsize,
                                                        phasecentre=phasecentre,
                                                        polarisation_frame=PolarisationFrame(
                                                            "stokesI"))
        for v in future_bvis_list]
    psf_list = rsexecute.compute(psf_list, sync=True)
    future_psf_list = rsexecute.scatter(psf_list)
    del psf_list

    if use_natural:
        print("Using natural weighting")
    else:
        print("Using uniform weighting")

        vis_list = [rsexecute.execute(convert_blockvisibility_to_visibility)(bvis)
                     for bvis in future_bvis_list]
        vis_list = weight_list_rsexecute_workflow(vis_list, future_psf_list)
        bvis_list = [rsexecute.execute(convert_visibility_to_blockvisibility)(vis)
                     for vis in vis_list]
        bvis_list = rsexecute.compute(bvis_list, sync=True)
        future_bvis_list = rsexecute.scatter(bvis_list)
        del bvis_list

    dopsf = False
    if dopsf:
        print("Calculating PSF")
        imaging_context = args.imaging_context
        if imaging_context == "ng":
            psf_list = invert_list_rsexecute_workflow(future_bvis_list, future_psf_list,
                                                      args.imaging_context, dopsf=True,
                                                      verbosity=0, nthreads=nthreads,
                                                      do_wstacking=not zerow)
        else:
            psf_list = invert_list_rsexecute_workflow(future_bvis_list, future_psf_list,
                                                      args.imaging_context, dopsf=True,
                                                      verbosity=0, nthreads=nthreads,
                                                      do_wstacking=not zerow)

        psf_list = rsexecute.compute(psf_list, sync=True)
        psf, sumwt = sum_invert_results(psf_list)
        print("PSF sumwt ", sumwt)
        print(qa_image(psf, context='PSF'))

        if export_images:
            export_image_to_fits(psf, 'PSF_rascil.fits')
        if show:
            show_image(psf, cm='gray_r', title='%s PSF' % basename, vmin=-0.01, vmax=0.1)
            plt.savefig('PSF_rascil.png')
            plt.show(block=False)
        del psf_list
        del future_psf_list

    if context == "s3sky":
        pb_list = [
            rsexecute.execute(create_image_from_visibility)(bv, npixel=pb_npixel,
                                                            frequency=frequency,
                                                            nchan=nfreqwin,
                                                            cellsize=pb_cellsize,
                                                            phasecentre=phasecentre,
                                                            override_cellsize=False,
                                                            polarisation_frame=PolarisationFrame(
                                                                "stokesI"))
            for bv in future_bvis_list]
        pb = rsexecute.compute(pb_list[0], sync=True)
        del pb_list

        print("There are " + str(
            len(original_components)) + " components before pb application")

        pb = create_pb(pb, pbtype, use_local=False)
        if show:
            show_image(pb, cm='gray_r', title='%s Primary beam' % basename,
                       components=original_components)
            plt.savefig('PB.png')
            plt.show(block=False)

        original_components = apply_beam_to_skycomponent(original_components, pb)
        print("There are " + str(
            len(original_components)) + " components after pb application")
        assert len(original_components) > 0, "No components after applying primary beam"

    # Make one image per visibility
    future_model_list = [
        rsexecute.execute(create_image_from_visibility)(fvis, npixel=npixel,
                                                        frequency=frequency,
                                                        nchan=nfreqwin, cellsize=cellsize,
                                                        phasecentre=phasecentre,
                                                        polarisation_frame=PolarisationFrame(
                                                            "stokesI"))
        for fvis in future_bvis_list]

    filename = seqfile.findNextFile(
        prefix='atmospheric_simulation_%s_' % socket.gethostname(),
        suffix='.csv')
    print('Saving results to %s' % filename)
    plotfile = seqfile.findNextFile(
        prefix='atmospheric_simulation_%s_' % socket.gethostname(),
        suffix='.png')

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
        result['plotfile'] = plotfile
        result['hostname'] = socket.gethostname()
        result['epoch'] = epoch
        result['basename'] = basename
        result['nworkers'] = nworkers
        result['npixel'] = npixel
        result['pb_npixel'] = pb_npixel
        result['flux_limit'] = flux_limit
        result['pbtype'] = pbtype
        result['ra'] = ra
        result['declination'] = declination
        result['use_natural'] = use_natural
        result['integration_time'] = integration_time
        result['ntotal'] = ntotal
        result['r0'] = scenario
        result['type_atmosphere'] = type_atmosphere
        result['band'] = band
        result['frequency'] = frequency
        result['height'] = height
        result['screen'] = screen
        result['make_residual'] = make_residual
        result['selfcal'] = selfcal

        a2r = numpy.pi / (3600.0 * 180.0)

        rsexecute.init_statistics()

        # Calculate gaintables from the screen_image
        no_error_gtl, error_gtl = \
            create_atmospheric_errors_gaintable_rsexecute_workflow(future_bvis_list,
                                                                   original_components,
                                                                   r0=scenario,
                                                                   screen=screen_image,
                                                                   height=height,
                                                                   type_atmosphere=type_atmosphere,
                                                                   show=show,
                                                                   basename=basename)

        # Now make all the residual images using these gaintables
        vis_comp_chunk_dirty_list = \
            calculate_selfcal_residual_from_gaintables_rsexecute_workflow(
                future_bvis_list,
                original_components,
                future_model_list,
                no_error_gtl, error_gtl,
                residual=make_residual,
                selfcal=selfcal,
                context=args.imaging_context,
                nthreads=nthreads,
                do_wstacking=not zerow)

        # Add the resulting images
        error_dirty_list = sum_invert_results_rsexecute(vis_comp_chunk_dirty_list)

        # Actually compute the graph assembled above
        error_dirty, sumwt = rsexecute.compute(error_dirty_list, sync=True)
        print("Dirty image sumwt", sumwt)
        del error_dirty_list, no_error_gtl, error_gtl, vis_comp_chunk_dirty_list

        if export_images:
            export_image_to_fits(error_dirty,
                                 "residual_r0{scenario}.fits".format(scenario=scenario))

        if show:
            show_image(error_dirty, cm='gray_r',
                       title='Residual r0{scenario}'.format(scenario=scenario),
                       components=original_components)
            plt.savefig('residual_r0{scenario}_image.png'.format(scenario=scenario))
            plt.show(block=False)

        qa = qa_image(error_dirty, context=scenario)
        print(qa)
        for field in ['maxabs', 'rms', 'medianabs']:
            result["onsource_" + field] = qa.data[field]

        if dopsf:
            qa_psf = qa_image(psf)
            _, _, ny, nx = psf.shape
            for field in ['maxabs', 'rms', 'medianabs']:
                result["psf_" + field] = qa_psf.data[field]

        result['elapsed_time'] = time.time() - time_started
        print('Elapsed time = %.1f (s)' % result['elapsed_time'])

        results.append(result)

    pp.pprint(results)

    print("Total processing %g times-baselines-components-scenarios" % ntotal)
    processing_rate = ntotal / (nworkers * (time.time() - time_started))
    print("Processing rate of time-baseline-component-scenario = %g per worker-second"
          % processing_rate)

    rsexecute.save_statistics(name='atmospheric_simulation')

    for result in results:
        result["processing_rate"] = processing_rate

    with open(filename, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results[0].keys(), delimiter=',',
                                quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
        csvfile.close()

    title = '%s, %.3f GHz, %d times %s \n%s %s %s' % \
            (context, frequency[0] * 1e-9, ntimes, type_atmosphere, socket.gethostname(),
             epoch,
             basename)
    bar_width = 0.30
    opacity = 0.8

    plt.clf()
    index = numpy.arange(len(scenarios))
    fig, ax = plt.subplots()
    colors = ['b', 'r', 'g', 'y']
    for ifield, field in enumerate(['onsource_maxabs', 'onsource_rms',
                                    'onsource_medianabs']):
        plt.bar(index + ifield * bar_width,
                [1e6 * result[field] for result in results],
                bar_width, label=field, color=colors[ifield],
                alpha=opacity)
        ax.set_yscale('log')

    plt.xlabel('r0 (meters)')
    plt.ylabel('Error (uJy)')
    plt.xticks(numpy.arange(len(scenarios)) + 1.0 * bar_width, scenarios)
    plt.title(title)
    plt.legend(fontsize='x-small')
    print('Saving plot to %s' % plotfile)

    plt.tight_layout()
    plt.savefig(plotfile)
    plt.show(block=False)

    log.info("\nDistributed simulation of atmospheric errors for SKA-MID")
    log.info("Started at  %s" % start_epoch)
    log.info("Finished at %s" % time.asctime())
