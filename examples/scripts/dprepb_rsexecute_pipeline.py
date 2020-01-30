"""
This executes a DPREPB pipeline: deconvolution of calibrated spectral line data.

"""

import argparse
import logging

from dask.distributed import Client

# These are the RASCIL functions we need
from rascil.data_models import PolarisationFrame, rascil_path
from rascil.processing_components import create_visibility_from_ms, \
    create_visibility_from_rows, append_visibility, convert_visibility_to_stokes, \
    vis_select_uvrange, deconvolve_cube, restore_cube, export_image_to_fits, qa_image, \
    image_gather_channels, create_image_from_visibility
from rascil.workflows import invert_list_rsexecute_workflow
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Benchmark pipelines in numpy and dask')
    parser.add_argument('--use_dask', type=str, default='True', help='Use Dask?')
    parser.add_argument('--nworkers', type=int, default=4, help='Number of workers')
    parser.add_argument('--threads', type=int, default=1,
                        help='Number of threads per worker')
    parser.add_argument('--memory', dest='memory', default=8,
                        help='Memory per worker (GB)')
    parser.add_argument('--npixel', type=int, default=512,
                        help='Number of pixels per axis')
    parser.add_argument('--context', dest='context', default='wstack',
                        help='Context: 2d|timeslice|wstack')
    parser.add_argument('--nchan', type=int, default=40,
                        help='Number of channels to process')
    parser.add_argument('--scheduler', type=str, default=None, help='Dask scheduler')

    args = parser.parse_args()
    print(args)

    # Put the results in the test_results directory
    results_dir = rascil_path('test_results')
    dask_dir = rascil_path('test_results/dask-work-space')


    # Since the processing is distributed over multiple processes we have to tell each Dask worker
    # where to send the log messages
    def init_logging():
        logging.basicConfig(filename='%s/dprepb-pipeline.log' % results_dir,
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)


    log = logging.getLogger()
    logging.info("Starting Imaging pipeline")

    # Set up rsexecute to use Dask. This means that all computation is delayed until an
    # explicit rsexecute.compute call. If use_dask is False, all calls are computed immediately.
    # If running on a cluster, create a scheduler externally and pass in the IP address
    if args.scheduler is not None:
        c = Client(args.scheduler)
        rsexecute.set_client(c)
    else:
        rsexecute.set_client(use_dask=args.use_dask == 'True',
                             threads_per_worker=args.threads,
                             n_workers=args.nworkers, local_directory=dask_dir)
    print(rsexecute.client)
    rsexecute.run(init_logging)

    nchan = args.nchan
    uvmax = 450.0
    cellsize = 0.00015
    npixel = args.npixel

    context = args.context
    if context == 'wstack':
        vis_slices = 45
        print('wstack processing')
    elif context == 'timeslice':
        print('timeslice processing')
        vis_slices = 2
    else:
        print('2d processing')
        context = '2d'
        vis_slices = 1

    input_vis = [rascil_path('data/vis/sim-1.ms'), rascil_path('data/vis/sim-2.ms')]

    import time

    start = time.time()


    # Define a function to be executed by Dask to load the data, combine it, and select
    # only the short baselines. We load each channel separately.
    def load_ms(c):
        v1 = create_visibility_from_ms(input_vis[0], start_chan=c, end_chan=c)[0]
        v2 = create_visibility_from_ms(input_vis[1], start_chan=c, end_chan=c)[0]
        vf = append_visibility(v1, v2)
        vf = convert_visibility_to_stokes(vf)
        vf.configuration.diameter[...] = 35.0
        rows = vis_select_uvrange(vf, 0.0, uvmax=uvmax)
        return create_visibility_from_rows(vf, rows)


    # Construct the graph to load the data and persist the graph on the Dask cluster.
    vis_list = [rsexecute.execute(load_ms)(c) for c in range(nchan)]
    vis_list = rsexecute.persist(vis_list)

    # Construct the graph to define the model images and persist the graph to the cluster
    model_list = [rsexecute.execute(create_image_from_visibility)
                  (v, npixel=npixel, cellsize=cellsize,
                   polarisation_frame=PolarisationFrame("stokesIQUV"),
                   nchan=1) for v in vis_list]
    model_list = rsexecute.persist(model_list)

    # Construct the graphs to make the dirty image and psf, and persist these to the cluster
    dirty_list = invert_list_rsexecute_workflow(vis_list,
                                                template_model_imagelist=model_list,
                                                context=context,
                                                vis_slices=vis_slices)
    psf_list = invert_list_rsexecute_workflow(vis_list,
                                              template_model_imagelist=model_list,
                                              context=context,
                                              dopsf=True,
                                              vis_slices=vis_slices)


    # Construct the graphs to do the clean and restoration, and gather the channel images
    # into one image. Persist the graph on the cluster
    def deconvolve(d, p, m):

        c, resid = deconvolve_cube(d[0], p[0], m, threshold=0.01, fracthresh=0.01,
                                   window_shape='quarter', niter=100, gain=0.1,
                                   algorithm='hogbom-complex')
        r = restore_cube(c, p[0], resid)
        return r


    restored_list = [rsexecute.execute(deconvolve)(dirty_list[c], psf_list[c],
                                                   model_list[c])
                     for c in range(nchan)]
    restored_cube = rsexecute.execute(image_gather_channels, nout=1)(restored_list)

    # Up to this point all we have is a graph. Now we compute it and get the
    # final restored cleaned cube. During the compute, Dask shows diagnostic pages
    # at http://127.0.0.1:8787
    restored_cube = rsexecute.compute(restored_cube, sync=True)

    # Save the cube
    print("Processing took %.3f s" % (time.time() - start))
    print(qa_image(restored_cube, context='CLEAN restored cube'))
    export_image_to_fits(restored_cube,
                         '%s/dprepb_rsexecute_%s_clean_restored_cube.fits'
                         % (results_dir, context))

    rsexecute.close()
