"""
Continuum processing pipeline
"""
# # Pipeline processing using Dask

from rascil.data_models.parameters import rascil_path

results_dir = rascil_path('test_results')
dask_dir = rascil_path('test_results/dask-work-space')

from rascil.data_models import PolarisationFrame, import_blockvisibility_from_hdf5

from rascil.processing_components import export_image_to_fits, qa_image, create_image_from_visibility,  \
    convert_blockvisibility_to_visibility

from rascil.workflows import continuum_imaging_list_rsexecute_workflow

from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute

import logging

def init_logging():
    logging.basicConfig(filename='%s/ska-pipeline.log' % results_dir,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)


if __name__ == '__main__':
    
    log = logging.getLogger()
    logging.info("Starting continuum imaging pipeline")
    
    rsexecute.set_client(use_dask=True, threads_per_worker=1, memory_limit=32 * 1024 * 1024 * 1024, n_workers=8,
                          local_dir=dask_dir, verbose=True)
    print(rsexecute.client)
    rsexecute.run(init_logging)
    
    nfreqwin = 41
    ntimes = 5
    rmax = 750.0
    centre = nfreqwin // 2
    
    # Load data from previous simulation
    block_vislist = [rsexecute.execute(import_blockvisibility_from_hdf5)
                (rascil_path('%s/ska-pipeline_simulation_vislist_%d.hdf' % (results_dir, v)))
                for v in range(nfreqwin)]
    
    vis_list = [rsexecute.execute(convert_blockvisibility_to_visibility, nout=1)(bv) for bv in block_vislist]
    print('Reading visibilities')
    vis_list = rsexecute.compute(vis_list, sync=True)
    
    cellsize = 0.001
    npixel = 1024
    pol_frame = PolarisationFrame("stokesI")
    
    model_list = [rsexecute.execute(create_image_from_visibility)(v, npixel=npixel, cellsize=cellsize,
                                                                   polarisation_frame=pol_frame)
                  for v in vis_list]
    
    print('Creating model images')
    model_list = rsexecute.compute(model_list, sync=True)
    
    print('Creating graph')
    future_vis_list = rsexecute.scatter(vis_list)
    future_model_list = rsexecute.scatter(model_list)
    
    continuum_imaging_list = continuum_imaging_list_rsexecute_workflow(future_vis_list,
                                                                        model_imagelist=future_model_list,
                                                                        context='wstack', vis_slices=51,
                                                                        scales=[0, 3, 10], algorithm='mmclean',
                                                                        nmoment=3, niter=1000,
                                                                        fractional_threshold=0.1,
                                                                        threshold=0.1, nmajor=5, gain=0.25,
                                                                        deconvolve_facets=1,
                                                                        deconvolve_overlap=0,
                                                                        deconvolve_taper='tukey',
                                                                        timeslice='auto',
                                                                        psf_support=64)
    
    log.info('About to run continuum imaging workflow')
    result = rsexecute.compute(continuum_imaging_list, sync=True)
    rsexecute.close()
    
    deconvolved = result[0][centre]
    residual = result[1][centre]
    restored = result[2][centre]
    
    print(qa_image(deconvolved, context='Clean image'))
    export_image_to_fits(deconvolved, '%s/ska-continuum-imaging_rsexecute_deconvolved.fits' % (results_dir))
    
    print(qa_image(restored, context='Restored clean image'))
    export_image_to_fits(restored, '%s/ska-continuum-imaging_rsexecute_restored.fits' % (results_dir))
    
    print(qa_image(residual[0], context='Residual clean image'))
    export_image_to_fits(residual[0], '%s/ska-continuum-imaging_rsexecute_residual.fits' % (results_dir))
