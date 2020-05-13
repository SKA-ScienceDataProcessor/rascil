"""
Continuum processing pipeline
"""
# # Pipeline processing using Dask

from rascil.data_models.parameters import rascil_path

results_dir = './'
dask_dir = './'

from rascil.data_models import PolarisationFrame

from rascil.processing_components import export_image_to_fits, qa_image,  \
    create_image_from_visibility, create_blockvisibility_from_ms

from rascil.workflows import continuum_imaging_list_rsexecute_workflow

from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute

import logging


def init_logging():
    logging.basicConfig(filename='%s/ska-pipeline.log' % results_dir,
                        filemode='a',
                        format='%(asctime)s.%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)


if __name__ == '__main__':
    log = logging.getLogger()
    logging.info("Starting continuum imaging pipeline")
    
    rsexecute.set_client(use_dask=True)
    print(rsexecute.client)
    rsexecute.run(init_logging)
    
    nfreqwin = 8
    ntimes = 5
    rmax = 750.0
    centre = nfreqwin // 2
    
    # Load data from previous simulation
    vis_list = [rsexecute.execute(create_blockvisibility_from_ms)
                     (rascil_path('%s/ska-pipeline_simulation_vislist_%d.ms' % (results_dir, v)))[0]
                     for v in range(nfreqwin)]
    
    print('Reading visibilities')
    vis_list = rsexecute.persist(vis_list)
    
    cellsize = 0.0005
    npixel = 1024
    pol_frame = PolarisationFrame("stokesI")
    
    model_list = [rsexecute.execute(create_image_from_visibility)(v, npixel=npixel, cellsize=cellsize,
                                                                  polarisation_frame=pol_frame)
                  for v in vis_list]
    
    print('Creating model images')
    model_list = rsexecute.persist(model_list)
    
    imaging_context='ng'
    vis_slices = 1
    continuum_imaging_list = continuum_imaging_list_rsexecute_workflow(vis_list,
                                                                       model_imagelist=model_list,
                                                                       context=imaging_context,
                                                                       vis_slice=vis_slices,
                                                                       scales=[0, 3, 10], algorithm='mmclean',
                                                                       nmoment=2, niter=1000,
                                                                       fractional_threshold=0.1,
                                                                       threshold=0.1, nmajor=5, gain=0.25,
                                                                       deconvolve_facets=1,
                                                                       deconvolve_overlap=0,
                                                                       deconvolve_taper='tukey',
                                                                       restore_facets=8,
                                                                       timeslice='auto',
                                                                       psf_support=128)
    
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
