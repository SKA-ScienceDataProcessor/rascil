"""
Imaging pipeline
"""

# # Pipeline processing using Dask

from rascil.data_models.parameters import rascil_path

results_dir ='./'
dask_dir='./'

from rascil.data_models import PolarisationFrame

from rascil.processing_components import export_image_to_fits, create_image_from_visibility, \
    create_blockvisibility_from_ms

from rascil.workflows import invert_list_rsexecute_workflow

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
    logging.info("Starting Imaging pipeline")
    
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

    vis_list = rsexecute.persist(vis_list)
    
    cellsize = 0.0005
    npixel = 1024
    pol_frame = PolarisationFrame("stokesI")
    
    model_list = [rsexecute.execute(create_image_from_visibility)(v, npixel=npixel, cellsize=cellsize,
                                                                   polarisation_frame=pol_frame)
                  for v in vis_list]
    
    model_list = rsexecute.persist(model_list)
    
    imaging_context = 'ng'
    vis_slices = 1
    dirty_list = invert_list_rsexecute_workflow(vis_list, template_model_imagelist=model_list, context=imaging_context,
                                                vis_slices=vis_slices)
    
    log.info('About to run invert_list_rsexecute_workflow')
    result = rsexecute.compute(dirty_list, sync=True)
    dirty, sumwt = result[centre]
    
    rsexecute.close()
    
    export_image_to_fits(dirty, '%s/ska-imaging_rsexecute_dirty.fits' % (results_dir))
    
    exit(0)
