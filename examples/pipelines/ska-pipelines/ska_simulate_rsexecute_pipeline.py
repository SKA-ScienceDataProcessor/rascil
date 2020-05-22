"""
Simulation of observation for subsequent processing

Install the nifty gridder before using!

https://gitlab.mpcdf.mpg.de/ift/nifty_gridder


"""
from rascil.data_models import rascil_path

results_dir = './'
dask_dir = './'

import numpy
import logging

from astropy.coordinates import SkyCoord
from astropy import units as u

from rascil.data_models import PolarisationFrame
from rascil.processing_components import create_low_test_image_from_gleam
from rascil.workflows import predict_list_rsexecute_workflow, simulate_list_rsexecute_workflow, \
    corrupt_list_rsexecute_workflow
from rascil.processing_components import export_blockvisibility_to_ms
from rascil.workflows.rsexecute.execution_support.rsexecute import rsexecute

def init_logging():
    logging.basicConfig(filename='%s/ska-pipeline.log' % results_dir,
                        filemode='a',
                        format='%%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)

if __name__ == '__main__':
    log = logging.getLogger()
    print("Starting ska-pipelines simulation pipeline")
    
    rsexecute.set_client(use_dask=True)
    print(rsexecute.client)
    rsexecute.run(init_logging)
    
    # We create a graph to make the visibility. The parameter rmax determines the distance of the
    # furthest antenna/stations used. All over parameters are determined from this number.
    
    nfreqwin = 8
    ntimes = 5
    rmax = 750.0
    centre = nfreqwin // 2
    
    frequency = numpy.linspace(0.9e8, 1.1e8, nfreqwin)
    channel_bandwidth = numpy.array(nfreqwin * [frequency[1] - frequency[0]])
    times = numpy.linspace(-numpy.pi / 3.0, numpy.pi / 3.0, ntimes)
    phasecentre = SkyCoord(ra=+30.0 * u.deg, dec=-45.0 * u.deg, frame='icrs', equinox='J2000')
    
    vis_list = simulate_list_rsexecute_workflow('LOWBD2',
                                                rmax=rmax,
                                                frequency=frequency,
                                                channel_bandwidth=channel_bandwidth,
                                                times=times,
                                                phasecentre=phasecentre,
                                                order='frequency',
                                                format='blockvis')
    vis_list = rsexecute.persist(vis_list)
    
    npixel = 1024
    cellsize = 0.0005
    
    dprepb_model = [rsexecute.execute(create_low_test_image_from_gleam)(npixel=npixel,
                                                                        frequency=[frequency[f]],
                                                                        channel_bandwidth=[channel_bandwidth[f]],
                                                                        cellsize=cellsize,
                                                                        phasecentre=phasecentre,
                                                                        polarisation_frame=PolarisationFrame("stokesI"),
                                                                        flux_limit=3.0,
                                                                        applybeam=True)
                    for f, freq in enumerate(frequency)]

    def print_max(v):
        print(numpy.max(numpy.abs(v.vis)))
        return v
    
    imaging_context = "ng"
    vis_slices=1
    print('Using {}'.format(imaging_context))
    predicted_vislist = predict_list_rsexecute_workflow(vis_list, dprepb_model, context=imaging_context,
                                                        vis_slices=vis_slices, verbosity=2)
    corrupted_vislist = corrupt_list_rsexecute_workflow(predicted_vislist, phase_error=1.0, seed=180555)
    corrupted_vislist = [rsexecute.execute(print_max)(v) for v in corrupted_vislist]
    export_list = [rsexecute.execute(export_blockvisibility_to_ms)
                   (rascil_path('%s/ska-pipeline_simulation_vislist_%d.ms' % (results_dir, v)), [corrupted_vislist[v]])
                   for v, _ in enumerate(corrupted_vislist)]
    
    print('About to run predict and corrupt to get corrupted visibility, and write files')
    rsexecute.compute(export_list, sync=True)
    
    rsexecute.close()
