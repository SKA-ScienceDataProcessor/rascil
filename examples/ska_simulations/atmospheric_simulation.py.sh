#!/bin/bash
#!
python atmospheric_simulation.py --context doublesource --frequency 1.36e9 \
--rmax 1e4 --flux_limit 0.003 --show True --pbtype MID_GAUSS --memory 32 \
--integration_time 90 --use_agg False --type_atmosphere ionosphere \
--time_range -4 4 --nworkers 16 --time_chunk 1800 --use_natural True \
--screen ../../screens/mid_screen_5000.0r0_0.033rate.fits --serial False \
--make_residual True --selfcal True --imaging_context ng --npixel 5120 \
--zerow False --dask_worker_space /mnt/storage-ssd/tim/dask-worker-space \
--nthreads 16 | tee atmospheric_simulation.log
