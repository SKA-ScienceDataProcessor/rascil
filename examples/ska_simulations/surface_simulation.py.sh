#!/bin/bash
#!
python surface_simulation.py --context s3sky --rmax 1e5 --flux_limit 0.003 \
 --show False --elevation_sampling 5.0 --declination -45 \
--vp_directory /mnt/storage-ssd/tim/Code/sim-mid-surface/beams/interpolated/ \
 --band B2 --pbtype MID_FEKO_B2  --integration_time 120 --use_agg True \
--time_chunk 120 --time_range -6 6  | tee surface_simulation_P3_login.log
