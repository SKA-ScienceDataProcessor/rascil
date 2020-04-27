#!/bin/bash
#!
python polarisation.py --context s3sky --rmax 1e4 --flux_limit 0.01 \
 --show True --declination -45 \
 --band B2 --pbtype MID_FEKO_B2  --integration_time 120 --use_agg True \
--time_chunk 120 --time_range -4 4  | tee polarisation_simulation.log
