#!/bin/bash
#!
python pointing_simulation.py --context s3sky --frequency 1.36e9 --rmax 1e5 --flux_limit 0.003 \
 --show True --seed 18051955  --pbtype MID_FEKO_B2 --memory 32 --integration_time 30 --use_agg True \
 --time_series wind --time_chunk 1800 | tee pointing_simulation.log
