#!/bin/bash
#!
#! Dask job script for P3
#! Tim Cornwell
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J ATMOS
#! Which project should be charged:
#SBATCH -A SKA-SDP
#! How many whole nodes should be allocated?
#SBATCH --nodes=16
#! How many (MPI) tasks will there be in total? (<= nodes*16)
#SBATCH --ntasks=33
#! Memory limit: P3 has roughly 107GB per node
##SBATCH --mem 50000
#! How much wallclock time will be required?
#SBATCH --time=23:59:59
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=FAIL,END
#! Where to send email messages
#SBATCH --mail-user=realtimcornwell@gmail.com
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#! Do not change:
#SBATCH -p compute
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue

#! Modify the settings below to specify the application's environment, location
#! and launch method:

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
module purge                               # Removes all modules still loaded

#! Set up python
# . $HOME/alaska-venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$ARL
echo "PYTHONPATH is ${PYTHONPATH}"

echo -e "Running python: `which python`"
echo -e "Running dask-scheduler: `which dask-scheduler`"

cd $SLURM_SUBMIT_DIR
echo -e "Changed directory to `pwd`.\n"

JOBID=${SLURM_JOB_ID}
echo ${SLURM_JOB_NODELIST}

#! Create a hostfile:
scontrol show hostnames $SLURM_JOB_NODELIST | uniq > hostfile.$JOBID


scheduler=$(head -1 hostfile.$JOBID)
hostIndex=0
for host in `cat hostfile.$JOBID`; do
    echo "Working on $host ...."
    if [ "$hostIndex" = "0" ]; then
        echo "run dask-scheduler"
        ssh $host dask-scheduler --port=8786 &
        sleep 5
    fi
    echo "run dask-worker"
    ssh $host dask-worker --host ${host} --nprocs 2 --nthreads 1  \
    --memory-limit 16GB --local-directory /mnt/storage-ssd/tim/dask-workspace/${host} $scheduler:8786  &
        sleep 1
    hostIndex="1"
done
echo "Scheduler and workers now running"

#! We need to tell dask Client (inside python) where the scheduler is running
export RASCIL_DASK_SCHEDULER=${scheduler}:8786
echo "Scheduler is running at ${scheduler}"

CMD="python atmospheric_simulation.py --context doublesource --frequency 1.36e9 \
--rmax 1e4 --flux_limit 0.003 --show True --pbtype MID_GAUSS --memory 32 \
--integration_time 90 --use_agg False --type_atmosphere ionosphere \
--time_range -4 4 --nworkers 16 --time_chunk 1800 --use_natural True \
--screen ../../screens/mid_screen_5000.0r0_0.033rate.fits --serial False \
--make_residual True --selfcal True --imaging_context ng --npixel 5120 \
--zerow False --dask_worker_space /mnt/storage-ssd/tim/dask-worker-space \
--nthreads 16 | tee atmospheric_simulation.log
"
echo "About to execute $CMD"

eval $CMD

