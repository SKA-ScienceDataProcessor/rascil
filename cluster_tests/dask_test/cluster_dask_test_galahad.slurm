#!/bin/bash
#!
#! Dask job script for GALAHAD
#! Tim Cornwell
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J ERP
#! Which project should be charged:
##SBATCH -A SKA-SDP
#! How many whole nodes should be allocated?
#SBATCH --nodes=8
#! How many (MPI) tasks will there be in total? (<= nodes*16)
#SBATCH --ntasks=16
#! Memory limit
##SBATCH --mem 50000
#! How much wallclock time will be required?
#SBATCH --time=10:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=FAIL,END
#! Where to send email messages
#SBATCH --mail-user=realtimcornwell@gmail.com
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#! Do not change:
#SBATCH -p CLUSTER
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue

#! Modify the settings below to specify the application's environment, location
#! and launch method:

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
module load python37

export ERP=/home/cornwell/Code/eMERLIN_RASCIL_pipeline
export RASCIL=/home/cornwell/Code/rascil

export PYTHONPATH=$PYTHONPATH:$RASCIL:$ERP
echo "PYTHONPATH is ${PYTHONPATH}"

echo -e "Running python: `which python3`"
echo -e "Running dask-scheduler: `which dask-scheduler`"

cd $SLURM_SUBMIT_DIR
echo -e "Changed directory to `pwd`.\n"

JOBID=${SLURM_JOB_ID}
echo ${SLURM_JOB_NODELIST}

#! Create a hostfile:
scontrol show hostnames $SLURM_JOB_NODELIST | uniq > hostfile.$JOBID
cat hostfile.$JOBID

scheduler=$(head -1 hostfile.$JOBID)
hostIndex=0
for host in `cat hostfile.$JOBID`; do
    echo "Working on $host ...."
    if [ "$hostIndex" = "0" ]; then
        killall dask-scheduler
    fi
    killall dask-worker python3
    hostIndex="1"
done

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
    ssh $host dask-worker --nprocs 2 --nthreads 1 $scheduler:8786  &
        sleep 1
    hostIndex="1"
done
echo "Scheduler and workers now running"

#! We need to tell dask Client (inside python) where the scheduler is running
echo "Scheduler is running at ${scheduler}"

CMD="python3 cluster_dask_test.py ${scheduler}:8786 | tee cluster_dask_test.log"

eval $CMD

