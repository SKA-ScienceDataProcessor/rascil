#!/bin/bash
#!
#! Dask job script for generic HPC Cluster: e.g., CNLAB-Cluster
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J RASCIL_PERF
#! Which project should be charged:
#SBATCH -A astro
#! How many whole nodes should be allocated?
#SBATCH --nodes=8
#! How many (MPI) tasks will there be in total? (<= nodes*16)
#SBATCH --ntasks=128
#! Memory limit: CNLAB has roughly 110GB per node
#SBATCH --mem 110000
#! How much wallclock time will be required?
#SBATCH --time=01:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=FAIL,END
#! Where to send email messages
#SBATCH --mail-user=wangfeng@cnlab.net
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#! Do not change:
#SBATCH -p hpc
#SBATCH -M cluster

#SBATCH --exclusive
#! Same switch
#SBATCH --switches=1
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):

module purge                               # Removes all modules still loaded

module load miniconda3/4.7.12.1-sa
source /home/${USER}/.bashrc

conda activate /mnt/storage-ssd/${USER}/anaconda

#! Set up python
echo -e "Running python: `which python`"
echo -e "Running dask-scheduler: `which dask-scheduler`"

cd $SLURM_SUBMIT_DIR
echo -e "Changed directory to `pwd`.\n"

JOBID=${SLURM_JOB_ID}
echo ${SLURM_JOB_NODELIST}

mkdir local

# Run RASCIL script.
 
rascildir=/mnt/storage-ssd/${USER}/work/rascil

export RASCIL=$rascildir 
export RASCIL_DATA=$rascildir/data
export PYTHONPATH=${rascildir}

JOBID=${SLURM_JOB_ID}

#! Create a hostfile:
scontrol show hostnames $SLURM_JOB_NODELIST > hostfile.$JOBID
echo ${SLURM_JOB_NODELIST}
outfile=${SLURM_JOB_NAME}_${SLURM_JOB_ID}_scheduler.out

port=8786
hostIndex=0
for host in `cat hostfile.$JOBID`; do
    ibhost=${host}
    if [ "$hostIndex" = "0" ]; then
        echo "Working on ${ibhost} ...."
        scheduler=${ibhost}
        echo "run dask-scheduler"
        echo "dask-scheduler --host ${scheduler} --port=8786 &"
        dask-scheduler --host ${scheduler} --port=8786 &
        hostIndex="1"
    fi
done

# Start dask-worker on all nodes using srun.
echo "run dask-worker" 
/usr/bin/srun -o %x_%j_worker_%n.out dask-worker --nanny --interface ib0 --nthreads 1  --memory-limit 0.75 --local-directory ${SLURM_SUBMIT_DIR}/local ${scheduler}:${port} &

echo "Scheduler and workers now running"
sleep 5

#! We need to tell dask Client (inside python) where the scheduler is running
export RASCIL_DASK_SCHEDULER=${scheduler}:${port}
echo "Scheduler is running at ${scheduler}"

CMD="python pipelines_rsexecute_timings.py --nnodes $SLURM_NNODES --nworkers $SLURM_NTASKS --use_serial_imaging False 
--use_serial_clean False --nthreads 1 --jobid $JOBID  --nfreqwin 128 --context 'timeslice' --rmax 1200  > pipelines_rsexecute_timings.out"

echo "About to execute $CMD"
eval $CMD

exit

