#!/bin/bash
#!
#! Dask job script for generic HPC Cluster: e.g., CNLAB-Cluster
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J CLUSTER_TEST
#! Which project should be charged:
#SBATCH -A astro
#! How many whole nodes should be allocated?
#SBATCH --nodes=8
#! How many (MPI) tasks will there be in total? (<= nodes*16)
#SBATCH --ntasks=16
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

module load miniconda3-4.6.14-gcc-8.3.0-wv6owry
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

# Start dask-scheduler on first node.
 
scheduler=$(hostname)
port=8786
outfile=${SLURM_JOB_NAME}_${SLURM_JOB_ID}_scheduler.out
#dask-scheduler --host $scheduler --port $port &> $outfile &
dask-scheduler --host $scheduler --port $port &
echo dask-scheduler started on ${scheduler}:${port}
sleep 5
 
# Start dask-worker on all nodes using srun.
 
srun -o %x_%j_worker_%n.out dask-worker --nprocs 16 --nthreads 1 --interface ib0 --memory-limit 100GB --local-directory ${SLURM_SUBMIT_DIR}/local ${scheduler}:${port} &
echo dask-worker started on all nodes
sleep 5
 
# Run RASCIL script.
 
rascildir=/mnt/storage-ssd/${USER}/work/rascil

export RASCIL=$rascildir 
export PYTHONPATH=${arldir}
export RASCIL_DASK_SCHEDULER=${scheduler}:${port}

sleep 1
echo "Scheduler and workers now running"

#! We need to tell dask Client (inside python) where the scheduler is running
echo "Scheduler is running at ${scheduler}"

echo "run dask-worker"
python cluster_dask_test.py ${scheduler}:8786 | tee cluster_dask_test.log
 