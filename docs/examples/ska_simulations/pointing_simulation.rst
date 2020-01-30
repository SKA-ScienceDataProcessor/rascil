.. _ska_pointing_simulation:

SKA dish pointing error simulations
===================================

This calculates the change in a dirty image caused by wind-induced pointing errors:

    - The sky can be a point source at the half power point or a realistic sky constructed from S3-SEX catalog.
    - The observation is by MID over a range of hour angles
    - Processing can be divided into chunks of time (default 1800s)
    - Dask is used to distribute the processing over a number of workers.
    - Various plots are produced, The primary output is a csv file containing information about the statistics of the residual images.


The full set of test scripts are available at: https://github.com/ska-telescope/sim-mid-pointing

The python script is:

.. code:: python

     #!/bin/bash
     #!
     python pointing_simulation.py --context s3sky --frequency 1.36e9 --rmax 1e5 --flux_limit 0.003 \
      --show True --seed 18051955  --pbtype MID_FEKO_B2 --memory 32 --integration_time 30 --use_agg True \
      --time_series wind --time_chunk 1800 | tee pointing_simulation.log

The shell script to run is:


.. code:: sh

     #!/bin/bash
     #!
     python pointing_simulation.py --context s3sky --frequency 1.36e9 --rmax 1e5 --flux_limit 0.003 \
      --show True --seed 18051955  --pbtype MID_FEKO_B2 --memory 32 --integration_time 30 --use_agg True \
      --time_series wind --time_chunk 1800 | tee pointing_simulation.log

The SLURM batch file is:


.. code:: sh

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
     #SBATCH -J CASE5_30s
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
     
     CMD="python pointing_simulation.py --context s3sky --frequency 1.36e9 --rmax 1e5 --flux_limit 0.003 \
      --show True --seed 18051955  --pbtype MID_FEKO_B2 --memory 32 --integration_time 30 --use_agg True \
      --time_series wind --time_chunk 1800 | tee pointing_simulation.log"
     echo "About to execute $CMD"
     
     eval $CMD
     



.. code:: python

     #!/bin/bash
     #!
     python pointing_simulation.py --context s3sky --frequency 1.36e9 --rmax 1e5 --flux_limit 0.003 \
      --show True --seed 18051955  --pbtype MID_FEKO_B2 --memory 32 --integration_time 30 --use_agg True \
      --time_series wind --time_chunk 1800 | tee pointing_simulation.log

The shell script to run is:


.. code:: sh

     #!/bin/bash
     #!
     python pointing_simulation.py --context s3sky --frequency 1.36e9 --rmax 1e5 --flux_limit 0.003 \
      --show True --seed 18051955  --pbtype MID_FEKO_B2 --memory 32 --integration_time 30 --use_agg True \
      --time_series wind --time_chunk 1800 | tee pointing_simulation.log

The SLURM batch file is:


.. code:: sh

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
     #SBATCH -J CASE5_30s
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
     
     CMD="python pointing_simulation.py --context s3sky --frequency 1.36e9 --rmax 1e5 --flux_limit 0.003 \
      --show True --seed 18051955  --pbtype MID_FEKO_B2 --memory 32 --integration_time 30 --use_agg True \
      --time_series wind --time_chunk 1800 | tee pointing_simulation.log"
     echo "About to execute $CMD"
     
     eval $CMD
     


