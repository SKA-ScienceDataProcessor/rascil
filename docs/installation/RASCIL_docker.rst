
Running RASCIL under docker
***************************

For some of the steps below it is helpful to have the RASCIL code tree available. Use::

   git clone https://github.com/SKA-ScienceDataProcessor/rascil
   cd rascil

Running on existing docker images
---------------------------------

The docker containers for RASCIL are on github at::

    docker.io/timcornwell/rascil-no-data
    docker.io/timcornwell/rascil-full

The first does not have the RASCIL test data but is smaller in size (2GB vs 4GB). However, for many of the tests
and demonstrations the test data is needed.

To run RASCIL with your home directory available inside the image::

    docker run -it --volume $HOME:$HOME timcornwell/rascil-full

Now let's run an example. First it simplifies using the container if we do not
try to write inside the container, and that's why we mapped in our $HOME directory.
So to run the /rascil/examples/scripts/imaging.py script, we first change directory
to the name of the HOME directory, which is the same inside and outside the
container, and then give the full address of the script inside the container. This time
we will show the prompts from inside the container::

     % docker run -p 8888:8888 -v $HOME:$HOME -it timcornwell/rascil-full
     rascil@d0c5fc9fc19d:/rascil$ cd /<your home directory>
     rascil@d0c5fc9fc19d:/<your home directory>$ python3 /rascil/examples/scripts/imaging.py
     ...
     rascil@d0c5fc9fc19d:/<your home directory>$ ls -l imaging*.fits
     -rw-r--r-- 1 rascil rascil 2102400 Feb 11 14:04 imaging_dirty.fits
     -rw-r--r-- 1 rascil rascil 2102400 Feb 11 14:04 imaging_psf.fits
     -rw-r--r-- 1 rascil rascil 2102400 Feb 11 14:04 imaging_restored.fits

In this example, we change directory to an external location (my home directory in this case,
use yours instead), and then we run the script using the absolute path name inside the container.

Running notebooks
-----------------

We also want to be able to run jupyter notebooks inside the container::

    docker run -it -p 8888:8888 --volume $HOME:$HOME timcornwell/rascil-full
    cd /<your home directory>
    jupyter notebook --no-browser --ip 0.0.0.0  /rascil/examples/notebooks/

The juptyer server will start and output possible URLs to use::

    [I 14:08:39.041 NotebookApp] Serving notebooks from local directory: /rascil/examples/notebooks
    [I 14:08:39.041 NotebookApp] The Jupyter Notebook is running at:
    [I 14:08:39.042 NotebookApp] http://d0c5fc9fc19d:8888/?token=f050f82ed0f8224e559c2bdd29d4ed0d65a116346bcb5653
    [I 14:08:39.042 NotebookApp]  or http://127.0.0.1:8888/?token=f050f82ed0f8224e559c2bdd29d4ed0d65a116346bcb5653
    [I 14:08:39.042 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
    [W 14:08:39.045 NotebookApp] No web browser found: could not locate runnable browser.

The 127.0.0.1 is the one we want. Enter this address in your local browser. You should see
the standard jupyter directory page.

Running RASCIL as a cluster
---------------------------

The file docker/docker-compose in the RASCIL code tree provides a simple way to
create a local cluster of a Dask scheduler and a number of workers. First install
RASCIL using git clone as described above. The cluster is created using the
docker-compose up command. To scale to e.g. 4 dask workers::

    cd docker
    docker-compose up --scale worker=4

The scheduler and 4 worker should now be running. To connect to the cluster, run the following into another window::

    docker run -it --network host timcornwell/rascil-full

Then at the docker prompt, we can run a test program in python. This test program takes the
address of the scheduler on the command line. Hence the command at the docker command line::

    python3 /rascil/cluster_tests/ritoy/cluster_test_ritoy.py localhost:8786

The diagnostics page should be at http://127.0.0.1:8787. This test should conclude in about
two minutes.

If the RASCIL data is already locally available then the images can be built without data using a slightly
different compose file. This assumes that the environment variable RASCIL_DATA points to the
data::

    docker-compose --file docker-compose-no-data.yml up --scale worker=4

The scheduler and 4 workers should now be running. To connect to the cluster, run the
following into another window::

    docker run -it --network host --volume $HOME:$HOME timcornwell/rascil-full

Then at the docker prompt, do e.g.::

    cd /<your home directory>
    python3 /rascil/cluster_tests/ritoy/cluster_test_ritoy.py localhost:8786

A jupyter lab notebook is also started by this docker-compose. The URL will be output during the
initial set up, e.g.::

    notebook_1   | [I 15:17:05.681 NotebookApp] The Jupyter Notebook is running at:
    notebook_1   | [I 15:17:05.682 NotebookApp] http://notebook:8888/?token=0e77cf0e214fb0f5827b35fa5de8bbc5ebed6d4159e3d31e
    notebook_1   | [I 15:17:05.682 NotebookApp]  or http://127.0.0.1:8888/?token=0e77cf0e214fb0f5827b35fa5de8bbc5ebed6d4159e3d31e
    notebook_1   | [I 15:17:05.682 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).

Click on the 127.0.0.1 URL. We have used the jupyter lab interface instead of jupyter notebook interface
because the former allows control of Dask from the interface. This can be changed in the docker-compose.yml
file. Note also that the classic notebook interface can be selected at the lab interface.

CASA Measures Tables
--------------------

We use the CASA measures system for TAI/UTC corrections. These rely upon tables downloaded from NRAO.
It may happen that the tables become out ofdate. If so do the following at the command prompt inside a
docker image::

    rsync -avz rsync://casa-rsync.nrao.edu/casa-data/geodetic /var/lib/casacore/data


Singularity
-----------

`Singularity <https://sylabs.io/docs/>`_ can be used to load and run the docker images::

    singularity pull RASCIL.img docker://timcornwell/rascil-full-no-root
    singularity run RASCIL.img
    python3 /rascil/examples/scripts/imaging.py

Note that we use the -no-root versions of the docker images to avoid singularity
complaining about a non-existent user RASCIL. As in docker, don't run from the /rascil/directory.

Inside a SLURM file singularity can be used by prefacing dask and python commands
with singularity. For example::

    ssh $host singularity exec /home/<your-name>/workspace/RASCIL-full.img dask-scheduler --port=8786 &
    ssh $host singularity exec /home/<your-name>/workspace/RASCIL-full.img dask-worker --host ${host} --nprocs 4 --nthreads 1  \
    --memory-limit 100GB $scheduler:8786 &
    CMD="singularity exec /home/<your-name>/workspace/RASCIL-full.img python3 ./cluster_test_ritoy.py ${scheduler}:8786 | tee ritoy.log"
    eval $CMD

Customisability
---------------

The docker images described here are ones we have found useful. However,
if you have the RASCIL code tree installed then you can also make your own versions
working from these Dockerfiles.

