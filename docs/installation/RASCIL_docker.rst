
Running RASCIL under docker
***************************

For some of the step below it is helpful to have the RASCIL code tree available. Use::

   git clone https://github.com/SKA-ScienceDataProcessor/rascil
   cd rascil

Running on existing docker images
---------------------------------

The docker containers for RASCIL are at::

    timcornwell/rascil-no-data
    timcornwell/rascil-full

The first does not have the RASCIL test data but is smaller in size.

To run RASCIL with your home directory available inside the image::

    docker run -it --volume $HOME:$HOME timcornwell/rascil-full

Note that if you have the RASCIL code tree installed then you can also make your own versions
from the Dockerfiles in docker/rascil-no-data and docker/rascil-full.

To run an example script::

    python3 /rascil/examples/scripts/imaging.py


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

    python3 cluster_tests/ritoy/cluster_test_ritoy.py localhost:8786

The diagnostics page should be at http://127.0.0.1:8787

If the RASCIL data is already locally available then the images can be built without data using a slightly
different compose file. This assumes that the environment variable RASCIL_DATA points to the
data::

    docker-compose --file docker-compose-no-data.yml up --scale worker=4

The scheduler and 4 worker should now be running. To connect to the cluster, run the following into another window::

    docker run -it --network host timcornwell/rascil-full

To work with your own files inside a virtual machine, you can use the --volume command. The simplest approach is to map
your entire HOME directory in::

    docker run -it --network host --volume $HOME:$HOME timcornwell/rascil-full

Then at the docker prompt, do::

    python3 cluster_tests/ritoy/cluster_test_ritoy.py localhost:8786

