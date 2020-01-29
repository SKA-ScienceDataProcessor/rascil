
Running RASCIL under docker
***************************

docker-compose provides a simple way to create a local cluster of a Dask scheduler and a number of workers.
To scale to e.g. 4 dask workers::

    cd docker
    docker-compose --scale workers=4 up

To connect to the cluster::

    docker run -it --network host timcornwell/rascil-full

Then at the docker prompt, we can run a test program in python. This test program takes the
address of the scheduler on the command line. Hence the command at the docker command line::

    python cluster_tests/ritoy/cluster_test_ritoy.py localhost:8786

The diagnostics page should be at http://127.0.0.1:8787

If the RASCIL data is already locally available then the images can be built without data using a slightly
different compose file. This assumes that the environment variable RASCIL_DATA points to the
data::

    docker-compose --file docker-compose-no-data.yml up --scale workers=4
    docker run -it --network host timcornwell/rascil-full

Then at the docker prompt, do::

    python cluster_tests/ritoy/cluster_test_ritoy.py localhost:8786
