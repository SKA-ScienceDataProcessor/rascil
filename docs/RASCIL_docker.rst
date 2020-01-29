
Running RASCIL under docker
***************************

Using docker-compose
--------------------
docker-compose provides a simple way to create a cluster. To scale to e.g. 4 dask workers::

    cd docker
    docker-compose --scale workers=4 up

To connect to the cluster and run a test program::

    docker run -it --network host timcornwell/rascil-full

Then at the docker prompt, do::

    python cluster_tests/ritoy/cluster_test_ritoy.py localhost:8786

The diagnostics page should be at http://127.0.0.1:8787

Using docker build and run
--------------------------

To build::

    cd docker/rascil-full
    docker image build -t timcornwell:rascil-full

To run interactively::

    docker run -it  timcornwell/rascil-full
    rascil@a31557a2c30d:/$ python
    Python 3.7.4 (default, Aug 13 2019, 20:35:49)
    [GCC 7.3.0] :: Anaconda, Inc. on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import rascil
    >>>

To run an example script::

    docker run timcornwell/rascil-full python examples/scripts/imaging.py

Dask can be run internally to a single container made from the image timcornwell/rascil-full. To make the
Dask diagnostics viewable on http://localhost:8787 add the port mapping::

    docker run -p 8787:8787  timcornwell/rascil-full python examples/scripts/dprepb_rsexecute_pipeline.py

This image created from docker-full/Dockerfile is quite large, partially because of the data files included. If the
RASCIL data files are already available. Then you can build the base version::

    cd docker/rascil-minimal
    docker image build -t timcornwell:rascil-minimal

To run interactively, for example if the RASCIL data is in your current working directory::

    docker run -it -v $PWD/data:/rascil/data timcornwell/rascil-minimal
    rascil@a31557a2c30d:/$ python
    Python 3.7.4 (default, Aug 13 2019, 20:35:49)
    [GCC 7.3.0] :: Anaconda, Inc. on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import rascil
    >>>


To run an example script::

    docker run -v $PWD/data:/rascil/data timcornwell/rascil-base python rascil/examples/scripts/imaging.py

Dask can be run internally to a single container made from the image timcornwell/rascil-base. To make the
Dask diagnostics viewable on http://localhost:8787 add the port mapping::

    docker run -p 8787:8787 -v $PWD/data:/rascil/data timcornwell/rascil-base python rascil/examples/scripts/dprepb_rsexecute_pipeline.py



