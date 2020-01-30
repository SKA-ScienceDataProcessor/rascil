
Running RASCIL under docker
***************************

To build::

    cd docker/docker-with-data
    docker image build -t timcornwell:rascil-with-data

To run interactively::

    docker run -it  timcornwell/rascil-with-data
    rascil@a31557a2c30d:/$ python
    Python 3.7.4 (default, Aug 13 2019, 20:35:49)
    [GCC 7.3.0] :: Anaconda, Inc. on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import rascil
    >>>


To run an example script::

    docker run timcornwell/rascil-with-data python rascil/examples/scripts/imaging.py

Dask can be run internally to a single container made from the image timcornwell/rascil-base. To make the
Dask diagnostics viewable on http://localhost:8787 add the port mapping::

    docker run -p 8787:8787  timcornwell/rascil-with-data python rascil/examples/scripts/dprepb_rsexecute_pipeline.py

This image created from docker-with-data/Dockerfile is quite large, partially because of the data files included. If the
RASCIL data files are already available. Then you can build the base version::

    cd docker/docker-base
    docker image build -t timcornwell:rascil-base

To run interactively, for example if the RASCIL data is in your current working directory::

    docker run -it -v $PWD/data:/rascil/data timcornwell/rascil-base
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




