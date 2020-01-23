
Running RASCIL under docker
***************************

To build::

    cd docker/docker-base
    docker image build -t timcornwell:rascil-base
    docker images

To run interactively::

    docker run -it -v $PWD/data:/rascil/data timcornwell/rascil-base

