
Using Docker
=============

In the tools/ directory is a [Dockerfile](https://docs.docker.com/engine/reference/builder/)
that enables the notebooks, tests, and lint checks to be run in a container.

Install Docker
--------------
This has been tested with Docker-CE 17.09+ on Ubuntu 17.04. This can be installed by following the instructions [here](https://docs.docker.com/engine/installation/).

It is likely that the Makefile commands will not work on anything other than modern Linux systems (eg: Ubuntu 17.04) as it relies on command line tools to discover the host system IP address.

The project source code directory needs to be checked out where ever the containers are to be run
as the complete project is mounted into the container as a volume at `/rascil` .

For example - to launch the notebook server, the general recipe is:
```
docker run --name rascil_notebook --hostname rascil_notebook --volume /path/to/repo:/rascil \
-e IP=my.ip.address --net=host -p 8888:8888 -p 8787:8787 -p 8788:8788 -p 8789:8789 \
-d rascil_img
```
After a few seconds, check the logs for the Jupyter URL with:
```
docker logs rascil_notebook
```
See the `Makefile` for more examples.


Build Image
-----------
To build the container image `rascil_img` required to launch the dockerised notebooks,tests, and lint checks run (from the root directory of this checked out project) - pass in the PYTHON variable to specify which build of Python to use - python3 or python3.6:
```
make docker_build PYTHON=python3
```

Then push to a given Docker repository:
```
make docker_push PYTHON=python3 DOCKER_REPO=localhost:5000
```

Run
---
To run the Jupyter notebooks:
```
make docker_notebook
```
Wait for the command to complete, and it will print out the URL with token to use for access to the notebooks (example output):
```
...
Successfully built 8a4a7b55025b
...
docker run --name rascil_notebook --hostname rascil_notebook --volume $(pwd):/rascil -e IP=${IP} \
            --net=host -p 8888:8888 -p 8787:8787 -p 8788:8788 -p 8789:8789 -d rascil_img
Launching at IP: 10.128.26.15
da4fcfac9af117c92ac63d4228087a05c5cfbb2fc55b2e281f05ccdbbe3ca0be
sleep 3
docker logs rascil_notebook
[I 02:25:37.803 NotebookApp] Writing notebook server cookie secret to /root/.local/share/jupyter/runtime/notebook_cookie_secret
[I 02:25:37.834 NotebookApp] Serving notebooks from local directory: /rascil/examples/rascil
[I 02:25:37.834 NotebookApp] 0 active kernels
[I 02:25:37.834 NotebookApp] The Jupyter Notebook is running at:
[I 02:25:37.834 NotebookApp] http://10.128.26.15:8888/?token=2c9f8087252ea67b4c09404dc091563b16f154f3906282b7
[I 02:25:37.834 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 02:25:37.835 NotebookApp]

    Copy/paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://10.128.26.15:8888/?token=2c9f8087252ea67b4c09404dc091563b16f154f3906282b7
```

To run the tests:
```
make docker_tests
```

To run the lint checks:
```
make docker_lint
```

Dask with Docker Swarm
----------------------

If you have a Docker Swarm cluster then the Dask cluster can be launched as follows:

Assuming that the Docker Image for the RASCIL has been built and pushed to a repository tag eg:
```
10.128.26.15:5000/rascil_img:latest
```
And the Swarm cluster master resides on:
```
10.101.1.23
```
And the contents of rascil/data is available on every work at:
```
/home/ubuntu/rascildata
```
Then launch the Dask Scheduler, and Workers with the following:
```
docker -H 10.101.1.23 service create --detach=true \
 --constraint 'node.role == manager' \
 --name dask_scheduler --network host --mode=global \
   10.128.26.15:5000/rascil_img:latest \
   dask-scheduler --host 0.0.0.0 --bokeh --show

docker -H 10.101.1.23 service create --detach=true \
 --name dask_worker --network host --mode=global \
 --mount type=bind,source=/home/ubuntu/rascildata,destination=/rascil/data \
   10.128.26.15:5000/rascil_img:latest \
   dask-worker --host 0.0.0.0 --bokeh --bokeh-port 8788  --nprocs 4 --nthreads 1 --reconnect 10.101.1.23:8786
```

Now you can point the Dask client at the cluster with:
```
export RASCIL_DASK_SCHEDULER=10.101.1.23:8786
python examples/performance/pipelines-timings.py 4 4
```

