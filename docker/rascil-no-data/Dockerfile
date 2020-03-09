FROM ubuntu:18.04

LABEL author="Tim Cornwell <realtimcornwell@gmail.com>" \
      description="RASCIL no-data reference image" \
      license="Apache2.0"

# Install required system packages.
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get install -y \
    ca-certificates \
    gosu git wget curl \
    libxml2-dev liblapack-dev libcfitsio-dev libgtkmm-3.0-dev \
    cmake g++ build-essential gfortran libpng-dev casacore-dev \
    python3 python3-dev python3-pip python3-setuptools && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# This will clone into (and create) a directory at /rascil
# The data directory will not be loaded correctly until LFS is enabled
RUN git clone https://github.com/SKA-ScienceDataProcessor/rascil.git

# Set runtime environment variables.
ENV RASCIL=/rascil
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Change to the RASCIL root directory to run the commands below.
WORKDIR /rascil

# We choose to put the dask scratch files inside the container.
# It might be better to map it in with --volume
RUN mkdir -p /rascil/dask-worker-space && \
    chmod 777 /rascil/dask-worker-space

RUN pip3 install Cython setuptools && \
    pip3 install aotools \
    astropy \
    dask \
    distributed \
    bokeh \
    h5py \
    jupyter \
    jupyter_contrib_nbextensions \
    jupyter-server-proxy \
    jupyterlab \
    "matplotlib<3.1.3" \
    photutils\
    python-casacore \
    reproject \
    scikit-image \
    scipy \
    seqfile && \
    pip3 install "numpy>=1.16.0" && \
    python3 setup.py build && python3 setup.py install && \
    rm -rf /root/.cache

# Use entrypoint script to create a user on the fly and avoid running as root.
RUN chmod +x /rascil/entrypoint.sh
ENTRYPOINT ["/rascil/entrypoint.sh"]
CMD ["/bin/bash"]
