FROM timcornwell/rascil-full

LABEL author="Tim Cornwell <realtimcornwell@gmail.com>" \
      description="RASCIL py-wtowers reference image" \
      license="Apache2.0"

RUN apt-get update -y && apt-get install -y \
    ca-certificates \
    libhdf5-dev \
    libfftw3-dev && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

WORKDIR /rascil

RUN git clone https://gitlab.com/ska-telescope/py-wtowers.git

RUN ls /usr/lib/x86_64-linux-gnu/hdf5/serial/

RUN HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/serial/ pip3 install h5py

RUN cd py-wtowers && \
    HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/serial/ python3 setup.py build && \
    HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/serial/ python3 setup.py install
