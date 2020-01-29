FROM daskdev/dask

LABEL author="Tim Cornwell <realtimcornwell@gmail.com>" \
      description="RASCIL reference image" \
      license="Apache2.0"

# Install required system packages.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    gosu && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# This will clone into (and create) a directory at /rascil
RUN git clone  -b docker-original https://github.com/SKA-ScienceDataProcessor/rascil.git

# Set runtime environment variables.
ENV PYTHONPATH=/rascil:$PYTHONPATH
ENV RASCIL=/rascil

# Change to the RASCIL root directory to run the commands below.
WORKDIR /rascil

RUN mkdir -p /rascil/dask-worker-space && \
    chmod 777 /rascil/dask-worker-space

RUN rm -rf /rascil/data

# Install Python package dependencies,
# run setup (may not be required, as we've added /rascil to the PYTHONPATH)
# and clear pip cache. Use a single RUN statement to avoid multiple layers.
#
# Perhaps think about using a requirements-minimal.txt instead if possible?
RUN pip install -r docker/rascil-no-data/requirements.txt && \
    python setup.py build && python setup.py install && \
    rm -rf /root/.cache

# Use entrypoint script to create a user on the fly and avoid running as root.
RUN chmod +x /rascil/entrypoint.sh
ENTRYPOINT ["/rascil/entrypoint.sh"]
CMD ["/bin/bash"]