# Universal image for running Notebook, Dask pipelines, libs, and lint checkers
ARG BASE_IMAGE=daskdev/dask
FROM $BASE_IMAGE
ARG PYTHON=python3
ARG PIP=pip3

LABEL \
      author="Tim Cornwell <realtimcornwell@gmail.com>" \
      description="RASCIL reference image" \
      license="Apache2.0" \
      registry="library/timcornwell/rascil" \
      vendor="Catalyst" \
      org.skatelescope.team="Systems Team" \
      org.skatelescope.version="0.1.0" \
      org.skatelescope.website="http://github.com/SKA-ScienceDataProcessor/rascil/"

ENV PATH=/rascil/rascil_env/bin:$PATH \
    VIRTUAL_ENV=/rascil/rascil_env \
    HOME=/root \
    DEBIAN_FRONTEND=noninteractive

# the package basics for Python 3
RUN \
    apt-get update -y && \
    apt-get install -y software-properties-common pkg-config dirmngr \
            python3-software-properties build-essential curl wget fonts-liberation ca-certificates libcfitsio-dev libffi-dev && \
    add-apt-repository -y ppa:git-core/ppa && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 6B05F25D762E3157 && \
    apt-get install -y git-lfs && \
    git lfs install && \
    apt-get install -y $PYTHON-dev $PYTHON-tk flake8 $PYTHON-nose \
            virtualenv virtualenvwrapper && \
    apt-get install -y graphviz && \
    apt-get install -y nodejs npm && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN git clone -b feature_docker https://github.com/SKA-ScienceDataProcessor/rascil.git

RUN ls /rascil

# runtime specific environment
ENV PYTHONPATH /rascil:/rascil/rascil_env
ENV RASCIL /rascil

RUN touch "${HOME}/.bash_profile"

# run setup
RUN \
    cd /rascil && \
    conda env create -f environment.yml && \
    conda activate rascil_env && \
    conda config --env --prepend channels astropy && \
    $PYTHON setup.py build && \
    $PYTHON setup.py install

# create space for libs
RUN mkdir -p /rascil/test_results && \
    chmod 777 /rascil /rascil/test_results

# We share in the rascil data here
#VOLUME ["/rascil/data", "/rascil/tmp"]

# Use entrypoint script to create a user on the fly and avoid running as root.
COPY build/lib/rascil/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]