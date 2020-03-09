FROM ubuntu:18.04

LABEL author="Tim Cornwell <realtimcornwell@gmail.com>" \
      description="RASCIL reference image based on PyPI" \
      license="Apache2.0"

# Install required system packages.
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y && apt-get install -y \
    ca-certificates \
    gosu git curl \
    libxml2-dev liblapack-dev libcfitsio-dev libgtkmm-3.0-dev \
    cmake g++ build-essential gfortran libpng-dev casacore-dev \
    python3 python3-dev python3-pip python3-setuptools && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Set runtime environment variables.
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN pip3 install Cython setuptools && \
    pip3 install "numpy>=1.16.0" && \
    pip3 install rascil && \
    rm -rf /root/.cache

CMD ["/bin/bash"]
