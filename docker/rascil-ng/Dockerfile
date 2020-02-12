FROM timcornwell/rascil-full

LABEL author="Tim Cornwell <realtimcornwell@gmail.com>" \
      description="RASCIL eMERLIN reference image" \
      license="Apache2.0"

RUN apt-get update -y && apt-get install -y \
    ca-certificates \
    pybind11-dev && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

WORKDIR /rascil

RUN git clone https://gitlab.mpcdf.mpg.de/ift/nifty_gridder.git

RUN cd nifty_gridder && \
    pip3 install --user .
