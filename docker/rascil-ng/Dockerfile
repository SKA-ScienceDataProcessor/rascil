FROM timcornwell/rascil-no-data

LABEL author="Tim Cornwell <realtimcornwell@gmail.com>" \
      description="RASCIL eMERLIN reference image" \
      license="Apache2.0"

WORKDIR /rascil

RUN git clone https://gitlab.mpcdf.mpg.de/ift/nifty_gridder.git .

RUN cd nifty_gridder && \
    pip3 install --user .
