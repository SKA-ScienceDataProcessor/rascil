FROM timcornwell/rascil-no-data-no-root

LABEL author="Tim Cornwell <realtimcornwell@gmail.com>" \
      description="RASCIL reference image" \
      license="Apache2.0"

WORKDIR /rascil

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    git-lfs && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN git lfs install && \
    python3 setup.py build && python3 setup.py install && \
    rm -rf /root/.cache

WORKDIR /rascil

# git-lfs is already loaded so we just need to install it
RUN git lfs pull

CMD ["/bin/bash"]

