FROM timcornwell/rascil-full

LABEL author="Tim Cornwell <realtimcornwell@gmail.com>" \
      description="RASCIL reference image" \
      license="Apache2.0"

WORKDIR /rascil

RUN apt-get update -y && apt-get install -y \
    curl && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# We need to install nodejs for the subsequent jupyter lab features
RUN curl -sL https://deb.nodesource.com/setup_13.x | bash - && \
    apt-get install -y nodejs

RUN pip3 install dask-labextension \
    && jupyter labextension install @jupyter-widgets/jupyterlab-manager dask-labextension@1.0.1

