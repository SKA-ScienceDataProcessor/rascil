FROM timcornwell/rascil-no-data

LABEL author="Tim Cornwell <realtimcornwell@gmail.com>" \
      description="RASCIL eMERLIN reference image" \
      license="Apache2.0"

WORKDIR /rascil

# Install required system packages.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    aoflagger && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
