FROM timcornwell/rascil-no-data

LABEL author="Tim Cornwell <realtimcornwell@gmail.com>" \
      description="RASCIL reference image" \
      license="Apache2.0"

WORKDIR /rascil

# git-lfs is already loaded so we just need to install it
RUN git lfs install && \
    git checkout /rascil
