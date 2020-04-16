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


RUN cd /rascil && \
    git config user.email "realtimcornwell@gmail.com" && \
    git stash && \
    git checkout py-wtowers && \
    git pull && \
    python3 setup.py build && python3 setup.py install && \
    git branch -a

WORKDIR /

RUN git clone https://gitlab.com/ska-telescope/py-wtowers.git

ENV CFLAGS="-I/usr/lib/x86_64-linux-gnu/hdf5/serial/include"
ENV LDFLAGS="-L/usr/lib/x86_64-linux-gnu/hdf5/serial/"

RUN cd py-wtowers && \
    python3 setup.py build && \
    python3 setup.py install

# Use entrypoint script to create a user on the fly and avoid running as root.
RUN chmod +x /rascil/entrypoint.sh
ENTRYPOINT ["/rascil/entrypoint.sh"]
CMD ["/bin/bash"]
