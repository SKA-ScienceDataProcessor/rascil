.. _RASCIL_GALAHAD_install:

Installation of RASCIL on galahad
=================================

RASCIL is well-suited to running on galahad. Installation should be straightforward.
We strongly recommend the use of a python virtual environment. Be sure to load the
bare python3.7base module and the gcc920 modules before installing RASCIL::

    module load python37base gcc920

Follow the generic installation steps.

We recommend that RASCIL be installed on one of the preferred storage
systems e.g. /share/nas/<your-login>/rascil/

If you are using singularity containers, you will probably need to put the
singularity cache somewhere other than your home directory::

    mkdir /share/nas/<yourname>/.singularity
    export SINGULARITY_CACHEDIR=/share/nas/<yourname>/.singularity
    singularity pull RASCIL-full.img docker://timcornwell/rascil-full-no-root

.. _feedback: mailto:realtimcornwell@gmail.com
