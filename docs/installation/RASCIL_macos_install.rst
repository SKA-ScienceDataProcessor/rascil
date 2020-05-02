.. _RASCIL_macos_install:

Installation of RASCIL on macos
===============================

RASCIL is well-suited to running under macos. Installation should be straightforward. Although the pip approach can
be used, we recommend use of Anaconda https://www.anaconda.com. Using anaconda::

    git clone https://github.com/SKA-ScienceDataProcessor/rascil
    cd rascil
    conda env create -f environment.yml
    conda activate rascil
    conda config --env --append channels conda-forge

Then at the top level directory, do::

    pip install -e .

This will install it as a development package (this adds it into the path in situ).

Finally to get the casa measures data::

    rsync -avz rsync://casa-rsync.nrao.edu/casa-data/geodetic /opt/anaconda/envs/rascil/lib/casa/data/

Or if your anaconda is in your home directory::

        rsync -avz rsync://casa-rsync.nrao.edu/casa-data/geodetic ~/opt/anaconda/envs/rascil/lib/casa/data/


Finally, put the following definitions in your .bashrc::

    export RASCIL=/path/to/rascil
    export PYTHONPATH=$RASCIL:$PYTHONPATH


.. _feedback: mailto:realtimcornwell@gmail.com
