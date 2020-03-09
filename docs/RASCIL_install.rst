.. _RASCIL_install:

Installation
============

RASCIL can be run on a Linux or macos machine or cluster of machines. At least 16GB physical memory is necessary to run the full test suite. In general more memory is better. RASCIL uses Dask for multi-processing and can make good use of multi-core and multi-node machines.

Installation via pip
++++++++++++++++++++

If you just wish to run the package and do not intend to run simulations or tests, RASCIL can be installed using pip::

    pip install rascil

For simulations, you can add the data in a separate step::

    curl https://timcornwell.gitlab.io/rascil/rascil_data.tgz -o rascil.data.tgz
    tar zxf rascil_data.tgz
    export RASCIL_DATA=`pwd`/rascil_data

For tests, use one of the steps below.

Installation via docker
+++++++++++++++++++++++

If you are familar with docker, an easy approach is to use docker:

 .. toctree::
    installation/RASCIL_docker


Installation via git clone
++++++++++++++++++++++++++

Use of git clone is necessary if you wish to develop and possibly contribute RASCIL code. Installation should be straightforward. We strongly recommend the use of a python virtual environment.

RASCIL requires python 3.6 or 3.7. It has not yet been tested for 3.8.

The installation steps are:

- Use git to make a local clone of the Github respository::

   git clone https://github.com/SKA-ScienceDataProcessor/rascil

- Change into that directory::

   cd rascil

- Use pip to install required python packages::

   pip install pip --upgrade
   pip install -r requirements.txt

- Setup RASCIL::

   python setup.py install

- RASCIL makes use of a number of data files. These can be downloaded using Git LFS::

    pip install git-lfs
    git lfs install
    git-lfs pull

- Put the following definitions in your .bashrc::

    export RASCIL=/path/to/rascil
    export PYTHONPATH=$RASCIL:$PYTHONPATH

"python setup.py install" installs an egg in the correct site-packages location so the definition of PYTHONPATH is not needed
if you only don't intend to update or edit rascil in place. If you do intend to make changes, you will need the
definition of PYTHONPATH.

Installation via conda
++++++++++++++++++++++

An alternative to the use of pip in the above sequence is to use Anaconda https://www.anaconda.com. The environment is defined in the conda environment file environment.yml::

   conda env create -f environment.yml
   conda activate rascil
   conda config --env --prepend channels astropy


Installation on specific machines
+++++++++++++++++++++++++++++++++

.. toctree::
   :maxdepth: 2

   installation/RASCIL_CSD3_install
   installation/RASCIL_P3_install

Trouble-shooting
++++++++++++++++

Check your installation by running a subset of the tests::

   pip install pytest pytest-xdist
   py.test -n 4 tests/processing_components

Or the full set::

   py.test -n 4 tests

- Ensure that pip is up-to-date. If not, some strange install errors may occur.
- Check that the contents of the data directories have plausible contents. If gif-lfs has not been run successfully then the data files will just containe meta data, leading to strange run-time errors.
- There may be some dependencies that require either conda (or brew install on a mac).



.. _feedback: mailto:realtimcornwell@gmail.com
