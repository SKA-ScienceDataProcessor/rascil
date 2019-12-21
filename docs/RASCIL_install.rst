.. installation

Installation
============

RASCIL can be run on a Linux or macos machine or cluster of machines. At least 16GB physical memory is necessary to run the full test suite. In general more memory is better. RASCIL uses Dask for multi-processing and can make good use of multi-core and multi-node machines.

Installation should be straightforward. We strongly recommend the use of a python virtual environment.

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
- An alternative to the use of pip is to use Anaconda https://www.anaconda.com. The environment is defined in the conda environment file environment.yml::

   conda env create -f environment.yml
   conda activate rascil
   conda config --env --prepend channels astropy

* :ref:`genindex`
* :ref:`modindex`

.. _feedback: mailto:realtimcornwell@gmail.com
