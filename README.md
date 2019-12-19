
Radio Astronomy Simulation, Calibration and Imaging Library
===========================================================

The Radio Astronomy Simulation, Calibration and Imaging Library
expresses radio interferometry calibration and imaging algorithms in
python and numpy. The interfaces all operate with familiar data structures
such as image, visibility table, gaintable, etc. The python source
code is directly accessible from these documentation pages: see the
source link in the top right corner.

To acheive sufficient performance we take a dual pronged approach -
using threaded libraries for shared memory processing, and the Dask
library for distributed processing.

Installing
----------

The RASCIL has a few dependencies:
* Python 3.6+
* git 2.11.0+
* git-lfs 2.2.1+
* Python package dependencies as defined in the file requirements.txt

Common Installation Process
---------------------------

* Use git to make a local clone of the Github repository::

   `git clone https://github.com/SKA-ScienceDataProcessor/rascil`

* Change into that directory::
   `cd rascil`

* Install required python package::
   `pip install -r requirements.txt`

* Setup RASCIL::
   `python setup.py install`

* Get the data files form Git LFS::
   `git-lfs pull`

* Finally fix up the python search path so that Jupyter can find the arl with something like::

    `export PYTHONPATH=$PYTHONPATH:/path/to/checked/out/rascil`
    `export RASCIL=/path/to/checked/out/rascil`

An alternative to the use of pip is to use [Anaconda](<https://www.anaconda.com>). The environment is defined in the conda environment file::

   conda env create -f environment.yml 
   conda activate rascil
   conda config --env --prepend channels astropy

Orientation
-----------

Here is a quick guide to the layout:

  * `rascil/data_models`: Data models such as Image, Visibility, GainTable
  * `rascil/processing_library`: Algorithm independent library code
  * `rascil/processing_components`: Processing functions used in algorithms
  * `rascil/workflows`: Serial and distributed processing workflows
  * `examples`: Example scripts and notebooks
  * `tests`: Unit and regression tests
  * `docs`: Complete documentation. Includes non-interactive output of examples.
  * `data`: Data used
  * `tools`: package requirements, and [Docker](https://www.docker.com/) image building recipe

Running Notebooks
-----------------

[Jupyter notebooks](http://jupyter.org/) end with `.ipynb` and can be run as follows from the
command line:

     $ jupyter-notebook rascil/examples/notebooks/imaging_serial.ipynb
     
Testing and deployment
----------------------

  * RASCIL code is hosted on [Github](https://github.com/SKA-ScienceDataProcessor/rascil)

  * RASCIL CI/CD occurs on  [Gitlab](https://gitlab.com/timcornwell/rascil)

  * The unittests and documentation builds occur in the pipeline on GitLab. 

  * The last successful build documentation is at https://timcornwell.gitlab.io/rascil/
    
  * The unittest coverage is displayed at https://timcornwell.gitlab.io/rascil/coverage
    
For building the documentation you will need Sphinx as well as
Pandoc. This will extract docstrings from the rascil source code,
evaluate all notebooks and compose the result to form the
documentation package.

You can build it as follows:

    $ make -C docs [format]

Omit [format] to view a list of documentation formats that Sphinx can
generate for you. You probably want dirhtml.


Running Tests
=============

Test and code analysis requires nosetests3 and flake8 to be installed.


Platform Specific Instructions
------------------------------

Ubuntu 17.04+
-------------

install flake8, nose, and pylint:
```
sudo apt-get install flake8 python3-nose pylint3
```

Running the Tests
-----------------

All unit tests can be run with:
```
make unittest
```
or nose:
```
make nosetests
```
or pytest:
```
make pytest
```

Code analysis can be run with:
```
make code-analysis
```
