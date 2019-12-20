
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

Installation instructions are available from the documentation pages <https://timcornwell.gitlab.io/rascil/

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
