.. Documentation master

.. toctree::

Radio Astronomy Simulation, Calibration and Imaging Library
###########################################################

The Radio Astronomy Simulation, Calibration and Imaging Library expresses radio interferometry calibration and
imaging algorithms in python and numpy. The interfaces all operate with familiar data
structures such as image, visibility table, gaintable, etc. The python source code is directly accessible from these
documentation pages: see the source link in the top right corner.

To acheive sufficient performance we take a dual pronged approach - using threaded libraries for shared memory
processing, and the `Dask <https:/www.dask.org>`_ library for distributed processing.

RASCIL replaces the Algorithm Reference Library (ARL), which is now frozen.

.. toctree::
   :maxdepth: 2

   RASCIL_install
   RASCIL_notebooks
   RASCIL_directory
   RASCIL_api
   RASCIL_otherinfo

* :ref:`genindex`
* :ref:`modindex`

.. _feedback: mailto:realtimcornwell@gmail.com
