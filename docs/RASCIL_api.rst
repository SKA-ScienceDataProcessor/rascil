.. _api:

API
===

Here is a quick guide to the layout of the package:

 - `rascil/data_models`: Data models such as Image, Visibility, GainTable
 - `rascil/processing_library`: Algorithm independent library code
 - `rascil/processing_components`: Processing functions used in algorithms
 - `rascil/workflows`: Serial and distributed processing workflows
 - `examples`: Example scripts and notebooks
 - `tests`: Unit and regression tests
 - `docs`: Complete documentation. Includes non-interactive output of examples.
 - `data`: Data used
 - `tools`: package requirements, and [Docker](https://www.docker.com/) image building recipe

The API is specified in the rascii directory.

.. toctree::
   :maxdepth: 1

   data_models/index.rst
   processing_components/index.rst
   processing_library/index.rst
   workflows/index.rst
   wrappers/index.rst

* :ref:`genindex`
* :ref:`modindex`

.. _feedback: mailto:realtimcornwell@gmail.com
