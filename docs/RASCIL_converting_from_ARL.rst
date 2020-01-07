
.. Converting

.. toctree::
   :maxdepth: 3

Converting from ARL to RASCIL
*****************************

+ The top level environment variable ARL is now RASCIL. PYTHONPATH should be set to include RASCIL::

    export RASCIL=/path/to/rascil
    export PYTHONPATH=$RASCIL:$PYTHONPATH

+ processing_library and processing_components have been combined

+ There is now a top level package called rascil. The source directories data_models, processing_components, and workflows have been moved to that directory.

+ Imports now only have to be to the e.g. rascil.processing_components level but deeper imports will still work.

+ The wrappers have been removed. For example, wrappers.arlexecute.imaging.primary_beams is now simply processing_components.imaging.primary_beams

+ arlexecute is now named rsexecute but remains exactly the same in function.

+ The name ARL has been retained when used in links to external documents.