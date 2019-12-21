
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

Installation instructions are available from the [documentation](https://timcornwell.gitlab.io/rascil/)
  
RASCIL code is hosted on [Github](https://github.com/SKA-ScienceDataProcessor/rascil)

RASCIL CI/CD occurs on  [Gitlab](https://gitlab.com/timcornwell/rascil)

The documentation builds occur in the pipeline on GitLab.  
