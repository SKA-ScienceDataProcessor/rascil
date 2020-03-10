
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

Documentation is at https://timcornwell.gitlab.io/rascil/

RASCIL can be installed using pip::

    pip install rascil
    
You may also need the rascil data for simulations:

    curl https://timcornwell.gitlab.io/rascil/rascil_data.tgz -o rascil.data.tgz
    tar zxf rascil_data.tgz
    export RASCIL_DATA=`pwd`/rascil_data

Alternatively, if you wish to develop using RASCIL then you can 
clone from the GitHub repository:
 
    git clone https://github.com/SKA-ScienceDataProcessor/rascil
    
RASCIL CI/CD occurs on  [Gitlab](https://gitlab.com/timcornwell/rascil)
