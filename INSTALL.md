Platform Specific Instructions
------------------------------

Ubuntu 17.04+
-------------

install required packages for git-lfs:
```
sudo apt-get install software-properties-common python-software-properties build-essential curl
sudo add-apt-repository ppa:git-core/ppa
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

# the following will update your ~/.gitconfig with the command filter for lfs
git lfs install
```

install the required packages for python3.6:
```
sudo apt-get install python3.6-dev python3-pip python3-tk virtualenv virtualenvwrapper

# note that pip will be install as pip3 and python will be python3
# so be mindful to make the correct substitutions below

# Optional for simple Dask test notebook - simple-dask.ipynb
sudo apt-get install graphviz
```

Optional: for those wishing to use pipenv and virtualenv
```
sudo pip3 install pipenv
virtualenv -p python3.6 /path/to/where/you/want/to/keep/arlvenv

# activate the virtualenv with:
. /path/to/where/you/want/to/keep/arlvenv/bin/activate

# optionally install the Bokeh server to enable the Dask diagnostics web interface
pip install bokeh

# if you want to use pytest
pip install pytest

# if you want to do the lint checking
pip install pylint

# permanently fix up the RASCIL lib path in the virtualenv with the following:
add2virtualenv /path/to/checked/out/rascil

# this updates arlvenv/lib/python3.x/site-packages/_virtualenv_path_extensions.pth

# to turn off/deactivate the virtualenv with:
deactivate
```

Using pipenv only
-----------------

Pipenv/pipfile now provides a complete integrated virtual requirement
and tracking which seem to be effective. You can run it simply as:

```
pipenv shell
```

And this will leave you in a shell with the environment fully setup.
