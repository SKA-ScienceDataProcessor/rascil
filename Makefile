# simple makefile to simplify repetitive build env management tasks under posix
PYTHON ?= python3
PYLINT ?= pylint
MAKE_DBG ?= ""
TESTS ?= tests/
FLAKE ?= flake8
NAME = rascil
IMG ?= $(NAME)
TAG ?= ubuntu18.04
DOCKER_IMAGE = $(IMG):$(TAG)
WORKER_RASCIL_DATA ?= /rascil/data
CURRENT_DIR = $(shell pwd)
JUPYTER_PASSWORD ?= changeme

# RASCIL data directory usualy found in ./data
RASCILDATA = $(CURRENT_DIR)/data

-include PrivateRules.mak

.DEFAULT_GOAL := help

clean: cleantests
	$(PYTHON) setup.py clean --all
	rm -rf dist

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build >/dev/null 2>&1 && $(PYTHON) setup.py install >/dev/null 2>&1 || (echo "'$(PYTHON) setup.py install' failed."; exit -1)

build: in  ## build and install this project - make sure pipenv shell is activated

cleantests: ## clean out the cache before tests are run
	rm -rf workers-*.dirlock
	cd tests && rm -rf __pycache__

unittest: cleantests  ## run tests using unittest
	MPLBACKEND=agg $(PYTHON) -m unittest -f --locals tests/*/test_*.py

pytest: cleantests  ## run tests using pytest
	pip install pytest >/dev/null 2>&1
	pytest -x $(TESTS)

trailing-spaces:
	find rascil/processing_components -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;
	find rascil/workflows -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;

docs: inplace  ## build docs
	$(MAKE) -C docs dirhtml

code-flake:
	# flake8 ignore long lines and trailing whitespace
	$(FLAKE) --ignore=E501,W293,F401 --builtins=ModuleNotFoundError rascil/processing_components rascil/workflows

code-lint:
	$(PYLINT) --extension-pkg-whitelist=numpy \
	  --ignored-classes=astropy.units,astropy.constants,HDUList \
	  -E rascil/processing_components rascil/workflows tests/

code-analysis: code-flake code-lint  ## run pylint and flake8 checks

examples: inplace  ## launch examples
	$(MAKE) -C rascil/examples/notebooks
	$(MAKE) -C rascil/examples/scripts
	$(MAKE) -C rascil/examples/skasimulations

docker:
	cd docker/rascil-no-data;docker build -t timcornwell/rascil-no-data .
	cd docker/rascil-full;docker build -t timcornwell/rascil-full .

docker-no-cache:
	cd docker/rascil-no-data;docker build --no-cache -t timcornwell/rascil-no-data .
	cd docker/rascil-full;docker build --no-cache -t timcornwell/rascil-full .

docker-push:
	docker push timcornwell/rascil-no-data
	docker push timcornwell/rascil-full