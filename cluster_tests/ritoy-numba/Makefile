
PYFILE        = cluster_test_ritoy_numba.py

.PHONY: clean
clean:
	\rm -rf *.log slurm*.out dask-worker-space hostfile.*

.PHONY: test
test: ${PYFILE}
	python ${PYFILE}





