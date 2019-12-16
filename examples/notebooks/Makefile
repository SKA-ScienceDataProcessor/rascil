
NBs = bandpass-calibration_rsexecute.ipynb imaging-pipelines_serial.ipynb mosaic_serial.ipynb  \
 queue.ipynb imaging-fits_rsexecute.ipynb  imaging-wterm_rsexecute.ipynb \
 mpc_simulation.ipynb simple-dask_rsexecute.ipynb imaging-pipelines_rsexecute.ipynb  imaging.ipynb

RSTs=  $(NBs:.ipynb=.rst)

JUPYTER ?= jupyter
TIMEOUT = 3600

.PHONY: rst
rst: $(RSTs)

%.rst: %.ipynb
	jupyter nbconvert --execute --ExecutePreprocessor.timeout=$(TIMEOUT) --to rst $<

%.html: %.rst
	${RST2HTML} $(<:.rstw=.rst)  > $@

.PHONY: clean
clean:
	rm -f ${RSTs}
