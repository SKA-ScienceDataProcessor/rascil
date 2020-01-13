
SCs = dprepb_rsexecute_pipeline.py imaging_sim2.py imaging.py

RSTs=  $(SCs:.py=.rst)

.PHONY: rst
rst: $(RSTs)

%.rst: %.py
	cat $@.header > $@
	sed  's/^/     /'  $< >> $@

%.html: %.rst
	${RST2HTML} $(<:.rstw=.rst)  > $@

.PHONY: clean
clean:
	rm -f ${RSTs}
