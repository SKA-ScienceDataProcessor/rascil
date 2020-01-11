
SCs = dprepb_rsexecute_pipeline.py imaging_sim2.py

RSTs=  $(SCs:.py=.rst)

.PHONY: rst
rst: $(RSTs)

%.rst: %.py
	echo $< > $@
	echo $< | sed -e 's/./=/g' >> $@
	echo " " >> $@
	echo ".. code:: ipython3" >> $@
	echo " " >> $@
	sed  's/^/     /'  $< >> $@

%.html: %.rst
	${RST2HTML} $(<:.rstw=.rst)  > $@

.PHONY: clean
clean:
	rm -f ${RSTs}
