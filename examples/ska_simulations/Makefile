
.SUFFIXES:

SCs = pointing_simulation.py surface_simulation.py

RSTs=  $(SCs:.py=.rst)

.PHONY: rst
rst: $(RSTs)

%.rst: %.py
	cp $@.header $@
	echo "\n.. code:: python\n" >> $@
	sed  's/^/     /'  $< >> $@
	echo "\nThe shell script to run is:\n" >> $@
	echo "\n.. code:: sh\n" >> $@
	sed  's/^/     /'  $<.sh >> $@
	echo "\nThe SLURM batch file is:\n" >> $@
	echo "\n.. code:: sh\n" >> $@
	sed  's/^/     /'  $<.slurm >> $@
	echo "\n" >> $@

%.html: %.rst
	${RST2HTML} $(<:.rstw=.rst)  > $@

.PHONY: clean
clean:
	rm -f ${RSTs}
