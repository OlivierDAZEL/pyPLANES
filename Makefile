clean:
	rm -f *.log *.pdf *.ilg *.idx *.aux *.fls *.synctex.gz *.out *.fdb_latexmk

sphinx:
	rm docs/rst/*.rst
	sphinx-apidoc -o docs/rst/ pyPLANES/ 
	cp docs/rst/modules.rst docs/rst/index.rst
	sphinx-build -b html docs/rst docs/html

