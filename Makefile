BLK         =   black
MAKE        =   make
PIP         =   pip

BASEDIR     =   $(CURDIR)
SRCDIR      =   ${BASEDIR}/scatfit

help:
	@echo 'Makefile for scatfit'
	@echo 'Usage:'
	@echo 'make black           reformat the code using black code formatter'
	@echo 'make build           generate distribution archives'
	@echo 'make clean           remove temporary files'
	@echo 'make install         install the module locally'
	@echo 'make profile         profile the code'
	@echo 'make test            run the regression tests'
	@echo 'make uninstall       uninstall the package'
	@echo 'make upload          upload the distribution to PyPI'
	@echo 'make uploadtest      upload the distribution to TestPyPI'

black:
	${BLK} *.py */*.py */*/*.py

build:
	python3 -m build

clean:
	rm -f ${SRCDIR}/*.pyc
	rm -f ${SRCDIR}/apps/*.pyc
	rm -rf ${SRCDIR}/__pycache__
	rm -rf ${SRCDIR}/apps/__pycache__
	rm -rf ${BASEDIR}/build
	rm -rf ${BASEDIR}/dist
	rm -rf ${BASEDIR}/scatfit.egg-info

install:
	${MAKE} clean
	${MAKE} uninstall
	${PIP} install .
	${MAKE} clean

profile:
	python3 -m cProfile -s tottime scatfit/apps/fit_frb.py extra/fake_burst_500_DM.fil 500.0 --smodel scattered_isotropic_bandintegrated --fscrunch 1024 --fast

test:
	nose2

uninstall:
	${PIP} uninstall scatfit

upload:
	${MAKE} clean
	${MAKE} build
	python3 -m twine upload dist/*

uploadtest:
	${MAKE} clean
	${MAKE} build
	python3 -m twine upload --repository testpypi dist/*

.PHONY: help black build clean install profile test uninstall upload uploadtest