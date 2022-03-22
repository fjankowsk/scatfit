BLK         =   black
PIP         =   pip

BASEDIR     =   $(CURDIR)
SRCDIR      =   ${BASEDIR}/scatfit

help:
	@echo 'Makefile for scatfit'
	@echo 'Usage:'
	@echo 'make black           reformat the code using black code formatter'
	@echo 'make clean           remove temporary files'
	@echo 'make install         install the module locally'

black:
	${BLK} *.py */*.py */*/*.py

clean:
	rm -f ${SRCDIR}/*.pyc
	rm -f ${SRCDIR}/apps/*.pyc
	rm -rf ${SRCDIR}/__pycache__
	rm -rf ${SRCDIR}/apps/__pycache__
	rm -rf ${BASEDIR}/build
	rm -rf ${BASEDIR}/dist
	rm -rf ${BASEDIR}/scatfit.egg-info

install:
	${PIP} install .

.PHONY: help black clean install